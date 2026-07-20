#!/usr/bin/env python3
"""Validate a scorer-free checkpoint generation run before allocating GPUs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def nested_value(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def model_weight_bytes(model_path: Path) -> int:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        metadata = json.loads(index_path.read_text(encoding="utf-8")).get("metadata", {})
        total_size = metadata.get("total_size")
        if total_size is not None:
            return int(total_size)

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors weights found under {model_path}")
    return sum(path.stat().st_size for path in shards)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--ground-truth-field", default="reward_model.ground_truth")
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, required=True)
    parser.add_argument("--enable-thinking", type=parse_bool, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--expected-response-length", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--physical-vram-gib", type=float, required=True)
    parser.add_argument("--runtime-reserve-gib", type=float, required=True)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {args.model_path / 'config.json'}")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Generation dataset not found: {args.data_path}")
    if args.expected_prompts <= 0 or args.samples_per_prompt <= 0 or args.generation_batch_size <= 0:
        raise ValueError("Prompt count, samples per prompt, and generation batch size must be positive")
    if args.num_gpus <= 0 or args.tensor_parallel_size <= 0:
        raise ValueError("GPU count and tensor parallel size must be positive")
    if args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError("The total GPU count must be divisible by tensor parallel size")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if args.expected_response_length > args.max_response_length:
        raise ValueError("expected_response_length cannot exceed max_response_length")
    if args.max_model_len < args.max_prompt_length + args.max_response_length:
        raise ValueError("max_model_len must fit max_prompt_length + max_response_length")

    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    rows = pq.read_table(args.data_path).to_pylist()
    if len(rows) != args.expected_prompts:
        raise ValueError(f"Expected {args.expected_prompts} prompts, found {len(rows)}")
    if any(args.prompt_key not in row or not row[args.prompt_key] for row in rows):
        raise ValueError(f"Every row must contain a non-empty {args.prompt_key!r}")
    if any(
        nested_value(row, args.ground_truth_field) is None
        or nested_value(row, args.ground_truth_field) == ""
        for row in rows
    ):
        raise ValueError(f"Every row must contain {args.ground_truth_field!r} for offline rescoring")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompt_lengths = []
    for row in rows:
        prompt = row[args.prompt_key]
        if isinstance(prompt, str):
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        else:
            token_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        prompt_lengths.append(len(token_ids))

    observed_max_prompt = max(prompt_lengths)
    if observed_max_prompt > args.max_prompt_length:
        raise ValueError(
            f"Observed prompt length {observed_max_prompt} exceeds configured maximum {args.max_prompt_length}"
        )

    num_layers = int(config.num_hidden_layers)
    num_attention_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // num_attention_heads))
    kv_heads_per_gpu = math.ceil(num_kv_heads / args.tensor_parallel_size)
    kv_bytes_per_token = 2 * num_layers * kv_heads_per_gpu * head_dim * args.kv_dtype_bytes

    expected_sequence_gib = (
        (args.max_prompt_length + args.expected_response_length) * kv_bytes_per_token / 1024**3
    )
    worst_sequence_gib = (
        (args.max_prompt_length + args.max_response_length) * kv_bytes_per_token / 1024**3
    )

    weights_total_gib = model_weight_bytes(args.model_path) / 1024**3
    weights_per_gpu_gib = weights_total_gib / args.tensor_parallel_size
    vllm_budget_gib = args.physical_vram_gib * args.gpu_memory_utilization
    usable_kv_gib = vllm_budget_gib - weights_per_gpu_gib - args.runtime_reserve_gib
    if usable_kv_gib <= 0:
        raise ValueError("No estimated KV capacity remains after weights and runtime reserve")
    if worst_sequence_gib > usable_kv_gib:
        raise ValueError("A single maximum-length sequence does not fit the estimated KV budget")

    expected_capacity = math.floor(usable_kv_gib / expected_sequence_gib)
    worst_capacity = math.floor(usable_kv_gib / worst_sequence_gib)
    replicas = args.num_gpus // args.tensor_parallel_size
    cluster_active_capacity = replicas * args.max_num_seqs
    prompts_per_batch = min(args.generation_batch_size, args.expected_prompts)
    queued_per_batch = prompts_per_batch * args.samples_per_prompt
    warnings = []
    if args.max_num_seqs > expected_capacity:
        warnings.append(
            f"max_num_seqs={args.max_num_seqs} exceeds expected-length KV capacity {expected_capacity}"
        )
    if args.max_num_seqs > worst_capacity:
        warnings.append(
            f"max_num_seqs={args.max_num_seqs} exceeds all-max-length KV capacity {worst_capacity}; "
            "effective resident concurrency must fall as responses grow"
        )

    report = {
        "status": "pass_with_warnings" if warnings else "pass",
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "prompt_key": args.prompt_key,
        "ground_truth_field": args.ground_truth_field,
        "expected_prompts": args.expected_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "expected_output_rows": args.expected_prompts * args.samples_per_prompt,
        "generation_batch_size": args.generation_batch_size,
        "queued_responses_per_generation_batch": queued_per_batch,
        "enable_thinking": args.enable_thinking,
        "max_prompt_length": args.max_prompt_length,
        "observed_max_prompt_length": observed_max_prompt,
        "max_response_length": args.max_response_length,
        "expected_response_length": args.expected_response_length,
        "max_model_len": args.max_model_len,
        "model_structure": {
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "parallelism": {
            "num_gpus": args.num_gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
            "replicas": replicas,
            "max_num_seqs_per_replica": args.max_num_seqs,
            "cluster_active_sequence_capacity": cluster_active_capacity,
            "queued_to_active_ratio": queued_per_batch / cluster_active_capacity,
        },
        "memory": {
            "physical_vram_gib": args.physical_vram_gib,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "nominal_vllm_budget_gib": vllm_budget_gib,
            "model_weights_total_gib": weights_total_gib,
            "model_weights_per_gpu_gib": weights_per_gpu_gib,
            "runtime_reserve_gib": args.runtime_reserve_gib,
            "estimated_usable_kv_gib": usable_kv_gib,
            "kv_bytes_per_token_per_gpu": kv_bytes_per_token,
            "expected_kv_gib_per_sequence": expected_sequence_gib,
            "worst_kv_gib_per_sequence": worst_sequence_gib,
            "expected_length_capacity_per_gpu": expected_capacity,
            "worst_length_capacity_per_gpu": worst_capacity,
            "expected_active_kv_gib_per_gpu": expected_sequence_gib * args.max_num_seqs,
            "worst_active_kv_gib_per_gpu": worst_sequence_gib * args.max_num_seqs,
        },
        "warnings": warnings,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.output_json)

    print("========== CHECKPOINT GENERATION PREFLIGHT ==========")
    print(f"model                              : {args.model_path}")
    print(f"dataset                            : {args.data_path}")
    print(f"prompts                            : {args.expected_prompts}")
    print(f"samples per prompt                 : {args.samples_per_prompt}")
    print(f"expected JSONL rows                : {report['expected_output_rows']}")
    print(f"generation prompt batch            : {prompts_per_batch}")
    print(f"queued responses/batch             : {queued_per_batch}")
    print(f"observed/configured prompt max     : {observed_max_prompt}/{args.max_prompt_length}")
    print(f"max response length                : {args.max_response_length}")
    print(f"vLLM replicas                      : {replicas}")
    print(f"max_num_seqs/replica               : {args.max_num_seqs}")
    print(f"cluster active sequence capacity   : {cluster_active_capacity}")
    print(f"queued/active ratio                : {queued_per_batch / cluster_active_capacity:.3f}x")
    print(f"vLLM memory budget/GPU             : {vllm_budget_gib:.2f} GiB")
    print(f"expected/worst KV per sequence     : {expected_sequence_gib:.3f}/{worst_sequence_gib:.3f} GiB")
    print(f"expected/worst capacity per GPU    : {expected_capacity}/{worst_capacity}")
    for warning in warnings:
        print(f"WARNING                            : {warning}")
    print(f"report                             : {args.output_json}")
    print("=====================================================")


if __name__ == "__main__":
    main()
