#!/usr/bin/env python3
"""Run generation and offline Ye rescoring for a checkpoint-by-benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-config",
        type=Path,
        required=True,
        help="Evaluation environment JSON containing paths, Python runtimes, and hardware.",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        required=True,
        help="Validation JSON containing checkpoint identity, sampling, and vLLM parameters.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=script_dir / "benchmark_catalog.json",
        help="Benchmark catalog JSON.",
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark IDs. Defaults to every enabled catalog entry.",
    )
    parser.add_argument(
        "--steps",
        help="Comma-separated checkpoint steps. Overrides checkpoint.steps in the experiment config.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate complete existing outputs.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later matrix tasks after a task fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved matrix without requiring remote files or GPUs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a JSON object")
    return value


def resolve_from_root(root: Path, value: Any, field: str) -> Path:
    if value is None or str(value) == "":
        raise ValueError(f"{field} cannot be empty")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def resolve_evaluation_config(
    environment: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    if int(environment.get("schema_version", 0)) != 1:
        raise ValueError("Environment config schema_version must be 1")
    if int(validation.get("schema_version", 0)) != 1:
        raise ValueError("Validation config schema_version must be 1")

    environment_keys = {
        "schema_version",
        "environment_id",
        "workspace_root",
        "python",
        "paths",
        "hardware",
    }
    validation_keys = {
        "schema_version",
        "validation_id",
        "algorithm_id",
        "model_id",
        "training_run_id",
        "checkpoint",
        "generation",
        "rollout_runtime",
        "rescoring",
    }
    unexpected_environment = sorted(set(environment) - environment_keys)
    unexpected_validation = sorted(set(validation) - validation_keys)
    if unexpected_environment:
        raise ValueError(
            f"Environment config contains non-environment fields: {unexpected_environment}"
        )
    if unexpected_validation:
        raise ValueError(
            f"Validation config contains unsupported fields: {unexpected_validation}"
        )

    workspace_root = Path(str(environment["workspace_root"]))
    if not workspace_root.is_absolute():
        raise ValueError("environment.workspace_root must be an absolute path")

    python_config = require_mapping(environment, "python")
    path_config = require_mapping(environment, "paths")
    hardware = require_mapping(environment, "hardware")
    rollout_runtime = require_mapping(validation, "rollout_runtime")
    overlapping_runtime = sorted(set(hardware) & set(rollout_runtime))
    if overlapping_runtime:
        raise ValueError(
            f"Hardware fields must not be repeated in rollout_runtime: {overlapping_runtime}"
        )
    generation = require_mapping(validation, "generation")
    for field in ("enable_thinking", "do_sample"):
        if not isinstance(generation.get(field), bool):
            raise ValueError(f"validation.generation.{field} must be true or false")
    if generation["do_sample"] and float(generation.get("temperature", 0.0)) <= 0:
        raise ValueError("Sampled validation requires generation.temperature > 0")
    if not generation["do_sample"] and int(generation.get("n_samples", 0)) != 1:
        raise ValueError("Greedy validation requires generation.n_samples=1")
    min_p = float(generation.get("min_p", -1.0))
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("validation.generation.min_p must be between 0 and 1")

    checkpoint = dict(require_mapping(validation, "checkpoint"))
    checkpoint["root"] = str(
        resolve_from_root(workspace_root, checkpoint.get("root"), "checkpoint.root")
    )

    rescoring = require_mapping(validation, "rescoring")
    if not isinstance(rescoring.get("enabled"), bool):
        raise ValueError("validation.rescoring.enabled must be true or false")

    return {
        "schema_version": 1,
        "environment_id": require_identifier(
            environment["environment_id"],
            "environment_id",
        ),
        "validation_id": require_identifier(validation["validation_id"], "validation_id"),
        "feng_j_root": str(workspace_root),
        "verl_python": str(
            resolve_from_root(workspace_root, python_config.get("verl"), "python.verl")
        ),
        "algorithm_id": validation["algorithm_id"],
        "model_id": validation["model_id"],
        "training_run_id": validation["training_run_id"],
        "checkpoint": checkpoint,
        "generation": generation,
        "runtime": {**hardware, **rollout_runtime},
        "rescoring": {
            "enabled": rescoring["enabled"],
            "python_path": str(
                resolve_from_root(
                    workspace_root,
                    python_config.get("ye_rescore"),
                    "python.ye_rescore",
                )
            ),
            "ye_parser_path": str(
                resolve_from_root(
                    workspace_root,
                    path_config.get("ye_parser"),
                    "paths.ye_parser",
                )
            ),
        },
        "output": {
            "results_root": str(
                resolve_from_root(
                    workspace_root,
                    path_config.get("results_root"),
                    "paths.results_root",
                )
            ),
            "work_root": str(
                resolve_from_root(
                    workspace_root,
                    path_config.get("work_root"),
                    "paths.work_root",
                )
            ),
        },
    }


def require_identifier(value: Any, field: str) -> str:
    value = str(value)
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field} may contain only letters, digits, dot, underscore, and hyphen: {value!r}"
        )
    return value


def positive_int(value: Any, field: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return value


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Comma-separated selection cannot be empty")
    return values


def resolve_steps(args: argparse.Namespace, checkpoint_config: dict[str, Any]) -> list[int]:
    raw_steps: list[Any]
    if args.steps:
        raw_steps = parse_csv(args.steps)
    else:
        configured = checkpoint_config.get("steps", [])
        if not isinstance(configured, list):
            raise ValueError("checkpoint.steps must be a JSON array")
        raw_steps = configured

    steps = sorted({positive_int(step, "checkpoint step") for step in raw_steps})
    if not steps:
        raise ValueError("No checkpoint steps selected; set checkpoint.steps or pass --steps")
    return steps


def resolve_benchmarks(
    args: argparse.Namespace,
    catalog: dict[str, Any],
    *,
    require_ye_data_name: bool,
    workspace_root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    entries = require_mapping(catalog, "benchmarks")
    selected = parse_csv(args.benchmarks)
    if not selected:
        selected = [benchmark_id for benchmark_id, entry in entries.items() if entry.get("enabled")]
    if not selected:
        raise ValueError("No enabled benchmarks found in the catalog")

    resolved = []
    for benchmark_id in selected:
        require_identifier(benchmark_id, "benchmark ID")
        catalog_entry = entries.get(benchmark_id)
        if not isinstance(catalog_entry, dict):
            raise KeyError(f"Unknown benchmark: {benchmark_id}")
        entry = dict(catalog_entry)
        if not entry.get("enabled"):
            raise ValueError(
                f"Benchmark {benchmark_id!r} is disabled. Set enabled=true and provide data_path first."
            )
        if not entry.get("data_path"):
            raise ValueError(f"Benchmark {benchmark_id!r} has no data_path")
        if require_ye_data_name and not entry.get("ye_data_name"):
            raise ValueError(
                f"Benchmark {benchmark_id!r} has no ye_data_name for Ye Wenxuan rescoring"
            )
        positive_int(entry.get("expected_prompts"), f"{benchmark_id}.expected_prompts")
        positive_int(entry.get("max_prompt_length"), f"{benchmark_id}.max_prompt_length")
        entry["data_path"] = str(
            resolve_from_root(
                workspace_root,
                entry["data_path"],
                f"{benchmark_id}.data_path",
            )
        )
        resolved.append((benchmark_id, entry))
    return resolved


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def scan_generation_jsonl(
    path: Path,
    *,
    expected_prompts: int,
    samples_per_prompt: int,
    expected_metadata: dict[str, Any] | None,
) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rollout_ids: dict[str, set[int]] = defaultdict(set)
    row_count = 0
    digest = hashlib.sha256()

    with path.open("rb") as raw_handle:
        for raw_line in raw_handle:
            digest.update(raw_line)

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            sample_uid = record.get("sample_uid")
            rollout_id = record.get("rollout_id")
            if sample_uid is None or rollout_id is None:
                raise ValueError(f"{path}:{line_number} is missing sample_uid or rollout_id")
            if record.get("output") is None:
                raise ValueError(f"{path}:{line_number} is missing output")
            if record.get("gts") is None or record.get("gts") == "":
                raise ValueError(f"{path}:{line_number} is missing gts")
            if expected_metadata:
                for key, expected in expected_metadata.items():
                    if record.get(key) != expected:
                        raise ValueError(
                            f"{path}:{line_number} has {key}={record.get(key)!r}; expected {expected!r}"
                        )
            rollout_ids[str(sample_uid)].add(int(rollout_id))
            row_count += 1

    expected_rows = expected_prompts * samples_per_prompt
    if row_count != expected_rows:
        raise ValueError(f"{path} has {row_count} rows; expected {expected_rows}")
    if len(rollout_ids) != expected_prompts:
        raise ValueError(f"{path} has {len(rollout_ids)} sample_uids; expected {expected_prompts}")

    expected_ids = set(range(samples_per_prompt))
    invalid = [
        (sample_uid, sorted(ids))
        for sample_uid, ids in rollout_ids.items()
        if ids != expected_ids
    ]
    if invalid:
        raise ValueError(
            f"{path} has invalid rollout IDs; first examples={invalid[:5]}, expected={sorted(expected_ids)}"
        )
    return row_count, digest.hexdigest()


def package_generation_jsonl(
    source: Path,
    destination: Path,
    *,
    training_run_id: str,
    checkpoint_step: int,
    benchmark_id: str,
    expected_prompts: int,
    samples_per_prompt: int,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    metadata = {
        "training_run_id": training_run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_id": f"step_{checkpoint_step}",
        "benchmark_id": benchmark_id,
    }

    try:
        with source.open(encoding="utf-8") as input_handle, temporary.open(
            "w", encoding="utf-8"
        ) as output_handle:
            for raw_line in input_handle:
                if not raw_line.strip():
                    continue
                source_record = json.loads(raw_line)
                record = {**source_record, **metadata}
                output_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        row_count, digest = scan_generation_jsonl(
            temporary,
            expected_prompts=expected_prompts,
            samples_per_prompt=samples_per_prompt,
            expected_metadata=metadata,
        )
        temporary.replace(destination)
        return row_count, digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def resolve_rescoring(
    config: dict[str, Any],
    *,
    script_dir: Path,
) -> tuple[bool, Path | None, Path | None, Path | None]:
    rescoring = config.get("rescoring", {})
    if not isinstance(rescoring, dict):
        raise ValueError("rescoring must be a JSON object")
    enabled = rescoring.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("rescoring.enabled must be true or false")
    if not enabled:
        return False, None, None, None

    scorer = script_dir / "rescore_generation_jsonl.py"
    if not scorer.is_file():
        raise FileNotFoundError(scorer)

    raw_parser_path = rescoring.get("ye_parser_path", "Lulu_OPSD-main/parser.py")
    parser_path = Path(str(raw_parser_path))
    if not parser_path.is_absolute():
        parser_path = Path(str(config["feng_j_root"])) / parser_path
    raw_python_path = rescoring.get("python_path", sys.executable)
    python_path = Path(str(raw_python_path))
    if not python_path.is_absolute():
        python_path = Path(str(config["feng_j_root"])) / python_path
    return True, scorer, parser_path, python_path


def run_ye_rescore(
    *,
    python_path: Path,
    scorer: Path,
    parser_path: Path,
    source_jsonl: Path,
    benchmark_id: str,
    data_name: str,
    expected_prompts: int,
    samples_per_prompt: int,
) -> dict[str, Any]:
    stem = source_jsonl.name.removesuffix(source_jsonl.suffix)
    details_path = source_jsonl.with_name(f"{stem}.ye_rescored.jsonl")
    summary_path = source_jsonl.with_name(f"{stem}.ye_metrics.json")
    subprocess.run(
        [
            str(python_path),
            str(scorer),
            "--input-jsonl",
            str(source_jsonl),
            "--benchmark-id",
            benchmark_id,
            "--data-name",
            data_name,
            "--expected-prompts",
            str(expected_prompts),
            "--samples-per-prompt",
            str(samples_per_prompt),
            "--ye-parser-path",
            str(parser_path),
            "--details-jsonl",
            str(details_path),
            "--summary-json",
            str(summary_path),
            "--reuse-if-current",
        ],
        check=True,
    )
    summary = load_json(summary_path)
    return {
        "data_name": data_name,
        "details_path": str(details_path),
        "summary_path": str(summary_path),
        "metrics": summary["metrics"],
        "parser_path": summary["scorer"]["parser_path"],
        "parser_sha256": summary["scorer"]["parser_sha256"],
        "environment": summary["scorer"]["environment"],
    }


def metric_percent(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.2f}%"


def write_checkpoint_summary(
    *,
    final_root: Path,
    training_run_id: str,
    checkpoint_step: int,
    benchmark_ids: list[str],
    task_status: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    checkpoint_id = f"step_{checkpoint_step}"
    benchmark_results: dict[str, Any] = {}
    complete = True

    for benchmark_id in benchmark_ids:
        task_key = f"{checkpoint_id}/{benchmark_id}"
        task = task_status.get(task_key)
        if not isinstance(task, dict):
            benchmark_results[benchmark_id] = {"status": "missing"}
            complete = False
            continue

        entry: dict[str, Any] = {"status": task.get("status", "unknown")}
        rescore = task.get("ye_rescore")
        if task.get("status") == "complete" and isinstance(rescore, dict):
            entry.update(
                {
                    "data_name": rescore["data_name"],
                    "metrics": rescore["metrics"],
                    "details_path": rescore["details_path"],
                    "summary_path": rescore["summary_path"],
                }
            )
        else:
            complete = False
            if task.get("error"):
                entry["error"] = task["error"]
        benchmark_results[benchmark_id] = entry

    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "training_run_id": training_run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_id": checkpoint_id,
        "status": "complete" if complete else "incomplete",
        "selected_benchmarks": benchmark_ids,
        "benchmarks": benchmark_results,
    }
    summary_path = final_root / checkpoint_id / "ye_benchmark_summary.json"
    write_json_atomic(summary_path, summary)

    print(f"\n===== Ye benchmark summary: {checkpoint_id} =====")
    print(
        f"{'benchmark':<16} {'data_name':<16} {'Pass@1':>8} {'correct rollouts':>18} "
        f"{'first-draw':>12} {'empirical Pass@K':>18} {'solved questions':>18}"
    )
    for benchmark_id in benchmark_ids:
        entry = benchmark_results[benchmark_id]
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            print(
                f"{benchmark_id:<16} {'-':<16} {'-':>8} {entry['status']:>18} "
                f"{'-':>12} {'-':>18} {'-':>18}"
            )
            continue
        empirical = (
            f"{metric_percent(metrics['empirical_pass_at_k'])} "
            f"(Pass@{metrics['k']})"
        )
        correct_rollouts = f"{metrics['correct_rollouts']}/{metrics['total_rollouts']}"
        solved_questions = (
            f"{metrics['questions_with_any_correct']}/{metrics['total_questions']}"
        )
        print(
            f"{benchmark_id:<16} {entry['data_name']:<16} "
            f"{metric_percent(metrics['pass_at_1_estimator']):>8} "
            f"{correct_rollouts:>18} "
            f"{metric_percent(metrics['first_draw_accuracy']):>12} "
            f"{empirical:>18} "
            f"{solved_questions:>18}"
        )
    print(f"status={summary['status']}")
    print(f"YE_CHECKPOINT_SUMMARY_JSON={summary_path}\n")
    return summary_path, summary


def task_environment(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_backend: str,
    checkpoint_step: int,
    benchmark_id: str,
    benchmark: dict[str, Any],
    work_root: Path,
) -> tuple[dict[str, str], Path]:
    generation = require_mapping(config, "generation")
    runtime = require_mapping(config, "runtime")
    feng_j = Path(str(config["feng_j_root"]))
    algorithm_id = require_identifier(config["algorithm_id"], "algorithm_id")
    model_id = require_identifier(config["model_id"], "model_id")
    training_run_id = require_identifier(config["training_run_id"], "training_run_id")
    model_label = f"step_{checkpoint_step}"
    eval_id = require_identifier(
        f"matrix_{checkpoint_step}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}",
        "eval_id",
    )

    max_prompt_length = positive_int(benchmark["max_prompt_length"], "max_prompt_length")
    max_response_length = positive_int(
        benchmark.get("max_response_length", generation["max_response_length"]),
        "max_response_length",
    )
    max_model_len = positive_int(
        benchmark.get("max_model_len", max_prompt_length + max_response_length + 1),
        "max_model_len",
    )
    if max_model_len < max_prompt_length + max_response_length:
        raise ValueError(
            f"{benchmark_id}.max_model_len cannot fit max_prompt_length + max_response_length"
        )

    environment = os.environ.copy()
    environment.update(
        {
            "FENG_J": str(feng_j),
            "ALGORITHM_ID": algorithm_id,
            "MODEL_ID": model_id,
            "TRAINING_RUN_ID": training_run_id,
            "MODEL_LABEL": model_label,
            "MODEL_PATH": str(checkpoint_path),
            "MODEL_BACKEND": checkpoint_backend,
            "BENCHMARK_ID": benchmark_id,
            "BENCHMARK_FILE": str(benchmark["data_path"]),
            "EXPECTED_PROMPTS": str(positive_int(benchmark["expected_prompts"], "expected_prompts")),
            "PROMPT_KEY": str(benchmark.get("prompt_key", "prompt")),
            "RESPONSES_KEY": str(benchmark.get("responses_key", "responses")),
            "GROUND_TRUTH_FIELD": str(
                benchmark.get("ground_truth_field", "reward_model.ground_truth")
            ),
            "N_SAMPLES": str(positive_int(generation["n_samples"], "generation.n_samples")),
            "GENERATION_BATCH_SIZE": str(
                positive_int(
                    benchmark.get(
                        "generation_batch_size", runtime["generation_batch_size"]
                    ),
                    "generation_batch_size",
                )
            ),
            "ENABLE_THINKING": str(bool(generation["enable_thinking"])),
            "MAX_PROMPT_LENGTH": str(max_prompt_length),
            "MAX_RESPONSE_LENGTH": str(max_response_length),
            "EXPECTED_RESPONSE_LENGTH": str(
                positive_int(
                    benchmark.get(
                        "expected_response_length",
                        generation.get("expected_response_length", max_response_length),
                    ),
                    "expected_response_length",
                )
            ),
            "MAX_MODEL_LEN": str(max_model_len),
            "GENERATION_DO_SAMPLE": str(bool(generation["do_sample"])),
            "GENERATION_TEMPERATURE": str(float(generation["temperature"])),
            "GENERATION_TOP_P": str(float(generation["top_p"])),
            "GENERATION_TOP_K": str(int(generation["top_k"])),
            "GENERATION_MIN_P": str(float(generation["min_p"])),
            "NNODES": str(positive_int(runtime.get("nnodes", 1), "runtime.nnodes")),
            "NGPUS_PER_NODE": str(
                positive_int(runtime["gpus_per_node"], "runtime.gpus_per_node")
            ),
            "ROLLOUT_TP": str(
                positive_int(runtime["tensor_parallel_size"], "runtime.tensor_parallel_size")
            ),
            "ROLLOUT_GPU_MEM_UTIL": str(float(runtime["gpu_memory_utilization"])),
            "MAX_NUM_SEQS": str(
                positive_int(runtime["max_num_seqs_per_replica"], "runtime.max_num_seqs")
            ),
            "MAX_NUM_BATCHED_TOKENS": str(
                positive_int(runtime["max_num_batched_tokens"], "runtime.max_num_batched_tokens")
            ),
            "PHYSICAL_VRAM_GIB": str(float(runtime["physical_vram_gib"])),
            "RUNTIME_RESERVE_GIB": str(float(runtime["runtime_reserve_gib"])),
            "RESULTS_ROOT": str(work_root),
            "EVAL_ID": eval_id,
        }
    )

    source_jsonl = (
        work_root
        / algorithm_id
        / model_id
        / "runs"
        / training_run_id
        / benchmark_id
        / eval_id
        / "models"
        / model_label
        / "validation"
        / "0.jsonl"
    )
    return environment, source_jsonl


def main() -> int:
    args = parse_args()
    environment_config = load_json(args.environment_config)
    validation_config = load_json(args.validation_config)
    config = resolve_evaluation_config(environment_config, validation_config)
    catalog = load_json(args.catalog)
    if int(catalog.get("schema_version", 0)) != 1:
        raise ValueError("Benchmark catalog schema_version must be 1")

    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_checkpoint_generation.sh"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    rescore_enabled, rescorer, ye_parser_path, rescore_python_path = resolve_rescoring(
        config,
        script_dir=script_dir,
    )

    checkpoint_config = require_mapping(config, "checkpoint")
    checkpoint_root = Path(str(checkpoint_config["root"]))
    checkpoint_backend = str(checkpoint_config.get("backend", "fsdp"))
    actor_subdir = str(checkpoint_config.get("actor_subdir", "actor"))
    steps = resolve_steps(args, checkpoint_config)
    benchmarks = resolve_benchmarks(
        args,
        catalog,
        require_ye_data_name=rescore_enabled,
        workspace_root=Path(config["feng_j_root"]),
    )

    training_run_id = require_identifier(config["training_run_id"], "training_run_id")
    results_root = Path(str(require_mapping(config, "output")["results_root"]))
    work_root = Path(str(require_mapping(config, "output")["work_root"]))
    final_root = results_root / training_run_id
    manifest_path = final_root / "manifest.json"

    print("===== checkpoint generation matrix =====")
    print(f"environment     : {args.environment_config}")
    print(f"validation      : {args.validation_config}")
    print(f"environment ID  : {config['environment_id']}")
    print(f"validation ID   : {config['validation_id']}")
    print(f"catalog         : {args.catalog}")
    print(f"training run    : {training_run_id}")
    print(f"checkpoint steps: {steps}")
    print(f"benchmarks      : {[benchmark_id for benchmark_id, _ in benchmarks]}")
    generation = require_mapping(config, "generation")
    print(
        "model mode     : "
        + ("thinking" if generation["enable_thinking"] else "no-thinking")
    )
    print(
        "sampling       : "
        + (
            f"enabled, n={generation['n_samples']}, "
            f"temperature={generation['temperature']}, "
            f"top_p={generation['top_p']}, top_k={generation['top_k']}, "
            f"min_p={generation['min_p']}"
            if generation["do_sample"]
            else "disabled (greedy, n=1)"
        )
    )
    print(f"final root      : {final_root}")
    print(f"work root       : {work_root}")
    print(
        "offline rescore : "
        + (
            f"Ye Wenxuan parser at {ye_parser_path}"
            if rescore_enabled
            else "disabled"
        )
    )
    if rescore_enabled:
        print(f"rescore Python  : {rescore_python_path}")
    print("execution       : sequential, one checkpoint/benchmark task at a time")

    tasks = []
    for step in steps:
        checkpoint_path = checkpoint_root / f"global_step_{step}" / actor_subdir
        for benchmark_id, benchmark in benchmarks:
            destination = final_root / f"step_{step}" / f"{benchmark_id}.jsonl"
            tasks.append((step, checkpoint_path, benchmark_id, benchmark, destination))
            print(
                f"  step={step:<6} benchmark={benchmark_id:<12} "
                f"data_name={benchmark.get('ye_data_name', '-'):<16} "
                f"checkpoint={checkpoint_path} output={destination}"
            )
    print("========================================")

    if args.dry_run:
        return 0

    configured_verl_python = Path(config["verl_python"])
    if not configured_verl_python.is_file():
        raise FileNotFoundError(f"Configured Verl Python not found: {configured_verl_python}")
    if Path(sys.executable).resolve() != configured_verl_python.resolve():
        raise RuntimeError(
            f"Matrix is running under {Path(sys.executable).resolve()}, but the evaluation "
            f"environment requires {configured_verl_python.resolve()}"
        )

    if rescore_enabled:
        assert rescorer is not None
        assert ye_parser_path is not None
        assert rescore_python_path is not None
        subprocess.run(
            [
                str(rescore_python_path),
                str(rescorer),
                "--ye-parser-path",
                str(ye_parser_path),
                "--check-parser-only",
            ],
            check=True,
        )

    final_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    manifest.update(
        {
            "schema_version": 1,
            "training_run_id": training_run_id,
            "environment_id": config["environment_id"],
            "validation_id": config["validation_id"],
            "algorithm_id": config["algorithm_id"],
            "model_id": config["model_id"],
            "environment_config_path": str(args.environment_config),
            "validation_config_path": str(args.validation_config),
            "catalog_path": str(args.catalog),
            "generation": config["generation"],
            "runtime": config["runtime"],
            "rescoring": config.get("rescoring", {"enabled": False}),
            "updated_at_utc": utc_now(),
        }
    )
    task_status = manifest.setdefault("tasks", {})
    checkpoint_summaries = manifest.setdefault("checkpoint_summaries", {})
    selected_benchmark_ids = [benchmark_id for benchmark_id, _ in benchmarks]
    last_benchmark_id = selected_benchmark_ids[-1]

    failures = 0
    for step, checkpoint_path, benchmark_id, benchmark, destination in tasks:
        task_key = f"step_{step}/{benchmark_id}"
        metadata = {
            "training_run_id": training_run_id,
            "checkpoint_step": step,
            "checkpoint_id": f"step_{step}",
            "benchmark_id": benchmark_id,
        }
        expected_prompts = int(benchmark["expected_prompts"])
        samples_per_prompt = int(config["generation"]["n_samples"])

        generation_complete = False
        row_count = 0
        digest = ""
        if destination.is_file() and not args.force:
            try:
                row_count, digest = scan_generation_jsonl(
                    destination,
                    expected_prompts=expected_prompts,
                    samples_per_prompt=samples_per_prompt,
                    expected_metadata=metadata,
                )
                generation_complete = True
                print(f"SKIP generation: {task_key} ({row_count} rows)")
            except Exception as exc:
                print(f"Existing output is incomplete and will be regenerated: {destination}: {exc}")

        stage = "generation"
        try:
            if not generation_complete:
                if not checkpoint_path.is_dir():
                    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
                benchmark_path = Path(str(benchmark["data_path"]))
                if not benchmark_path.is_file():
                    raise FileNotFoundError(f"Benchmark parquet not found: {benchmark_path}")

                environment, source_jsonl = task_environment(
                    config=config,
                    checkpoint_path=checkpoint_path,
                    checkpoint_backend=checkpoint_backend,
                    checkpoint_step=step,
                    benchmark_id=benchmark_id,
                    benchmark=benchmark,
                    work_root=work_root,
                )
                print(f"RUN {task_key}")
                subprocess.run(["bash", str(runner)], env=environment, check=True)
                row_count, digest = package_generation_jsonl(
                    source_jsonl,
                    destination,
                    training_run_id=training_run_id,
                    checkpoint_step=step,
                    benchmark_id=benchmark_id,
                    expected_prompts=expected_prompts,
                    samples_per_prompt=samples_per_prompt,
                )
                print(f"PASS generation {task_key}: {row_count} rows -> {destination}")

            rescore_result = None
            if rescore_enabled:
                stage = "rescoring"
                assert rescorer is not None
                assert ye_parser_path is not None
                assert rescore_python_path is not None
                rescore_result = run_ye_rescore(
                    python_path=rescore_python_path,
                    scorer=rescorer,
                    parser_path=ye_parser_path,
                    source_jsonl=destination,
                    benchmark_id=benchmark_id,
                    data_name=str(benchmark["ye_data_name"]),
                    expected_prompts=expected_prompts,
                    samples_per_prompt=samples_per_prompt,
                )

            task_status[task_key] = {
                "status": "complete",
                "checkpoint_path": str(checkpoint_path),
                "output_path": str(destination),
                "rows": row_count,
                "sha256": digest,
                "ye_rescore": rescore_result,
                "updated_at_utc": utc_now(),
            }
        except Exception as exc:
            failures += 1
            task_status[task_key] = {
                "status": f"{stage}_failed",
                "checkpoint_path": str(checkpoint_path),
                "output_path": str(destination),
                "rows": row_count or None,
                "sha256": digest or None,
                "error": str(exc),
                "updated_at_utc": utc_now(),
            }
            print(f"FAILED {task_key}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                manifest["updated_at_utc"] = utc_now()
                write_json_atomic(manifest_path, manifest)
                raise
        finally:
            manifest["updated_at_utc"] = utc_now()
            write_json_atomic(manifest_path, manifest)

        if rescore_enabled and benchmark_id == last_benchmark_id:
            try:
                summary_path, checkpoint_summary = write_checkpoint_summary(
                    final_root=final_root,
                    training_run_id=training_run_id,
                    checkpoint_step=step,
                    benchmark_ids=selected_benchmark_ids,
                    task_status=task_status,
                )
                checkpoint_summaries[f"step_{step}"] = {
                    "status": checkpoint_summary["status"],
                    "summary_path": str(summary_path),
                    "selected_benchmarks": selected_benchmark_ids,
                    "updated_at_utc": utc_now(),
                }
            except Exception as exc:
                failures += 1
                checkpoint_summaries[f"step_{step}"] = {
                    "status": "failed",
                    "error": str(exc),
                    "updated_at_utc": utc_now(),
                }
                print(f"FAILED checkpoint summary step_{step}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    manifest["updated_at_utc"] = utc_now()
                    write_json_atomic(manifest_path, manifest)
                    raise
            finally:
                manifest["updated_at_utc"] = utc_now()
                write_json_atomic(manifest_path, manifest)

    print(f"Generation matrix finished: tasks={len(tasks)}, failures={failures}")
    print(f"MANIFEST_JSON={manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
