#!/usr/bin/env python3
"""Summarize a checkpoint benchmark run and update its long-lived registry."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--resolved-models", type=Path, required=True)
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def read_models(path: Path) -> list[dict[str, str]]:
    models = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected 4 tab-separated fields")
            label, backend, source_path, resolved_path = fields
            models.append(
                {
                    "label": label,
                    "backend": backend,
                    "source_path": source_path,
                    "resolved_path": resolved_path,
                }
            )
    if not models:
        raise ValueError(f"no models found in {path}")
    return models


def as_correct(row: dict[str, Any]) -> bool:
    if "acc" in row:
        return bool(row["acc"])
    score = row.get("score")
    return isinstance(score, (int, float)) and score > 0


def mean_or_none(values: list[float | int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    shared_prompts = sorted(set(baseline) & set(candidate))
    if not shared_prompts:
        return {"shared_prompts": 0, "accuracy_delta_pp": None, "paired_bootstrap_95ci_pp": None}

    prompt_deltas = [candidate[prompt] - baseline[prompt] for prompt in shared_prompts]
    observed = statistics.fmean(prompt_deltas) * 100
    rng = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(samples):
        draw = [prompt_deltas[rng.randrange(len(prompt_deltas))] for _ in prompt_deltas]
        bootstrap_deltas.append(statistics.fmean(draw) * 100)

    return {
        "shared_prompts": len(shared_prompts),
        "accuracy_delta_pp": observed,
        "paired_bootstrap_95ci_pp": [
            percentile(bootstrap_deltas, 0.025),
            percentile(bootstrap_deltas, 0.975),
        ],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None, None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        return tokenizer, None
    except Exception as error:  # tokenizer metrics are useful but must not discard accuracy results
        return None, f"could not load cached tokenizer {tokenizer_path!r}: {error}"


def summarize_model(
    model: dict[str, str],
    eval_root: Path,
    expected_prompts: int,
    samples_per_prompt: int,
    max_response_length: int,
    tokenizer,
    tokenizer_warning: str | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    model_root = eval_root / "models" / model["label"]
    status_path = model_root / "exit_status.txt"
    generations_path = model_root / "validation" / "0.jsonl"
    warnings = []

    try:
        exit_status = int(status_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        exit_status = None
        warnings.append("missing or invalid exit_status.txt")

    rows = read_jsonl(generations_path) if generations_path.exists() else []
    if not generations_path.exists():
        warnings.append("missing validation/0.jsonl")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("input", ""))].append(row)

    group_sizes = Counter(len(group) for group in grouped.values())
    expected_rows = expected_prompts * samples_per_prompt
    if len(rows) != expected_rows:
        warnings.append(f"expected {expected_rows} rows, found {len(rows)}")
    if len(grouped) != expected_prompts:
        warnings.append(f"expected {expected_prompts} unique prompts, found {len(grouped)}")
    if group_sizes != Counter({samples_per_prompt: expected_prompts}):
        warnings.append(f"unexpected samples-per-prompt distribution: {dict(sorted(group_sizes.items()))}")

    correct_values = [as_correct(row) for row in rows]
    prompt_accuracy = {
        prompt: statistics.fmean(as_correct(row) for row in prompt_rows)
        for prompt, prompt_rows in grouped.items()
    }
    exact_pass = mean_or_none([any(as_correct(row) for row in prompt_rows) for prompt_rows in grouped.values()])

    has_predictions = bool(rows) and all("pred" in row for row in rows)
    invalid_count = None
    majority_accuracy = None
    if has_predictions:
        invalid_count = sum(str(row["pred"]).strip() in {"", "[INVALID]"} for row in rows)
        majority_results = []
        for prompt_rows in grouped.values():
            vote_counts = Counter(str(row["pred"]) for row in prompt_rows)
            majority_prediction = vote_counts.most_common(1)[0][0]
            representative = next(row for row in prompt_rows if str(row["pred"]) == majority_prediction)
            majority_results.append(as_correct(representative))
        majority_accuracy = mean_or_none(majority_results)

    rewards = [row["reward"] for row in rows if isinstance(row.get("reward"), (int, float))]
    output_chars = [len(str(row.get("output", ""))) for row in rows]
    token_lengths = None
    near_limit_count = None
    if tokenizer is not None:
        token_lengths = [
            len(tokenizer.encode(str(row.get("output", "")), add_special_tokens=False)) for row in rows
        ]
        near_limit_count = sum(length >= max_response_length - 8 for length in token_lengths)
    elif tokenizer_warning:
        warnings.append(tokenizer_warning)

    if invalid_count is not None and rows and invalid_count / len(rows) > 0.01:
        warnings.append("more than 1% of predictions could not be extracted")
    if near_limit_count is not None and rows and near_limit_count / len(rows) > 0.05:
        warnings.append("more than 5% of responses are within 8 tokens of the response limit")

    prompt_records = [
        {
            "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "samples": len(prompt_rows),
            "accuracy": prompt_accuracy[prompt],
            "any_correct": any(as_correct(row) for row in prompt_rows),
        }
        for prompt, prompt_rows in sorted(grouped.items())
    ]

    status = "pass" if exit_status == 0 and not warnings else "pass_with_warnings"
    if exit_status not in (0, None) or not rows:
        status = "fail"

    summary = {
        **model,
        "status": status,
        "exit_status": exit_status,
        "generations_path": str(generations_path),
        "samples": len(rows),
        "expected_samples": expected_rows,
        "unique_prompts": len(grouped),
        "samples_per_prompt_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
        "correct": sum(correct_values),
        "accuracy_mean_at_n": mean_or_none(correct_values),
        "exact_pass_at_n": exact_pass,
        "majority_vote_accuracy": majority_accuracy,
        "invalid_prediction_count": invalid_count,
        "invalid_prediction_rate": invalid_count / len(rows) if invalid_count is not None and rows else None,
        "reward_mean": mean_or_none(rewards),
        "mean_output_chars": mean_or_none(output_chars),
        "mean_retokenized_output_tokens": mean_or_none(token_lengths or []),
        "retokenized_near_limit_count": near_limit_count,
        "retokenized_near_limit_rate": near_limit_count / len(rows) if near_limit_count is not None and rows else None,
        "warnings": warnings,
        "per_prompt": prompt_records,
    }
    return summary, prompt_accuracy


def update_registry(registry_path: Path, run_summary: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        else:
            registry = {
                "schema_version": 1,
                "algorithm_id": run_summary["algorithm_id"],
                "model_id": run_summary["model_id"],
                "benchmark_runs": [],
            }

        if registry.get("algorithm_id") != run_summary["algorithm_id"]:
            raise ValueError("registry algorithm_id does not match this run")
        if registry.get("model_id") != run_summary["model_id"]:
            raise ValueError("registry model_id does not match this run")

        identity = (
            run_summary["training_run_id"],
            run_summary["benchmark_id"],
            run_summary["eval_id"],
        )
        retained = [
            item
            for item in registry["benchmark_runs"]
            if (item["training_run_id"], item["benchmark_id"], item["eval_id"]) != identity
        ]
        retained.append(run_summary)
        registry["benchmark_runs"] = sorted(
            retained,
            key=lambda item: (item["training_run_id"], item["benchmark_id"], item["eval_id"]),
        )
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()

        temporary_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(registry_path)


def main() -> None:
    args = parse_args()
    models = read_models(args.resolved_models)
    config_snapshot = json.loads(args.config_snapshot.read_text(encoding="utf-8"))
    tokenizer, tokenizer_warning = load_tokenizer(args.tokenizer_path)

    model_summaries = {}
    prompt_accuracies = {}
    for model in models:
        summary, prompt_accuracy = summarize_model(
            model=model,
            eval_root=args.eval_root,
            expected_prompts=args.expected_prompts,
            samples_per_prompt=args.samples_per_prompt,
            max_response_length=args.max_response_length,
            tokenizer=tokenizer,
            tokenizer_warning=tokenizer_warning,
        )
        model_summaries[model["label"]] = summary
        prompt_accuracies[model["label"]] = prompt_accuracy

    baseline_label = models[0]["label"]
    comparisons = {}
    for model in models[1:]:
        label = model["label"]
        comparisons[f"{label}_vs_{baseline_label}"] = paired_bootstrap(
            baseline=prompt_accuracies[baseline_label],
            candidate=prompt_accuracies[label],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    run_summary = {
        "algorithm_id": args.algorithm_id,
        "model_id": args.model_id,
        "training_run_id": args.training_run_id,
        "benchmark_id": args.benchmark_id,
        "eval_id": args.eval_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_root": str(args.eval_root),
        "config": config_snapshot,
        "protocol": {
            "expected_prompts": args.expected_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "max_response_length": args.max_response_length,
            "tokenizer_path": args.tokenizer_path,
        },
        "baseline_label": baseline_label,
        "models": model_summaries,
        "comparisons": comparisons,
    }

    summary_path = args.eval_root / "summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_registry(args.registry_path, run_summary)

    print("\nMODEL RESULTS")
    print(f"{'model':28s} {'status':20s} {'samples':>8s} {'correct':>8s} {'accuracy':>10s} {'pass@n':>10s}")
    for label, summary in model_summaries.items():
        accuracy = summary["accuracy_mean_at_n"]
        pass_at_n = summary["exact_pass_at_n"]
        accuracy_text = f"{accuracy:.6f}" if accuracy is not None else "n/a"
        pass_text = f"{pass_at_n:.6f}" if pass_at_n is not None else "n/a"
        print(
            f"{label:28s} {summary['status']:20s} {summary['samples']:8d} "
            f"{summary['correct']:8d} {accuracy_text:>10s} {pass_text:>10s}"
        )
    print(f"SUMMARY_JSON={summary_path}")
    print(f"REGISTRY_JSON={args.registry_path}")


if __name__ == "__main__":
    main()
