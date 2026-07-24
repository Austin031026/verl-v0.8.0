#!/usr/bin/env python3
"""Re-score a complete rollout JSONL with Ye Wenxuan's parser."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


SCORER_NAME = "ye_wenxuan_parser"
SCORER_VERSION = 1


def default_ye_parser_path() -> Path:
    configured = os.environ.get("YE_WENXUAN_PARSER_PATH")
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "Lulu_OPSD-main" / "parser.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--benchmark-id")
    parser.add_argument(
        "--data-name",
        help="Exact data_name passed to Ye's extract_answer; MATH-500 uses math.",
    )
    parser.add_argument("--expected-prompts", type=int)
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--ye-parser-path", type=Path, default=default_ye_parser_path())
    parser.add_argument("--details-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--check-parser-only",
        action="store_true",
        help="Import the configured Ye parser and exit without reading a rollout JSONL.",
    )
    parser.add_argument(
        "--reuse-if-current",
        action="store_true",
        help="Reuse sidecars only when source, parser, dataset, counts, and details hash all match.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_fingerprint() -> dict[str, Any]:
    packages = {}
    for distribution in ("latex2sympy2", "numpy", "regex", "sympy", "word2number"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def default_output_paths(input_jsonl: Path) -> tuple[Path, Path]:
    stem = input_jsonl.name.removesuffix(input_jsonl.suffix)
    return (
        input_jsonl.with_name(f"{stem}.ye_rescored.jsonl"),
        input_jsonl.with_name(f"{stem}.ye_metrics.json"),
    )


def load_ye_parser(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Ye Wenxuan parser not found: {path}")

    spec = importlib.util.spec_from_file_location("_ye_wenxuan_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Ye Wenxuan parser from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Cannot import Ye Wenxuan parser dependency {exc.name!r}. "
            "Install examples/evaluation/requirements_ye_rescore.txt in this Python environment."
        ) from exc

    required = ("extract_answer", "strip_string", "math_equal")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"{path} is missing required callables: {missing}")
    return module


def load_rows(
    path: Path,
    *,
    benchmark_id: str,
    expected_prompts: int,
    samples_per_prompt: int,
) -> tuple[list[dict[str, Any]], str, bool]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_prompts <= 0 or samples_per_prompt <= 0:
        raise ValueError("expected_prompts and samples_per_prompt must be greater than zero")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if row.get("output") is None:
                raise ValueError(f"{path}:{line_number} is missing output")
            if row.get("gts") is None or row.get("gts") == "":
                raise ValueError(f"{path}:{line_number} is missing gts")
            row_benchmark = row.get("benchmark_id")
            if row_benchmark is not None and str(row_benchmark) != benchmark_id:
                raise ValueError(
                    f"{path}:{line_number} has benchmark_id={row_benchmark!r}; "
                    f"expected {benchmark_id!r}"
                )
            rows.append(row)

    expected_rows = expected_prompts * samples_per_prompt
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows; expected {expected_rows}")

    uid_presence = [row.get("sample_uid") is not None for row in rows]
    if any(uid_presence) and not all(uid_presence):
        raise ValueError("sample_uid must be present on every row or absent from every row")
    key_source = "sample_uid" if all(uid_presence) else "input"
    if key_source == "input" and any(row.get("input") is None for row in rows):
        raise ValueError("Rows without sample_uid must all contain input")

    rollout_id_presence = [row.get("rollout_id") is not None for row in rows]
    if any(rollout_id_presence) and not all(rollout_id_presence):
        raise ValueError("rollout_id must be present on every row or absent from every row")
    has_rollout_ids = all(rollout_id_presence)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped_indices[str(row[key_source])].append(row_index)

    if len(grouped_indices) != expected_prompts:
        raise ValueError(
            f"{path} has {len(grouped_indices)} distinct {key_source} values; "
            f"expected {expected_prompts}"
        )

    expected_ids = set(range(samples_per_prompt))
    for question_key, indices in grouped_indices.items():
        if len(indices) != samples_per_prompt:
            raise ValueError(
                f"Question {question_key!r} has {len(indices)} rollouts; "
                f"expected {samples_per_prompt}"
            )
        if has_rollout_ids:
            ids = [int(rows[index]["rollout_id"]) for index in indices]
            if len(set(ids)) != len(ids) or set(ids) != expected_ids:
                raise ValueError(
                    f"Question {question_key!r} has rollout IDs {sorted(ids)}; "
                    f"expected {sorted(expected_ids)}"
                )
        else:
            for derived_rollout_id, index in enumerate(indices):
                rows[index]["_derived_rollout_id"] = derived_rollout_id

        gold_values = {str(rows[index]["gts"]) for index in indices}
        if len(gold_values) != 1:
            raise ValueError(f"Question {question_key!r} has inconsistent gts values")
        input_values = {
            str(rows[index]["input"])
            for index in indices
            if rows[index].get("input") is not None
        }
        if len(input_values) > 1:
            raise ValueError(f"Question {question_key!r} has inconsistent input values")

    return rows, key_source, has_rollout_ids


def details_record(
    row: dict[str, Any],
    *,
    question_key: str,
    rollout_id: int,
    data_name: str,
    prediction: str,
    correct: bool,
    error: str | None,
) -> dict[str, Any]:
    record = {
        key: row[key]
        for key in (
            "training_run_id",
            "checkpoint_step",
            "checkpoint_id",
            "benchmark_id",
            "sample_uid",
        )
        if row.get(key) is not None
    }
    if "sample_uid" not in record:
        record["input"] = question_key
    record.update(
        {
            "rollout_id": rollout_id,
            "ye_data_name": data_name,
            "ye_pred_answer": prediction,
            "ye_correct": correct,
        }
    )
    if error is not None:
        record["ye_error"] = error
    return record


def score_rows(
    rows: list[dict[str, Any]],
    *,
    key_source: str,
    has_rollout_ids: bool,
    data_name: str,
    ye_parser: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    grouped_correctness: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    judge_errors = 0
    unextractable = 0

    for row in rows:
        question_key = str(row[key_source])
        rollout_id = int(row["rollout_id"] if has_rollout_ids else row["_derived_rollout_id"])
        prediction = ""
        correct = False
        error = None

        try:
            prediction = str(
                ye_parser.extract_answer(str(row["output"]), data_name=data_name)
            )
            gold = ye_parser.strip_string(str(row["gts"]))
            try:
                correct = bool(ye_parser.math_equal(prediction, gold))
            except Exception as exc:
                error = f"math_equal: {type(exc).__name__}: {exc}"
        except Exception as exc:
            error = f"extract_or_normalize: {type(exc).__name__}: {exc}"

        if prediction == "":
            unextractable += 1
        if error is not None:
            judge_errors += 1

        details.append(
            details_record(
                row,
                question_key=question_key,
                rollout_id=rollout_id,
                data_name=data_name,
                prediction=prediction,
                correct=correct,
                error=error,
            )
        )
        grouped_correctness[question_key].append((rollout_id, correct))

    correct_rollouts = sum(record["ye_correct"] for record in details)
    first_draw_correct = 0
    questions_with_any_correct = 0
    for judgments in grouped_correctness.values():
        ordered = sorted(judgments)
        first_draw_correct += int(ordered[0][1])
        questions_with_any_correct += int(any(correct for _, correct in ordered))

    total_rollouts = len(details)
    total_questions = len(grouped_correctness)
    metrics = {
        "pass_at_1_estimator": correct_rollouts / total_rollouts,
        "correct_rollouts": correct_rollouts,
        "total_rollouts": total_rollouts,
        "first_draw_accuracy": first_draw_correct / total_questions,
        "first_draw_correct": first_draw_correct,
        "total_questions": total_questions,
        "empirical_pass_at_k": questions_with_any_correct / total_questions,
        "questions_with_any_correct": questions_with_any_correct,
        "k": len(next(iter(grouped_correctness.values()))),
        "unextractable_rollouts": unextractable,
        "judge_errors": judge_errors,
    }
    return details, metrics


def write_details_atomic(path: Path, details: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in details:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def current_summary(
    path: Path,
    *,
    source_sha256: str,
    parser_sha256: str,
    benchmark_id: str,
    data_name: str,
    expected_prompts: int,
    samples_per_prompt: int,
    details_path: Path,
    environment: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file() or not details_path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "scorer_name": SCORER_NAME,
            "scorer_version": SCORER_VERSION,
            "benchmark_id": benchmark_id,
            "data_name": data_name,
            "source_sha256": source_sha256,
            "parser_sha256": parser_sha256,
            "expected_prompts": expected_prompts,
            "samples_per_prompt": samples_per_prompt,
            "environment": environment,
        }
        actual = {
            "schema_version": summary.get("schema_version"),
            "scorer_name": summary.get("scorer", {}).get("name"),
            "scorer_version": summary.get("scorer", {}).get("version"),
            "benchmark_id": summary.get("benchmark_id"),
            "data_name": summary.get("scorer", {}).get("data_name"),
            "source_sha256": summary.get("source", {}).get("sha256"),
            "parser_sha256": summary.get("scorer", {}).get("parser_sha256"),
            "expected_prompts": summary.get("validation", {}).get("expected_prompts"),
            "samples_per_prompt": summary.get("validation", {}).get("samples_per_prompt"),
            "environment": summary.get("scorer", {}).get("environment"),
        }
        if actual != expected:
            return None
        if summary.get("details", {}).get("sha256") != sha256_file(details_path):
            return None
        return summary
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def print_result(summary: dict[str, Any], *, reused: bool) -> None:
    metrics = summary["metrics"]
    prefix = "REUSE" if reused else "YE SCORE"
    print(
        f"{prefix} {summary['checkpoint_id'] or '-'} / {summary['benchmark_id']} "
        f"(data_name={summary['scorer']['data_name']}): "
        f"Pass@1={percent(metrics['pass_at_1_estimator'])} "
        f"({metrics['correct_rollouts']}/{metrics['total_rollouts']}), "
        f"first-draw={percent(metrics['first_draw_accuracy'])} "
        f"({metrics['first_draw_correct']}/{metrics['total_questions']}), "
        f"empirical Pass@{metrics['k']}={percent(metrics['empirical_pass_at_k'])} "
        f"({metrics['questions_with_any_correct']}/{metrics['total_questions']})"
    )


def main() -> int:
    args = parse_args()
    parser_path = args.ye_parser_path.resolve()
    if args.check_parser_only:
        load_ye_parser(parser_path)
        print(f"YE_PARSER_OK={parser_path}")
        return 0

    required = {
        "--input-jsonl": args.input_jsonl,
        "--benchmark-id": args.benchmark_id,
        "--data-name": args.data_name,
        "--expected-prompts": args.expected_prompts,
        "--samples-per-prompt": args.samples_per_prompt,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required scoring arguments: {', '.join(missing)}")
    assert args.input_jsonl is not None
    assert args.benchmark_id is not None
    assert args.data_name is not None
    assert args.expected_prompts is not None
    assert args.samples_per_prompt is not None

    if not args.benchmark_id.strip():
        raise ValueError("benchmark_id cannot be empty")
    if not args.data_name.strip():
        raise ValueError("data_name cannot be empty")

    default_details, default_summary = default_output_paths(args.input_jsonl)
    details_path = args.details_jsonl or default_details
    summary_path = args.summary_json or default_summary
    if not parser_path.is_file():
        raise FileNotFoundError(f"Ye Wenxuan parser not found: {parser_path}")

    source_sha256 = sha256_file(args.input_jsonl)
    parser_sha256 = sha256_file(parser_path)
    environment = environment_fingerprint()
    if args.reuse_if_current:
        summary = current_summary(
            summary_path,
            source_sha256=source_sha256,
            parser_sha256=parser_sha256,
            benchmark_id=args.benchmark_id,
            data_name=args.data_name,
            expected_prompts=args.expected_prompts,
            samples_per_prompt=args.samples_per_prompt,
            details_path=details_path,
            environment=environment,
        )
        if summary is not None:
            print_result(summary, reused=True)
            print(f"YE_DETAILS_JSONL={details_path}")
            print(f"YE_METRICS_JSON={summary_path}")
            return 0

    rows, key_source, has_rollout_ids = load_rows(
        args.input_jsonl,
        benchmark_id=args.benchmark_id,
        expected_prompts=args.expected_prompts,
        samples_per_prompt=args.samples_per_prompt,
    )
    ye_parser = load_ye_parser(parser_path)
    details, metrics = score_rows(
        rows,
        key_source=key_source,
        has_rollout_ids=has_rollout_ids,
        data_name=args.data_name,
        ye_parser=ye_parser,
    )
    details_sha256 = write_details_atomic(details_path, details)

    first_row = rows[0]
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "training_run_id": first_row.get("training_run_id"),
        "checkpoint_step": first_row.get("checkpoint_step"),
        "checkpoint_id": first_row.get("checkpoint_id"),
        "benchmark_id": args.benchmark_id,
        "source": {
            "path": str(args.input_jsonl),
            "sha256": source_sha256,
        },
        "scorer": {
            "name": SCORER_NAME,
            "version": SCORER_VERSION,
            "data_name": args.data_name,
            "parser_path": str(parser_path),
            "parser_sha256": parser_sha256,
            "environment": environment,
            "procedure": (
                "gold=strip_string(str(gts)); "
                "pred=extract_answer(str(output), data_name=data_name); "
                "correct=bool(math_equal(pred, gold))"
            ),
        },
        "validation": {
            "expected_prompts": args.expected_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "expected_rollouts": args.expected_prompts * args.samples_per_prompt,
            "question_key": key_source,
            "rollout_id_source": "rollout_id" if has_rollout_ids else "within-group file order",
        },
        "details": {
            "path": str(details_path),
            "rows": len(details),
            "sha256": details_sha256,
        },
        "metrics": metrics,
    }
    write_json_atomic(summary_path, summary)
    print_result(summary, reused=False)
    print(f"YE_DETAILS_JSONL={details_path}")
    print(f"YE_METRICS_JSON={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
