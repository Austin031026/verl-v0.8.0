#!/usr/bin/env python3
"""Remove the repeated prompts from the released DAPO-Math-17K parquet.

The released parquet currently contains many copies of the same prompt.  This
script keeps the first complete verl row for every exact ``prompt`` value,
drops prompt groups that disagree on ``ground_truth``, and writes a new parquet
with the input Arrow schema preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Released DAPO parquet")
    parser.add_argument("--output", required=True, type=Path, help="Exact-prompt-deduplicated parquet")
    parser.add_argument("--report", required=True, type=Path, help="JSON audit report")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--expected-input-rows",
        type=int,
        default=1_791_700,
        help="Set to 0 to disable the released-file row-count check",
    )
    parser.add_argument(
        "--expected-output-rows",
        type=int,
        default=17_391,
        help="Set to 0 to disable the known clean-row count check",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_key(prompt: Any, row_index: int) -> str:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"row {row_index}: prompt must be a non-empty list")
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError(f"row {row_index}: prompt message must contain string content")
    return canonical_json(prompt)


def ground_truth_key(reward_model: Any, row_index: int) -> str:
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError(f"row {row_index}: reward_model.ground_truth is missing")
    return canonical_json(reward_model["ground_truth"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def column_values(batch: pa.RecordBatch, name: str) -> list[Any]:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"missing required parquet column: {name}")
    return batch.column(index).to_pylist()


def scan_input(args: argparse.Namespace, parquet: pq.ParquetFile) -> dict[str, Any]:
    input_rows = parquet.metadata.num_rows
    if args.expected_input_rows and input_rows != args.expected_input_rows:
        raise ValueError(
            f"input row count is {input_rows}, expected {args.expected_input_rows}; "
            "pass --expected-input-rows 0 only if this is an intentional dataset revision"
        )

    missing = REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"input parquet is not in the expected verl schema; missing: {sorted(missing)}")

    seen: dict[str, list[Any]] = {}
    conflicts: dict[str, dict[str, Any]] = {}
    global_index = 0

    scan_columns = ["prompt", "reward_model"]
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=scan_columns):
        prompts = column_values(batch, "prompt")
        rewards = column_values(batch, "reward_model")

        for prompt, reward in zip(prompts, rewards, strict=True):
            key = prompt_key(prompt, global_index)
            answer = ground_truth_key(reward, global_index)

            state = seen.get(key)
            if state is None:
                seen[key] = [answer, global_index, 1]
            else:
                state[2] += 1
                if state[0] != answer and key not in conflicts:
                    conflicts[key] = {
                        "first_row": state[1],
                        "conflicting_row": global_index,
                        "first_ground_truth": json.loads(state[0]),
                        "conflicting_ground_truth": json.loads(answer),
                    }

            global_index += 1

    if global_index != input_rows:
        raise RuntimeError(f"scanned {global_index} rows but parquet metadata reports {input_rows}")
    unique_prompt_rows = len(seen)
    selected_indices = sorted(state[1] for key, state in seen.items() if key not in conflicts)
    clean_rows = len(selected_indices)
    if args.expected_output_rows and clean_rows != args.expected_output_rows:
        raise ValueError(
            f"exact prompt dedup plus conflict filtering produced {clean_rows} rows, "
            f"expected {args.expected_output_rows}; "
            "the input may differ from the released parquet"
        )

    occurrence_distribution = Counter(state[2] for state in seen.values())
    duplicate_prompt_groups = sum(1 for state in seen.values() if state[2] > 1)

    return {
        "selected_indices": selected_indices,
        "audit": {
            "input_rows": input_rows,
            "exact_unique_prompt_rows": unique_prompt_rows,
            "conflicting_prompt_groups_dropped": len(conflicts),
            "clean_output_rows": clean_rows,
            "rows_removed_total": input_rows - clean_rows,
            "duplicate_prompt_groups": duplicate_prompt_groups,
            "ground_truth_conflicts": len(conflicts),
            "conflict_examples": list(conflicts.values())[:20],
            "occurrence_count_distribution": {
                str(count): groups for count, groups in sorted(occurrence_distribution.items())
            },
        },
    }


def write_selected_rows(
    input_path: Path,
    output_path: Path,
    selected_indices: list[int],
    batch_size: int,
) -> None:
    selected = set(selected_indices)
    last_selected = selected_indices[-1]
    parquet = pq.ParquetFile(input_path)
    writer = pq.ParquetWriter(output_path, parquet.schema_arrow, compression="zstd")
    global_offset = 0
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            local_indices = [
                index - global_offset
                for index in selected
                if global_offset <= index < global_offset + batch.num_rows
            ]
            if local_indices:
                local_indices.sort()
                output_batch = batch.take(pa.array(local_indices, type=pa.int64()))
                writer.write_batch(output_batch)
                written += output_batch.num_rows
            global_offset += batch.num_rows
            if global_offset > last_selected:
                break
    finally:
        writer.close()

    if written != len(selected_indices):
        raise RuntimeError(f"wrote {written} rows, expected {len(selected_indices)}")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "fail",
        "input": str(args.input),
        "output": str(args.output),
    }

    try:
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}; pass --overwrite to replace it")
        if args.report.exists() and not args.overwrite:
            raise FileExistsError(f"report already exists: {args.report}; pass --overwrite to replace it")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        parquet = pq.ParquetFile(args.input)
        scan = scan_input(args, parquet)

        if args.output.exists():
            args.output.unlink()
        write_selected_rows(args.input, args.output, scan["selected_indices"], args.batch_size)

        output_parquet = pq.ParquetFile(args.output)
        schema_preserved = output_parquet.schema_arrow.equals(parquet.schema_arrow, check_metadata=True)
        if output_parquet.metadata.num_rows != scan["audit"]["clean_output_rows"]:
            raise RuntimeError("output parquet row count failed post-write validation")
        if not schema_preserved:
            raise RuntimeError("output parquet Arrow schema differs from the input schema")

        report.update(scan["audit"])
        report.update(
            {
                "status": "pass",
                "schema_preserved": True,
                "input_sha256": file_sha256(args.input),
                "output_sha256": file_sha256(args.output),
            }
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if args.output.exists():
            args.output.unlink()
        write_report(args.report, report)
        raise

    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
