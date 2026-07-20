#!/usr/bin/env python3
"""Expand scorer-free generation parquet rows into auditable rollout JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested_value(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        contents = []
        for message in prompt:
            if isinstance(message, dict) and "content" in message:
                contents.append(str(message["content"]))
            else:
                contents.append(str(message))
        return "\n".join(contents)
    return str(prompt)


def sample_uid(row: dict[str, Any], row_index: int, benchmark_id: str) -> str:
    candidates = (
        row.get("uid"),
        nested_value(row, "extra_info.unique_id"),
        nested_value(row, "extra_info.uid"),
        nested_value(row, "extra_info.index"),
    )
    for candidate in candidates:
        if candidate is not None and candidate != "":
            return str(candidate)
    return f"{benchmark_id}:{row_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--responses-key", default="responses")
    parser.add_argument("--ground-truth-field", default="reward_model.ground_truth")
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"Generation parquet not found: {args.input_parquet}")

    import pyarrow.parquet as pq

    rows = pq.read_table(args.input_parquet).to_pylist()
    if len(rows) != args.expected_prompts:
        raise ValueError(f"Expected {args.expected_prompts} prompt rows, found {len(rows)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    output_rows = 0
    seen_sample_uids = set()

    with temporary_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            if args.prompt_key not in row:
                raise ValueError(f"Row {row_index} is missing prompt key {args.prompt_key!r}")
            responses = row.get(args.responses_key)
            if not isinstance(responses, list) or len(responses) != args.samples_per_prompt:
                count = len(responses) if isinstance(responses, list) else None
                raise ValueError(
                    f"Row {row_index} has {count} responses; expected {args.samples_per_prompt}"
                )

            ground_truth = nested_value(row, args.ground_truth_field)
            if ground_truth is None or ground_truth == "":
                raise ValueError(f"Row {row_index} is missing {args.ground_truth_field!r}")

            prompt = row[args.prompt_key]
            uid = sample_uid(row, row_index, args.benchmark_id)
            if uid in seen_sample_uids:
                raise ValueError(f"Duplicate sample_uid {uid!r} at row {row_index}")
            seen_sample_uids.add(uid)
            for rollout_id, response in enumerate(responses):
                record = {
                    "sample_uid": uid,
                    "rollout_id": rollout_id,
                    "data_source": row.get("data_source"),
                    "input": prompt_text(prompt),
                    "prompt": prompt,
                    "output": "" if response is None else str(response),
                    "gts": ground_truth,
                }
                if row.get("extra_info") is not None:
                    record["extra_info"] = row["extra_info"]
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                output_rows += 1

    expected_output_rows = args.expected_prompts * args.samples_per_prompt
    if output_rows != expected_output_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {expected_output_rows} JSONL rows, wrote {output_rows}")
    temporary_path.replace(args.output_jsonl)
    print(f"Exported {output_rows} rollout rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
