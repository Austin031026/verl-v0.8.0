#!/usr/bin/env python3
"""Prepare the released AIME-2024 data for verl best-of-N validation.

The released ``BytedTsinghua-SIA/AIME-2024`` parquet contains the 30 AIME
2024 problems repeated 32 times. Modern verl validation can repeat each
unique prompt with ``actor_rollout_ref.rollout.val_kwargs.n`` and preserve a
shared UID for metric aggregation, so this script writes one row per prompt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import datasets


DATASET_NAME = "BytedTsinghua-SIA/AIME-2024"
REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_save_dir",
        type=Path,
        default=Path("~/data/aime-2024").expanduser(),
        help="Directory for the unique parquet and audit report.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local parquet file or datasets-compatible dataset path.",
    )
    parser.add_argument("--output_name", default="aime-2024-unique-30.parquet")
    parser.add_argument("--report_name", default="aime-2024-prepare-report.json")
    parser.add_argument("--expected_source_rows", type=int, default=960)
    parser.add_argument("--expected_unique_rows", type=int, default=30)
    parser.add_argument("--expected_repeats_per_prompt", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_source(local_dataset_path: str | None) -> datasets.Dataset:
    if local_dataset_path is None:
        return datasets.load_dataset(DATASET_NAME, "default", split="train")

    local_path = Path(local_dataset_path).expanduser()
    if local_path.is_file():
        return datasets.load_dataset("parquet", data_files=str(local_path), split="train")

    return datasets.load_dataset(local_dataset_path, "default", split="train")


def prepare_unique_dataset(source: datasets.Dataset) -> tuple[datasets.Dataset, dict[str, Any]]:
    missing = REQUIRED_COLUMNS - set(source.column_names)
    if missing:
        raise ValueError(f"source dataset is missing required verl columns: {sorted(missing)}")

    prompt_states: dict[str, dict[str, Any]] = {}
    selected_indices = []
    conflicts = []

    for index, row in enumerate(source):
        prompt = row["prompt"]
        reward_model = row["reward_model"]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"row {index}: prompt must be a non-empty message list")
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise ValueError(f"row {index}: reward_model.ground_truth is missing")

        prompt_key = canonical_json(prompt)
        ground_truth_key = canonical_json(reward_model["ground_truth"])
        state = prompt_states.get(prompt_key)

        if state is None:
            prompt_states[prompt_key] = {
                "ground_truth": ground_truth_key,
                "first_index": index,
                "count": 1,
            }
            selected_indices.append(index)
        else:
            state["count"] += 1
            if state["ground_truth"] != ground_truth_key:
                conflicts.append(
                    {
                        "first_index": state["first_index"],
                        "conflicting_index": index,
                        "first_ground_truth": json.loads(state["ground_truth"]),
                        "conflicting_ground_truth": reward_model["ground_truth"],
                    }
                )

    if conflicts:
        raise ValueError(f"found {len(conflicts)} prompt groups with conflicting ground truths")

    occurrence_distribution = Counter(state["count"] for state in prompt_states.values())
    unique_dataset = source.select(selected_indices)
    report = {
        "dataset": DATASET_NAME,
        "source_rows": len(source),
        "unique_prompt_rows": len(unique_dataset),
        "duplicate_rows_removed": len(source) - len(unique_dataset),
        "ground_truth_conflicts": 0,
        "occurrence_count_distribution": {
            str(count): groups for count, groups in sorted(occurrence_distribution.items())
        },
        "columns": unique_dataset.column_names,
    }
    return unique_dataset, report


def validate_release(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.expected_source_rows and report["source_rows"] != args.expected_source_rows:
        raise ValueError(
            f"source has {report['source_rows']} rows, expected {args.expected_source_rows}; "
            "pass --expected_source_rows 0 only for an intentional dataset revision"
        )
    if args.expected_unique_rows and report["unique_prompt_rows"] != args.expected_unique_rows:
        raise ValueError(
            f"deduplication produced {report['unique_prompt_rows']} unique prompts, "
            f"expected {args.expected_unique_rows}"
        )

    distribution = report["occurrence_count_distribution"]
    expected_distribution = {str(args.expected_repeats_per_prompt): args.expected_unique_rows}
    if args.expected_repeats_per_prompt and distribution != expected_distribution:
        raise ValueError(
            f"unexpected prompt repetition distribution: {distribution}; "
            f"expected {expected_distribution}"
        )


def main() -> None:
    args = parse_args()
    output_path = args.local_save_dir / args.output_name
    report_path = args.local_save_dir / args.report_name

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"report already exists: {report_path}; pass --overwrite to replace it")

    source = load_source(args.local_dataset_path)
    unique_dataset, report = prepare_unique_dataset(source)
    validate_release(args, report)

    args.local_save_dir.mkdir(parents=True, exist_ok=True)
    unique_dataset.to_parquet(output_path)
    report.update(
        {
            "status": "pass",
            "output": str(output_path),
            "report": str(report_path),
            "recommended_validation_n": 32,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
