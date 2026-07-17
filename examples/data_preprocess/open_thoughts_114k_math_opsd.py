#!/usr/bin/env python3
"""Convert the OpenThoughts-114k math subset to verl parquet for OPSD."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


DATASET_ID = "open-thoughts/OpenThoughts-114k"
DATASET_CONFIG = "metadata"
DATASET_SPLIT = "train"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-save-dir",
        required=True,
        help="Directory for train.parquet, train_example.json, and conversion_report.json.",
    )
    parser.add_argument(
        "--local-dataset-path",
        default=None,
        help="Optional local dataset path to use instead of downloading from Hugging Face.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional output limit for a small conversion smoke test.",
    )
    return parser.parse_args()


def extract_last_boxed_answer(solution: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` or ``\\fbox{...}``."""
    marker_index = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if marker_index < 0:
        return None

    left_brace = solution.find("{", marker_index)
    if left_brace < 0:
        return None

    depth = 0
    for index in range(left_brace, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                answer = solution[left_brace + 1 : index].strip()
                return answer or None
    return None


def is_math_example(example: dict[str, Any]) -> bool:
    domain = example.get("domain")
    return isinstance(domain, str) and domain.strip().casefold() == "math"


def is_usable_math_example(example: dict[str, Any]) -> bool:
    if not is_math_example(example):
        return False

    problem = example.get("problem")
    reason = example.get("deepseek_reasoning")
    ground_truth_solution = example.get("ground_truth_solution")
    return bool(
        isinstance(problem, str)
        and problem.strip()
        and isinstance(reason, str)
        and reason.strip()
        and isinstance(ground_truth_solution, str)
        and extract_last_boxed_answer(ground_truth_solution)
    )


def make_verl_row(example: dict[str, Any], index: int) -> dict[str, Any]:
    problem = example["problem"].strip()
    reason = example["deepseek_reasoning"].strip()
    answer = extract_last_boxed_answer(example["ground_truth_solution"])
    if answer is None:
        raise ValueError(f"row {index}: ground_truth_solution has no balanced boxed answer")

    source = example.get("source")
    return {
        "data_source": "math",
        "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
            "reason": reason,
        },
        "extra_info": {
            "dataset": DATASET_ID,
            "source": source.strip() if isinstance(source, str) else None,
            "split": DATASET_SPLIT,
            "index": index,
        },
    }


def load_source_dataset(local_dataset_path: str | None):
    import datasets

    dataset_path = local_dataset_path or DATASET_ID
    return datasets.load_dataset(dataset_path, DATASET_CONFIG, split=DATASET_SPLIT)


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    source_dataset = load_source_dataset(args.local_dataset_path)
    input_rows = len(source_dataset)
    math_dataset = source_dataset.filter(is_math_example, desc="Selecting math rows")
    usable_dataset = math_dataset.filter(is_usable_math_example, desc="Validating OPSD fields")
    usable_rows = len(usable_dataset)
    if usable_rows == 0:
        raise ValueError("no usable math rows found; check the dataset version and field schema")

    if args.max_samples is not None:
        usable_dataset = usable_dataset.select(range(min(args.max_samples, usable_rows)))

    output_dataset = usable_dataset.map(
        make_verl_row,
        with_indices=True,
        remove_columns=usable_dataset.column_names,
        desc="Converting to verl OPSD schema",
    )

    output_dir = os.path.abspath(os.path.expanduser(args.local_save_dir))
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "train.parquet")
    example_path = os.path.join(output_dir, "train_example.json")
    report_path = os.path.join(output_dir, "conversion_report.json")

    output_dataset.to_parquet(parquet_path)
    with open(example_path, "w", encoding="utf-8") as stream:
        json.dump(output_dataset[0], stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    report = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "input_rows": input_rows,
        "math_rows": len(math_dataset),
        "dropped_unusable_math_rows": len(math_dataset) - usable_rows,
        "output_rows": len(output_dataset),
        "output": parquet_path,
    }
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
