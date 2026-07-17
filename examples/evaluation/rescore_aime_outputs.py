#!/usr/bin/env python3
"""Re-score AIME generations with a controlled, format-tolerant extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_MARKER_RE = re.compile(r"(?i)\b(?:final\s+)?answer\s*(?::|is\b)")
BOXED_INTEGER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d{1,3})\s*\}")
PLAIN_BOXED_INTEGER_RE = re.compile(r"\\boxed\s+([+-]?\d{1,3})(?![\d.])")
STANDALONE_INTEGER_RE = re.compile(r"^[\s$*_`{}()\[\]\\,:;.=+-]*([+-]?\d{1,3})[\s$*_`{}()\[\]\\,:;.!]*$")
LEADING_INTEGER_RE = re.compile(r"^([+-]?\d{1,3})(?![\d.])")


def parse_aime_integer(value: Any) -> int | None:
    """Parse a canonical AIME integer in the range 0..999."""
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d{1,3}", text):
        return None
    number = int(text)
    return number if 0 <= number <= 999 else None


def extract_leading_integer(text: str) -> int | None:
    """Extract an integer immediately following an explicit answer marker."""
    text = text.lstrip()

    # Handle outputs such as ``Answer: **Answer: 116**`` without accepting an
    # arbitrary integer later in the reasoning text.
    for _ in range(3):
        marker = ANSWER_MARKER_RE.match(text)
        if marker is None:
            break
        text = text[marker.end() :].lstrip()

    for prefix in ("**", "__", "`", "$$", "$", r"\(", r"\["):
        while text.startswith(prefix):
            text = text[len(prefix) :].lstrip()

    boxed = BOXED_INTEGER_RE.match(text)
    if boxed is not None:
        return parse_aime_integer(boxed.group(1))

    plain_boxed = PLAIN_BOXED_INTEGER_RE.match(text)
    if plain_boxed is not None:
        return parse_aime_integer(plain_boxed.group(1))

    match = LEADING_INTEGER_RE.match(text)
    return parse_aime_integer(match.group(1)) if match is not None else None


def extract_lenient_aime_answer(output: str) -> tuple[int | None, str]:
    """Extract a final AIME answer without using arbitrary reasoning numbers.

    Precedence is: the last explicit ``Answer:``/``final answer is`` containing
    an integer, then the last ``\\boxed{integer}``, then a standalone integer
    on the final non-empty line.
    """
    marker_matches = list(ANSWER_MARKER_RE.finditer(output))
    for marker in reversed(marker_matches):
        answer = extract_leading_integer(output[marker.end() :])
        if answer is not None:
            return answer, "answer_marker"

    boxed_matches = list(BOXED_INTEGER_RE.finditer(output))
    if boxed_matches:
        return parse_aime_integer(boxed_matches[-1].group(1)), "boxed"

    plain_boxed_matches = list(PLAIN_BOXED_INTEGER_RE.finditer(output))
    if plain_boxed_matches:
        return parse_aime_integer(plain_boxed_matches[-1].group(1)), "boxed"

    nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if nonempty_lines:
        match = STANDALONE_INTEGER_RE.fullmatch(nonempty_lines[-1])
        if match is not None:
            return parse_aime_integer(match.group(1)), "final_line"

    return None, "unextracted"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
    return rows


def mean(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def rescore_model(label: str, model: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    generations_path = Path(model["generations_path"])
    if not generations_path.exists():
        relocated_path = summary_path.parent / "models" / label / "validation" / "0.jsonl"
        if relocated_path.exists():
            generations_path = relocated_path
    rows = read_jsonl(generations_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    rescored_rows = []

    for row in rows:
        gold = parse_aime_integer(row.get("gts"))
        if gold is None:
            raise ValueError(f"{label}: invalid AIME ground truth {row.get('gts')!r}")

        prediction, source = extract_lenient_aime_answer(str(row.get("output", "")))
        strict_correct = bool(row.get("acc", False))
        extracted_correct = prediction == gold
        lenient_correct = strict_correct or extracted_correct
        source_counts[source] += 1

        rescored = {
            "input": str(row.get("input", "")),
            "gold": gold,
            "strict_correct": strict_correct,
            "lenient_prediction": prediction,
            "lenient_source": source,
            "lenient_correct": lenient_correct,
            "rescued_by_lenient": lenient_correct and not strict_correct,
        }
        rescored_rows.append(rescored)
        grouped[rescored["input"]].append(rescored)

    prompt_records = []
    for prompt, prompt_rows in grouped.items():
        strict_values = [row["strict_correct"] for row in prompt_rows]
        lenient_values = [row["lenient_correct"] for row in prompt_rows]
        prompt_records.append(
            {
                "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                "gold": prompt_rows[0]["gold"],
                "samples": len(prompt_rows),
                "strict_correct": sum(strict_values),
                "lenient_correct": sum(lenient_values),
                "strict_any_correct": any(strict_values),
                "lenient_any_correct": any(lenient_values),
            }
        )

    strict_values = [row["strict_correct"] for row in rescored_rows]
    lenient_values = [row["lenient_correct"] for row in rescored_rows]
    extracted_values = [row["lenient_prediction"] is not None for row in rescored_rows]
    rescued_values = [row["rescued_by_lenient"] for row in rescored_rows]

    return {
        "label": label,
        "generations_path": str(generations_path),
        "samples": len(rows),
        "unique_prompts": len(grouped),
        "strict_correct": sum(strict_values),
        "strict_pass_at_1": mean(strict_values),
        "strict_pass_at_n": mean([row["strict_any_correct"] for row in prompt_records]),
        "lenient_correct": sum(lenient_values),
        "lenient_pass_at_1": mean(lenient_values),
        "lenient_pass_at_n": mean([row["lenient_any_correct"] for row in prompt_records]),
        "rescued_correct": sum(rescued_values),
        "extracted_count": sum(extracted_values),
        "extraction_rate": mean(extracted_values),
        "extraction_sources": dict(sorted(source_counts.items())),
        "per_prompt": prompt_records,
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to the benchmark summary.json")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--show-problems", action="store_true", help="Print per-problem strict and lenient counts")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    samples_per_prompt = summary.get("protocol", {}).get("samples_per_prompt")
    models = {
        label: rescore_model(label, model, args.summary)
        for label, model in summary.get("models", {}).items()
    }
    result = {
        "source_summary": str(args.summary),
        "samples_per_prompt": samples_per_prompt,
        "definition": (
            "Lenient scoring accepts the last explicit Answer/final-answer integer, "
            "the last boxed integer, or a standalone integer on the final non-empty line. "
            "It never extracts an arbitrary integer from reasoning prose."
        ),
        "models": models,
    }

    output_path = args.output or args.summary.with_name("aime_lenient_rescore.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nAIME STRICT VS LENIENT RESULTS")
    print(
        f"{'model':28s} {'samples':>7s} {'strict':>8s} {'strict@1':>10s} "
        f"{'strict@n':>10s} {'lenient':>8s} {'lenient@1':>10s} {'lenient@n':>10s} {'rescued':>8s}"
    )
    for model in models.values():
        print(
            f"{model['label']:28s} {model['samples']:7d} {model['strict_correct']:8d} "
            f"{percent(model['strict_pass_at_1']):>10s} {percent(model['strict_pass_at_n']):>10s} "
            f"{model['lenient_correct']:8d} {percent(model['lenient_pass_at_1']):>10s} "
            f"{percent(model['lenient_pass_at_n']):>10s} {model['rescued_correct']:8d}"
        )
        print(
            f"  extraction: {model['extracted_count']}/{model['samples']} "
            f"({percent(model['extraction_rate'])}); sources={model['extraction_sources']}"
        )

        if args.show_problems:
            for index, prompt in enumerate(model["per_prompt"], 1):
                print(
                    f"  problem={index:02d} id={prompt['prompt_id']} gold={prompt['gold']} "
                    f"strict={prompt['strict_correct']}/{prompt['samples']} "
                    f"lenient={prompt['lenient_correct']}/{prompt['samples']}"
                )

    print(f"\nRESCORE_JSON={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
