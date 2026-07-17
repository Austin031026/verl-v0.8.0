"""Conservative numeric reward for AIME answers in the range 0..999."""

from __future__ import annotations

import re
from typing import Any


ANSWER_MARKER_RE = re.compile(r"(?i)\b(?:final\s+)?answer\s*(?::|is\b)")
BOXED_INTEGER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d{1,3})\s*\}")
PLAIN_BOXED_INTEGER_RE = re.compile(r"\\boxed\s+([+-]?\d{1,3})(?![\d.])")
ANSWER_LINE_INTEGER_RE = re.compile(
    r"^([+-]?\d{1,3})(?![\d.])[\s$*_`{}()\[\]\\,:;.!]*(?:$|\n)"
)
FINAL_LINE_INTEGER_RE = re.compile(
    r"^[\s$#*_`{}()\[\]\\,:;.=+-]*([+-]?\d{1,3})[\s$*_`{}()\[\]\\,:;.!]*$"
)


def parse_aime_integer(value: Any) -> int | None:
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d{1,3}", text):
        return None
    number = int(text)
    return number if 0 <= number <= 999 else None


def extract_after_answer_marker(text: str) -> int | None:
    text = text.lstrip()
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

    match = ANSWER_LINE_INTEGER_RE.match(text)
    return parse_aime_integer(match.group(1)) if match is not None else None


def extract_aime_answer(output: str) -> tuple[int | None, str]:
    markers = list(ANSWER_MARKER_RE.finditer(output))
    if markers:
        answer = extract_after_answer_marker(output[markers[-1].end() :])
        if answer is not None:
            return answer, "answer_marker"

    boxed = list(BOXED_INTEGER_RE.finditer(output))
    if boxed:
        return parse_aime_integer(boxed[-1].group(1)), "boxed"

    plain_boxed = list(PLAIN_BOXED_INTEGER_RE.finditer(output))
    if plain_boxed:
        return parse_aime_integer(plain_boxed[-1].group(1)), "boxed"

    nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if nonempty_lines:
        match = FINAL_LINE_INTEGER_RE.fullmatch(nonempty_lines[-1])
        if match is not None:
            return parse_aime_integer(match.group(1)), "final_line"

    return None, "unextracted"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source, extra_info, kwargs
    gold = parse_aime_integer(ground_truth)
    if gold is None:
        raise ValueError(f"Invalid AIME ground truth: {ground_truth!r}")

    prediction, extraction_method = extract_aime_answer(solution_str)
    correct = prediction == gold

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": str(prediction) if prediction is not None else "[INVALID]",
        "extraction_method": extraction_method,
        "answer_extracted": prediction is not None,
    }
