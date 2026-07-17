# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from decimal import Decimal, InvalidOperation

_SOLUTION_CLIP_CHARS = 2000
_NUMBER_PATTERN = r"-?\d[\d,]*(?:\.\d+)?"
_BOXED_PATTERN = re.compile(rf"\\boxed\{{\s*(?:\\?\$)?\s*({_NUMBER_PATTERN})")


def _normalize_answer(answer):
    if answer is None:
        return None

    answer = answer.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        value = Decimal(answer)
    except InvalidOperation:
        return None

    if value == 0:
        value = abs(value)
    return format(value.normalize(), "f")


def _extract_marked_line(solution_str, marker):
    for line in reversed(solution_str.splitlines()):
        line = line.strip().replace("**", "").replace("`", "")
        line = line.replace("✅", "").strip()

        if marker == "####":
            if not line.startswith("####"):
                continue
            answer_text = line[4:].strip()
            answer_text = re.sub(r"(?i)^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", answer_text)
        else:
            match = re.match(r"(?i)^#{0,6}\s*final\s+answer\s*(?:is|:)\s*(.*)$", line)
            if match is None:
                continue
            answer_text = match.group(1).strip()

        match = re.fullmatch(rf"(?:\\?\$)?\s*({_NUMBER_PATTERN})\s*[.!]?", answer_text)
        if match is not None:
            return match.group(1)
    return None


def _extract_flexible_number(solution_str):
    content_lines = []
    for line in solution_str.splitlines():
        if line.strip().startswith("####"):
            continue
        content_lines.append(line)

    answers = re.findall(_NUMBER_PATTERN, "\n".join(content_lines))
    return answers[-1] if answers else None


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "boxed", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # Only inspect the response tail, where the final answer should appear.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "boxed":
        solutions = _BOXED_PATTERN.findall(solution_str)
        if solutions:
            final_answer = solutions[-1]
        else:
            final_answer = _extract_marked_line(solution_str, marker="####")
            if final_answer is None:
                final_answer = _extract_marked_line(solution_str, marker="final_answer")
            if final_answer is None:
                final_answer = _extract_flexible_number(solution_str)
    elif method == "flexible":
        final_answer = _extract_flexible_number(solution_str)
    return _normalize_answer(final_answer)


def compute_score(solution_str, ground_truth, method="boxed", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict', 'boxed', and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == _normalize_answer(ground_truth):
            return score
        else:
            return format_score
