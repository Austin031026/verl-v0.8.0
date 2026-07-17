from verl.utils.reward_score.gsm8k import compute_score, extract_solution


def test_boxed_answer_is_the_default():
    response = "The calculation gives 9 * 2 = 18.\n\\boxed{18}"

    assert extract_solution(response, method="boxed") == "18"
    assert compute_score(response, "18") == 1.0


def test_boxed_answer_takes_priority_over_trailing_numbers():
    response = "\\boxed{64}\nKylar pays $64 for 16 glasses."

    assert extract_solution(response, method="boxed") == "64"


def test_boxed_answer_normalizes_common_numeric_formatting():
    response = "Final answer:\n\\boxed{\\$1,200.00 \\text{ dollars}}"

    assert extract_solution(response, method="boxed") == "1200"
    assert compute_score(response, "1200") == 1.0


def test_legacy_hash_answer_remains_supported():
    assert extract_solution("#### 18", method="boxed") == "18"
    assert extract_solution("#### The answer is **$18**.", method="boxed") == "18"


def test_markdown_heading_is_not_treated_as_an_answer():
    assert extract_solution("#### 3. Calculate the revenue", method="boxed") is None


def test_boxed_method_falls_back_to_the_last_unmarked_number():
    assert extract_solution("Therefore, Janet makes $18 every day.", method="boxed") == "18"


def test_strict_method_keeps_legacy_behavior():
    assert extract_solution("#### 18", method="strict") == "18"
    assert extract_solution("\\boxed{18}", method="strict") is None


def test_flexible_method_normalizes_the_last_number():
    assert extract_solution("The final answer is $1,200.00.", method="flexible") == "1200"
