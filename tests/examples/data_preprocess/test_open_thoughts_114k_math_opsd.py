import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "examples"
    / "data_preprocess"
    / "open_thoughts_114k_math_opsd.py"
)
SPEC = importlib.util.spec_from_file_location("open_thoughts_114k_math_opsd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OpenThoughts114kMathOpsdTest(unittest.TestCase):
    def test_make_verl_row_keeps_only_opsd_and_verl_fields(self):
        example = {
            "problem": "  What is 6 times 7?  ",
            "deepseek_reasoning": "  Multiplication gives the result.  ",
            "deepseek_solution": "This generated answer is intentionally discarded.",
            "ground_truth_solution": "Work gives \\boxed{42}.",
            "domain": "math",
            "source": "AI-MO/NuminaMath-CoT",
            "test_cases": "discard me",
            "starter_code": "discard me",
        }

        row = MODULE.make_verl_row(example, 3)

        self.assertEqual(set(row), {"data_source", "prompt", "ability", "reward_model", "extra_info"})
        self.assertEqual(row["data_source"], "math")
        self.assertEqual(row["reward_model"]["ground_truth"], "42")
        self.assertEqual(row["reward_model"]["reason"], "Multiplication gives the result.")
        self.assertEqual(row["extra_info"]["index"], 3)
        self.assertNotIn("deepseek_solution", str(row))
        self.assertNotIn("test_cases", str(row))
        self.assertNotIn("starter_code", str(row))

    def test_validation_rejects_non_math_missing_reason_and_unboxed_answer(self):
        valid = {
            "problem": "Question",
            "deepseek_reasoning": "Reason",
            "ground_truth_solution": "Nested answer: \\boxed{\\frac{1}{2}}",
            "domain": " Math ",
        }
        self.assertTrue(MODULE.is_usable_math_example(valid))
        self.assertEqual(MODULE.extract_last_boxed_answer(valid["ground_truth_solution"]), "\\frac{1}{2}")

        self.assertFalse(MODULE.is_usable_math_example({**valid, "domain": "science"}))
        self.assertFalse(MODULE.is_usable_math_example({**valid, "deepseek_reasoning": " "}))
        self.assertFalse(MODULE.is_usable_math_example({**valid, "ground_truth_solution": "answer 1/2"}))


if __name__ == "__main__":
    unittest.main()
