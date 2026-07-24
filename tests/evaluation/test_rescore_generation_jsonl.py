from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER = REPO_ROOT / "examples" / "evaluation" / "rescore_generation_jsonl.py"


class RescoreGenerationJsonlTest(unittest.TestCase):
    def test_scores_every_rollout_and_reuses_matching_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "math500.jsonl"
            parser_path = root / "parser.py"
            details_path = root / "details.jsonl"
            summary_path = root / "summary.json"

            parser_path.write_text(
                "\n".join(
                    [
                        "def strip_string(value):",
                        "    return str(value).strip()",
                        "",
                        "def extract_answer(value, data_name=''):",
                        "    assert data_name == 'math'",
                        "    return value.rsplit('ANSWER:', 1)[-1].strip() if 'ANSWER:' in value else ''",
                        "",
                        "def math_equal(prediction, reference):",
                        "    return prediction == reference",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "sample_uid": "q1",
                    "rollout_id": 0,
                    "checkpoint_id": "step_20",
                    "benchmark_id": "math500",
                    "output": "reasoning\nANSWER: 1",
                    "gts": "1",
                    "score": 0,
                },
                {
                    "sample_uid": "q1",
                    "rollout_id": 1,
                    "checkpoint_id": "step_20",
                    "benchmark_id": "math500",
                    "output": "ANSWER: 0",
                    "gts": "1",
                    "score": 1,
                },
                {
                    "sample_uid": "q2",
                    "rollout_id": 0,
                    "checkpoint_id": "step_20",
                    "benchmark_id": "math500",
                    "output": "",
                    "gts": "2",
                    "score": 1,
                },
                {
                    "sample_uid": "q2",
                    "rollout_id": 1,
                    "checkpoint_id": "step_20",
                    "benchmark_id": "math500",
                    "output": "ANSWER: 2",
                    "gts": "2",
                    "score": 0,
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            parser_check = subprocess.run(
                [
                    sys.executable,
                    str(SCORER),
                    "--ye-parser-path",
                    str(parser_path),
                    "--check-parser-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("YE_PARSER_OK", parser_check.stdout)

            command = [
                sys.executable,
                str(SCORER),
                "--input-jsonl",
                str(input_path),
                "--benchmark-id",
                "math500",
                "--data-name",
                "math",
                "--expected-prompts",
                "2",
                "--samples-per-prompt",
                "2",
                "--ye-parser-path",
                str(parser_path),
                "--details-jsonl",
                str(details_path),
                "--summary-json",
                str(summary_path),
                "--reuse-if-current",
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("YE SCORE", first.stdout)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["metrics"]
            self.assertEqual(metrics["correct_rollouts"], 2)
            self.assertEqual(metrics["pass_at_1_estimator"], 0.5)
            self.assertEqual(metrics["first_draw_correct"], 1)
            self.assertEqual(metrics["first_draw_accuracy"], 0.5)
            self.assertEqual(metrics["questions_with_any_correct"], 2)
            self.assertEqual(metrics["empirical_pass_at_k"], 1.0)
            self.assertEqual(metrics["unextractable_rollouts"], 1)

            details = [
                json.loads(line)
                for line in details_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["ye_correct"] for row in details], [True, False, False, True])
            self.assertNotIn("output", details[0])
            self.assertNotIn("gts", details[0])
            self.assertEqual(details[0]["ye_data_name"], "math")

            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("REUSE", second.stdout)


if __name__ == "__main__":
    unittest.main()
