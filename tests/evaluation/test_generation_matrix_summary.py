from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = REPO_ROOT / "examples" / "evaluation" / "run_generation_matrix.py"


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("_run_generation_matrix", MATRIX_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerationMatrixSummaryTest(unittest.TestCase):
    def test_checked_in_qwen3_4b_validations_are_no_thinking_sampled(self) -> None:
        validation_dir = (
            REPO_ROOT / "examples" / "evaluation" / "configs" / "validation"
        )
        for path in sorted(validation_dir.glob("qwen3_4b_*.json")):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(config["generation"]["enable_thinking"], path)
            self.assertTrue(config["generation"]["do_sample"], path)
            self.assertEqual(config["generation"]["temperature"], 0.7, path)
            self.assertEqual(config["generation"]["top_p"], 0.8, path)
            self.assertEqual(config["generation"]["top_k"], 20, path)
            self.assertEqual(config["generation"]["min_p"], 0.0, path)
            self.assertGreater(config["generation"]["n_samples"], 1, path)

    def test_separates_environment_from_validation_parameters(self) -> None:
        matrix = load_matrix_module()
        environment = {
            "schema_version": 1,
            "environment_id": "cluster_6gpu",
            "workspace_root": "/cluster/workspace",
            "python": {
                "verl": "envs/verl/bin/python",
                "ye_rescore": "envs/ye/bin/python",
            },
            "paths": {
                "ye_parser": "Lulu/parser.py",
                "results_root": "results",
                "work_root": "work",
            },
            "hardware": {
                "nnodes": 1,
                "gpus_per_node": 6,
                "physical_vram_gib": 80,
            },
        }
        validation = {
            "schema_version": 1,
            "validation_id": "validation_1",
            "algorithm_id": "opsd",
            "model_id": "qwen3_4b",
            "training_run_id": "run_1",
            "checkpoint": {
                "root": "checkpoints/run_1",
                "backend": "fsdp",
                "actor_subdir": "actor",
                "steps": [],
            },
            "generation": {
                "enable_thinking": False,
                "do_sample": True,
                "temperature": 0.7,
                "min_p": 0.0,
                "max_response_length": 4096,
                "n_samples": 16,
            },
            "rollout_runtime": {
                "tensor_parallel_size": 1,
                "generation_batch_size": 9,
                "max_num_seqs_per_replica": 24,
                "max_num_batched_tokens": 8192,
                "gpu_memory_utilization": 0.7,
                "runtime_reserve_gib": 12,
            },
            "rescoring": {"enabled": True},
        }

        resolved = matrix.resolve_evaluation_config(environment, validation)

        self.assertEqual(resolved["environment_id"], "cluster_6gpu")
        self.assertEqual(resolved["validation_id"], "validation_1")
        self.assertEqual(
            resolved["checkpoint"]["root"],
            "/cluster/workspace/checkpoints/run_1",
        )
        self.assertEqual(
            resolved["rescoring"]["python_path"],
            "/cluster/workspace/envs/ye/bin/python",
        )
        self.assertEqual(resolved["runtime"]["gpus_per_node"], 6)
        self.assertEqual(resolved["runtime"]["generation_batch_size"], 9)
        self.assertFalse(resolved["generation"]["enable_thinking"])
        self.assertTrue(resolved["generation"]["do_sample"])

        validation["workspace_root"] = "/must/not/live/in/validation"
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            matrix.resolve_evaluation_config(environment, validation)

        validation.pop("workspace_root")
        validation["generation"]["enable_thinking"] = "false"
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            matrix.resolve_evaluation_config(environment, validation)

        validation["generation"]["enable_thinking"] = False
        validation["generation"]["temperature"] = 0
        with self.assertRaisesRegex(ValueError, "temperature"):
            matrix.resolve_evaluation_config(environment, validation)

    def test_writes_one_summary_for_all_selected_benchmarks(self) -> None:
        matrix = load_matrix_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_status = {
                "step_20/math500": {
                    "status": "complete",
                    "ye_rescore": {
                        "data_name": "math",
                        "details_path": "/tmp/math500.ye_rescored.jsonl",
                        "summary_path": "/tmp/math500.ye_metrics.json",
                        "metrics": {
                            "pass_at_1_estimator": 0.7,
                            "correct_rollouts": 1400,
                            "total_rollouts": 2000,
                            "first_draw_accuracy": 0.68,
                            "empirical_pass_at_k": 0.82,
                            "questions_with_any_correct": 410,
                            "total_questions": 500,
                            "k": 4,
                        },
                    },
                },
                "step_20/gsm8k": {
                    "status": "complete",
                    "ye_rescore": {
                        "data_name": "gsm8k",
                        "details_path": "/tmp/gsm8k.ye_rescored.jsonl",
                        "summary_path": "/tmp/gsm8k.ye_metrics.json",
                        "metrics": {
                            "pass_at_1_estimator": 0.8,
                            "correct_rollouts": 4221,
                            "total_rollouts": 5276,
                            "first_draw_accuracy": 0.79,
                            "empirical_pass_at_k": 0.9,
                            "questions_with_any_correct": 1187,
                            "total_questions": 1319,
                            "k": 4,
                        },
                    },
                },
            }

            summary_path, summary = matrix.write_checkpoint_summary(
                final_root=root,
                training_run_id="run_1",
                checkpoint_step=20,
                benchmark_ids=["math500", "gsm8k"],
                task_status=task_status,
            )

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["selected_benchmarks"], ["math500", "gsm8k"])
            self.assertEqual(
                summary["benchmarks"]["math500"]["metrics"]["pass_at_1_estimator"],
                0.7,
            )
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, summary)

    def test_marks_failed_benchmark_incomplete(self) -> None:
        matrix = load_matrix_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_status = {
                "step_40/math500": {
                    "status": "complete",
                    "ye_rescore": {
                        "data_name": "math",
                        "details_path": "/tmp/math500.ye_rescored.jsonl",
                        "summary_path": "/tmp/math500.ye_metrics.json",
                        "metrics": {
                            "pass_at_1_estimator": 0.7,
                            "correct_rollouts": 1400,
                            "total_rollouts": 2000,
                            "first_draw_accuracy": 0.68,
                            "empirical_pass_at_k": 0.82,
                            "questions_with_any_correct": 410,
                            "total_questions": 500,
                            "k": 4,
                        },
                    },
                },
                "step_40/gsm8k": {
                    "status": "rescoring_failed",
                    "error": "parser dependency missing",
                },
            }

            _, summary = matrix.write_checkpoint_summary(
                final_root=root,
                training_run_id="run_1",
                checkpoint_step=40,
                benchmark_ids=["math500", "gsm8k"],
                task_status=task_status,
            )

            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(
                summary["benchmarks"]["gsm8k"]["status"],
                "rescoring_failed",
            )
            self.assertEqual(
                summary["benchmarks"]["gsm8k"]["error"],
                "parser dependency missing",
            )


if __name__ == "__main__":
    unittest.main()
