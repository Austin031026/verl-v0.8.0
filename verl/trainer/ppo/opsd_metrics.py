# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Scalar training metrics for the pure OPSD path."""

from typing import Any

import torch

from verl.utils.metric import AggregationType, Metric


def compute_opsd_topk_masses(
    student_topk_log_probs: torch.Tensor,
    teacher_log_probs_on_student_topk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probability mass covered by the student-selected top-k IDs."""
    if student_topk_log_probs.shape != teacher_log_probs_on_student_topk.shape:
        raise ValueError(
            "OPSD top-k coverage requires matching student/teacher log-prob shapes, "
            f"got student={tuple(student_topk_log_probs.shape)}, "
            f"teacher={tuple(teacher_log_probs_on_student_topk.shape)}."
        )
    if student_topk_log_probs.dim() < 2:
        raise ValueError(
            "OPSD top-k coverage requires a top-k vocabulary dimension, "
            f"got shape={tuple(student_topk_log_probs.shape)}."
        )

    student_mass = student_topk_log_probs.detach().exp().sum(dim=-1)
    teacher_mass = teacher_log_probs_on_student_topk.detach().exp().sum(dim=-1)
    return student_mass, teacher_mass


def build_opsd_topk_metric_objects(
    student_topk_mass: torch.Tensor,
    teacher_mass_on_student_topk: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, Metric]:
    """Build DP-aware scalar metric objects from padded per-token masses."""
    response_mask = response_mask.bool()
    if not (
        student_topk_mass.shape == teacher_mass_on_student_topk.shape == response_mask.shape
    ):
        raise ValueError(
            "OPSD top-k mass tensors must match response_mask, "
            f"got student={tuple(student_topk_mass.shape)}, "
            f"teacher={tuple(teacher_mass_on_student_topk.shape)}, "
            f"mask={tuple(response_mask.shape)}."
        )

    valid_student_mass = student_topk_mass[response_mask]
    valid_teacher_mass = teacher_mass_on_student_topk[response_mask]
    if valid_student_mass.numel() == 0:
        raise ValueError("OPSD top-k coverage requires at least one valid response token.")

    return {
        "opsd_student_topk_mass_sum": Metric(AggregationType.SUM, valid_student_mass.sum()),
        "opsd_student_topk_mass_min": Metric(AggregationType.MIN, valid_student_mass.min()),
        "opsd_student_topk_mass_max": Metric(AggregationType.MAX, valid_student_mass.max()),
        "opsd_teacher_mass_on_student_topk_sum": Metric(AggregationType.SUM, valid_teacher_mass.sum()),
        "opsd_teacher_mass_on_student_topk_min": Metric(AggregationType.MIN, valid_teacher_mass.min()),
        "opsd_teacher_mass_on_student_topk_max": Metric(AggregationType.MAX, valid_teacher_mass.max()),
    }


def finalize_opsd_actor_metrics(actor_metrics: dict[str, Any]) -> dict[str, float]:
    """Convert worker sums into token-weighted means for the shared trainer log."""
    token_loss_sum = actor_metrics.get("actor/opsd_token_loss_sum")
    response_tokens = actor_metrics.get("actor/opsd_response_tokens")
    if token_loss_sum is None or response_tokens is None:
        raise KeyError("OPSD actor metrics require token loss sum and response token count.")
    if response_tokens <= 0:
        raise ValueError("OPSD actor metrics require at least one response token.")

    finalized = {"actor/opsd_token_loss_mean": token_loss_sum / response_tokens}
    coverage_sum_keys = (
        "actor/opsd_student_topk_mass_sum",
        "actor/opsd_teacher_mass_on_student_topk_sum",
    )
    present_coverage_keys = [key for key in coverage_sum_keys if key in actor_metrics]
    if present_coverage_keys and len(present_coverage_keys) != len(coverage_sum_keys):
        raise KeyError("OPSD top-k coverage requires both student and teacher mass sums.")
    if present_coverage_keys:
        finalized.update(
            {
                "actor/opsd_student_topk_mass_mean": actor_metrics[coverage_sum_keys[0]] / response_tokens,
                "actor/opsd_teacher_mass_on_student_topk_mean": (
                    actor_metrics[coverage_sum_keys[1]] / response_tokens
                ),
            }
        )
    return finalized
