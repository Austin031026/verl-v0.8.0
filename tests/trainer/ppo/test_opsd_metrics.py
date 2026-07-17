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

import pytest
import torch

from verl.trainer.ppo.opsd_metrics import (
    build_opsd_topk_metric_objects,
    compute_opsd_topk_masses,
    finalize_opsd_actor_metrics,
)


def test_opsd_topk_coverage_uses_full_distribution_log_probs():
    student_log_probs = torch.tensor([[[0.6, 0.2], [0.5, 0.25]]]).log()
    teacher_log_probs = torch.tensor([[[0.4, 0.1], [0.2, 0.2]]]).log()

    student_mass, teacher_mass = compute_opsd_topk_masses(student_log_probs, teacher_log_probs)

    torch.testing.assert_close(student_mass, torch.tensor([[0.8, 0.75]]))
    torch.testing.assert_close(teacher_mass, torch.tensor([[0.5, 0.4]]))


def test_opsd_topk_metrics_apply_response_mask():
    metrics = build_opsd_topk_metric_objects(
        student_topk_mass=torch.tensor([[0.8, 0.75]]),
        teacher_mass_on_student_topk=torch.tensor([[0.5, 0.4]]),
        response_mask=torch.tensor([[True, False]]),
    )

    assert metrics["opsd_student_topk_mass_sum"].aggregate() == pytest.approx(0.8)
    assert metrics["opsd_student_topk_mass_min"].aggregate() == pytest.approx(0.8)
    assert metrics["opsd_student_topk_mass_max"].aggregate() == pytest.approx(0.8)
    assert metrics["opsd_teacher_mass_on_student_topk_sum"].aggregate() == pytest.approx(0.5)


def test_finalize_opsd_actor_metrics_builds_token_weighted_means():
    finalized = finalize_opsd_actor_metrics(
        {
            "actor/opsd_token_loss_sum": 0.6,
            "actor/opsd_response_tokens": 2,
            "actor/opsd_student_topk_mass_sum": 1.5,
            "actor/opsd_teacher_mass_on_student_topk_sum": 0.9,
        }
    )

    assert finalized == pytest.approx(
        {
            "actor/opsd_token_loss_mean": 0.3,
            "actor/opsd_student_topk_mass_mean": 0.75,
            "actor/opsd_teacher_mass_on_student_topk_mean": 0.45,
        }
    )
