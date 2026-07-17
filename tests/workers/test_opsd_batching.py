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

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.utils import (
    build_balanced_micro_batch_indices,
    build_response_prediction_indices,
    build_synchronized_balanced_micro_batch_indices,
    compute_masked_loss_stats,
    prepare_micro_batches,
    raise_if_any_rank_has_errors,
)
from verl.workers.engine_workers import ActorRolloutRefWorker


def _make_opsd_no_padding_batch(sequences, attention_mask, responses, response_mask):
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor(sequence, dtype=torch.long) for sequence in sequences], layout=torch.jagged
    )
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(len(sequence), dtype=torch.long) for sequence in sequences], layout=torch.jagged
    )
    return TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        },
        batch_size=[len(sequences)],
    )


def _distributed_opsd_batching_worker(rank: int, world_size: int, init_method: str):
    loopback_interface = next(name for _, name in socket.if_nameindex() if name.startswith("lo"))
    os.environ["GLOO_SOCKET_IFNAME"] = loopback_interface
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        if rank == 0:
            teacher_lengths = [7, 6, 3, 2]
            student_lengths = [4, 4, 3, 2]
            local_group_count = 2
        else:
            teacher_lengths = [9, 7, 5, 4]
            student_lengths = [5, 4, 3, 2]
            local_group_count = 3

        partitions = build_synchronized_balanced_micro_batch_indices(
            primary_lengths=teacher_lengths,
            secondary_lengths=student_lengths,
            local_num_micro_batches=local_group_count,
            primary_token_budget=10,
            secondary_token_budget=8,
            dp_group=dist.group.WORLD,
            collective_device="cpu",
        )
        assert len(partitions) == 3
        assert all(partitions)
        assert sorted(index for partition in partitions for index in partition) == [0, 1, 2, 3]
        assert all(sum(teacher_lengths[index] for index in partition) <= 10 for partition in partitions)
        assert all(sum(student_lengths[index] for index in partition) <= 8 for partition in partitions)
        forward_counts = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
        local_forward_count = torch.tensor([len(partitions)], dtype=torch.long)
        dist.all_gather(forward_counts, local_forward_count)
        assert [int(count.item()) for count in forward_counts] == [3, 3]

        local_errors = ["rank 0 teacher input is too long"] if rank == 0 else []
        with pytest.raises(ValueError, match="rank 0 teacher input is too long"):
            raise_if_any_rank_has_errors(
                local_errors,
                dp_group=dist.group.WORLD,
                collective_device="cpu",
                error_prefix="OPSD test validation",
            )
    finally:
        dist.destroy_process_group()


def test_opsd_partitions_balance_teacher_and_respect_both_token_budgets():
    student_lengths = [300, 350, 280, 260]
    teacher_lengths = [900, 200, 700, 300]

    partitions = build_balanced_micro_batch_indices(
        primary_lengths=teacher_lengths,
        secondary_lengths=student_lengths,
        num_micro_batches=3,
        primary_token_budget=1200,
        secondary_token_budget=800,
    )

    assert partitions is not None
    assert sorted(index for partition in partitions for index in partition) == [0, 1, 2, 3]
    assert all(sum(teacher_lengths[index] for index in partition) <= 1200 for partition in partitions)
    assert all(sum(student_lengths[index] for index in partition) <= 800 for partition in partitions)
    teacher_group_tokens = [sum(teacher_lengths[index] for index in partition) for partition in partitions]
    assert teacher_group_tokens == sorted(teacher_group_tokens, reverse=True)


def test_opsd_partitions_report_when_group_count_cannot_meet_secondary_budget():
    partitions = build_balanced_micro_batch_indices(
        primary_lengths=[5, 5, 5],
        secondary_lengths=[6, 6, 6],
        num_micro_batches=2,
        primary_token_budget=10,
        secondary_token_budget=8,
    )

    assert partitions is None


def test_opsd_partitions_fall_back_to_singletons():
    partitions = build_balanced_micro_batch_indices(
        primary_lengths=[9, 8, 7],
        secondary_lengths=[6, 5, 4],
        num_micro_batches=3,
        primary_token_budget=9,
        secondary_token_budget=6,
    )

    assert partitions is not None
    assert sorted(partitions) == [[0], [1], [2]]


def test_opsd_partitions_enforce_fixed_micro_batch_size():
    partitions = build_balanced_micro_batch_indices(
        primary_lengths=[4, 3, 2, 1],
        secondary_lengths=[1, 1, 1, 1],
        num_micro_batches=2,
        max_micro_batch_size=1,
    )

    assert partitions is None


def test_prepare_micro_batches_uses_precomputed_indices_for_all_batch_fields():
    data = TensorDict(
        {
            "student_id": torch.tensor([10, 11, 12, 13]),
            "teacher_id": torch.tensor([20, 21, 22, 23]),
        },
        batch_size=[4],
    )
    plan = [[2], [0, 3], [1]]
    tu.assign_non_tensor_data(data, "precomputed_micro_batch_indices", plan)

    micro_batches, indices = prepare_micro_batches(data)

    assert indices == plan
    assert [batch["student_id"].tolist() for batch in micro_batches] == [[12], [10, 13], [11]]
    assert [batch["teacher_id"].tolist() for batch in micro_batches] == [[22], [20, 23], [21]]


def test_prepare_micro_batches_rejects_invalid_precomputed_indices():
    data = TensorDict({"sample_id": torch.arange(3)}, batch_size=[3])
    tu.assign_non_tensor_data(data, "precomputed_micro_batch_indices", [[0, 1], [1, 2]])

    with pytest.raises(ValueError, match="cover every batch row exactly once"):
        prepare_micro_batches(data)


def test_prepare_micro_batches_keeps_existing_fixed_batch_path_without_a_plan():
    data = TensorDict(
        {
            "input_ids": torch.arange(8).reshape(4, 2),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=2)

    micro_batches, indices = prepare_micro_batches(data)

    assert indices is None
    assert len(micro_batches) == 2
    torch.testing.assert_close(torch.cat([batch["input_ids"] for batch in micro_batches]), data["input_ids"])


def test_masked_loss_stats_handle_an_empty_response_mask():
    losses = torch.tensor([[0.5, -0.25], [1.5, 2.0]])
    loss_mask = torch.zeros_like(losses, dtype=torch.bool)

    valid_sum, valid_count, valid_min, valid_max = compute_masked_loss_stats(losses, loss_mask)

    assert valid_sum.item() == 0
    assert valid_count.item() == 0
    assert torch.isposinf(valid_min)
    assert torch.isneginf(valid_max)


def test_opsd_student_and_teacher_build_distinct_aligned_prediction_indices():
    responses = [[31, 32], [41, 42]]
    response_mask = [[1, 0], [0, 1]]
    student_data = _make_opsd_no_padding_batch(
        sequences=[[11, 12, 31, 32], [21, 41, 42]],
        attention_mask=[[1, 1, 1, 1], [0, 1, 1, 1]],
        responses=responses,
        response_mask=response_mask,
    )
    teacher_data = _make_opsd_no_padding_batch(
        sequences=[[11, 12, 13, 14, 31, 32], [21, 22, 23, 41, 42]],
        attention_mask=[[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]],
        responses=responses,
        response_mask=response_mask,
    )

    student_indices = build_response_prediction_indices(student_data)
    teacher_indices = build_response_prediction_indices(teacher_data)

    assert student_indices.tolist() == [1, 5]
    assert teacher_indices.tolist() == [3, 9]
    assert student_indices.numel() == teacher_indices.numel() == 2
    torch.testing.assert_close(student_data["input_ids"].values()[student_indices + 1], torch.tensor([31, 42]))
    torch.testing.assert_close(teacher_data["input_ids"].values()[teacher_indices + 1], torch.tensor([31, 42]))


def test_fsdp_opsd_logits_to_keep_is_applied_and_scalar_losses_are_scattered():
    data = _make_opsd_no_padding_batch(
        sequences=[[11, 12, 31, 32], [21, 41, 42]],
        attention_mask=[[1, 1, 1, 1], [0, 1, 1, 1]],
        responses=[[31, 32], [41, 42]],
        response_mask=[[1, 0], [0, 1]],
    )
    data["temperature"] = torch.ones(2)
    tu.assign_non_tensor(
        data,
        use_remove_padding=True,
        use_fused_kernels=False,
        opsd_use_logits_processor=True,
    )
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    model_inputs, output_args = engine.prepare_model_inputs(data)
    selected_indices = model_inputs["logits_to_keep"]
    assert selected_indices.tolist() == [1, 5]

    selected_logits = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    output = SimpleNamespace(logits=selected_logits)

    def selected_loss_processor(student_logits, data):
        del data
        return {"opsd_losses": student_logits.sum(dim=-1)}

    model_output = engine.prepare_model_outputs(
        output=output,
        output_args=output_args,
        micro_batch=data,
        logits_processor_func=selected_loss_processor,
    )

    scattered_losses = model_output["opsd_losses"].values()
    expected_losses = torch.zeros(7)
    expected_losses[selected_indices] = selected_logits.squeeze(0).sum(dim=-1)
    torch.testing.assert_close(scattered_losses, expected_losses)


def test_opsd_teacher_forward_batch_reuses_student_response_tensors():
    responses = torch.tensor([[31, 32], [41, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)
    teacher_input_ids = torch.tensor(
        [[11, 12, 21, 22, 31, 32], [0, 13, 23, 24, 41, 0]],
        dtype=torch.long,
    )
    teacher_attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    teacher_position_ids = (teacher_attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    data = TensorDict(
        {
            "responses": responses,
            "response_mask": response_mask,
            "opsd_teacher_input_ids": teacher_input_ids,
            "opsd_teacher_attention_mask": teacher_attention_mask,
            "opsd_teacher_position_ids": teacher_position_ids,
        },
        batch_size=[2],
    )
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)

    teacher_data = worker._build_opsd_teacher_forward_batch(data)

    assert "prompts" not in teacher_data
    torch.testing.assert_close(teacher_data["responses"], responses)
    torch.testing.assert_close(teacher_data["response_mask"], response_mask)
    torch.testing.assert_close(teacher_data["input_ids"], teacher_input_ids)


def test_opsd_group_count_and_validation_errors_are_synchronized_across_dp_ranks(tmp_path):
    init_method = f"file://{tmp_path / 'opsd_gloo_init'}"
    mp.start_processes(
        _distributed_opsd_batching_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
        start_method="fork",
    )
