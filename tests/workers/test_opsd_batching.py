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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import (
    build_balanced_micro_batch_indices,
    build_synchronized_balanced_micro_batch_indices,
    compute_masked_loss_stats,
    prepare_micro_batches,
    raise_if_any_rank_has_errors,
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


def test_opsd_group_count_and_validation_errors_are_synchronized_across_dp_ranks(tmp_path):
    init_method = f"file://{tmp_path / 'opsd_gloo_init'}"
    mp.start_processes(
        _distributed_opsd_batching_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
        start_method="fork",
    )
