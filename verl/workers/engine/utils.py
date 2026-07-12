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

import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_name, is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    rearrange_micro_batches,
    restore_dynamic_batch,
)


def build_balanced_micro_batch_indices(
    primary_lengths: Sequence[int],
    secondary_lengths: Sequence[int],
    num_micro_batches: int,
    primary_token_budget: int | None = None,
    secondary_token_budget: int | None = None,
    max_micro_batch_size: int | None = None,
) -> list[list[int]] | None:
    """Balance by primary lengths while enforcing both token budgets."""
    primary_lengths = [int(length) for length in primary_lengths]
    secondary_lengths = [int(length) for length in secondary_lengths]
    batch_size = len(primary_lengths)

    if batch_size == 0:
        raise ValueError("Cannot build micro-batches for an empty batch.")
    if len(secondary_lengths) != batch_size:
        raise ValueError(
            "Primary and secondary length lists must have the same size, "
            f"got {batch_size} and {len(secondary_lengths)}."
        )
    if any(length <= 0 for length in primary_lengths + secondary_lengths):
        raise ValueError("All sequence lengths must be greater than zero.")
    if not 1 <= num_micro_batches <= batch_size:
        raise ValueError(f"num_micro_batches must be in [1, {batch_size}], got {num_micro_batches}.")
    for name, value in (
        ("primary_token_budget", primary_token_budget),
        ("secondary_token_budget", secondary_token_budget),
        ("max_micro_batch_size", max_micro_batch_size),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero, got {value}.")

    partitions = get_seqlen_balanced_partitions(
        seqlen_list=primary_lengths,
        k_partitions=num_micro_batches,
        equal_size=False,
    )
    partitions.sort(
        key=lambda partition: (
            sum(primary_lengths[index] for index in partition),
            sum(secondary_lengths[index] for index in partition),
            partition[0],
        ),
        reverse=True,
    )

    for partition in partitions:
        if max_micro_batch_size is not None and len(partition) > max_micro_batch_size:
            return None
        if primary_token_budget is not None and (
            sum(primary_lengths[index] for index in partition) > primary_token_budget
        ):
            return None
        if secondary_token_budget is not None and (
            sum(secondary_lengths[index] for index in partition) > secondary_token_budget
        ):
            return None
    return partitions


def compute_masked_loss_stats(losses: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return sum, count, min, and max without failing on an empty mask."""
    if losses.shape != loss_mask.shape:
        raise ValueError(
            f"losses and loss_mask must have the same shape, got {tuple(losses.shape)} and {tuple(loss_mask.shape)}."
        )
    loss_mask = loss_mask.bool()
    valid_sum = (losses * loss_mask).sum()
    valid_count = loss_mask.sum()
    if losses.numel() == 0:
        return valid_sum, valid_count, losses.new_tensor(float("inf")), losses.new_tensor(float("-inf"))
    valid_min = losses.masked_fill(~loss_mask, float("inf")).min()
    valid_max = losses.masked_fill(~loss_mask, float("-inf")).max()
    return valid_sum, valid_count, valid_min, valid_max


def reduce_distributed_int(value: int, op, dp_group=None, collective_device=None) -> int:
    if not torch.distributed.is_initialized():
        return int(value)
    collective_device = collective_device or get_device_name()
    reduced = torch.tensor(int(value), dtype=torch.long, device=collective_device)
    torch.distributed.all_reduce(reduced, op=op, group=dp_group)
    return int(reduced.item())


def raise_if_any_rank_has_errors(
    local_errors: Sequence[str],
    dp_group=None,
    collective_device=None,
    error_prefix: str = "Distributed validation failed",
) -> None:
    """Raise the same validation error on every rank when any rank reports an error."""
    local_errors = list(local_errors)
    any_error = reduce_distributed_int(
        bool(local_errors),
        op=torch.distributed.ReduceOp.MAX,
        dp_group=dp_group,
        collective_device=collective_device,
    )
    if not any_error:
        return

    if torch.distributed.is_initialized():
        gathered_errors = [None] * torch.distributed.get_world_size(group=dp_group)
        torch.distributed.all_gather_object(gathered_errors, local_errors, group=dp_group)
        rank_errors = {rank: errors for rank, errors in enumerate(gathered_errors) if errors}
    else:
        rank_errors = {0: local_errors}
    raise ValueError(f"{error_prefix}: {rank_errors}.")


def build_synchronized_balanced_micro_batch_indices(
    primary_lengths: Sequence[int],
    secondary_lengths: Sequence[int],
    local_num_micro_batches: int,
    primary_token_budget: int | None = None,
    secondary_token_budget: int | None = None,
    max_micro_batch_size: int | None = None,
    dp_group=None,
    collective_device=None,
) -> list[list[int]]:
    """Build a budget-valid local plan with one shared micro-batch count across DP ranks."""
    batch_size = len(primary_lengths)
    min_batch_size = reduce_distributed_int(
        batch_size,
        op=torch.distributed.ReduceOp.MIN,
        dp_group=dp_group,
        collective_device=collective_device,
    )
    max_batch_size = reduce_distributed_int(
        batch_size,
        op=torch.distributed.ReduceOp.MAX,
        dp_group=dp_group,
        collective_device=collective_device,
    )
    if min_batch_size != max_batch_size:
        raise ValueError(
            "Joint batching requires the same local sample count on every DP rank, "
            f"got min={min_batch_size}, max={max_batch_size}."
        )

    group_count = reduce_distributed_int(
        local_num_micro_batches,
        op=torch.distributed.ReduceOp.MAX,
        dp_group=dp_group,
        collective_device=collective_device,
    )
    while group_count <= batch_size:
        partitions = build_balanced_micro_batch_indices(
            primary_lengths=primary_lengths,
            secondary_lengths=secondary_lengths,
            num_micro_batches=group_count,
            primary_token_budget=primary_token_budget,
            secondary_token_budget=secondary_token_budget,
            max_micro_batch_size=max_micro_batch_size,
        )
        any_partition_failed = reduce_distributed_int(
            partitions is None,
            op=torch.distributed.ReduceOp.MAX,
            dp_group=dp_group,
            collective_device=collective_device,
        )
        if not any_partition_failed:
            assert partitions is not None
            return partitions
        group_count += 1

    raise RuntimeError("Unable to build a synchronized micro-batch plan even with singleton groups.")


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if is_npu_available:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if is_npu_available:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Prepare micro batches from data.
    """
    precomputed_indices = tu.get_non_tensor_data(data=data, key="precomputed_micro_batch_indices", default=None)
    if precomputed_indices is not None:
        normalized_indices = []
        for partition in precomputed_indices:
            partition = [int(index) for index in partition]
            if not partition:
                raise ValueError("Precomputed micro-batch partitions must be non-empty.")
            normalized_indices.append(partition)

        flattened_indices = [index for partition in normalized_indices for index in partition]
        if sorted(flattened_indices) != list(range(len(data))):
            raise ValueError(
                "Precomputed micro-batch indices must cover every batch row exactly once, "
                f"got indices={normalized_indices} for batch_size={len(data)}."
            )
        micro_batches = [tu.index_select_tensor_dict(data, partition) for partition in normalized_indices]
        return micro_batches, normalized_indices

    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)

    force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)

    if use_dynamic_bsz:
        assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
        max_token_len_per_gpu = data["max_token_len_per_gpu"]
        max_token_len = max_token_len_per_gpu * sp_size
        micro_batches, batch_idx_list = rearrange_micro_batches(
            data,
            max_token_len=max_token_len,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
            force_group_size=force_group_size,
        )
    else:
        total_data_size = len(data)
        micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
        assert total_data_size % (force_group_size * micro_batch_size_per_gpu) == 0, (
            "data size must be divisible by force_group_size * micro_batch_size_per_gpu"
        )
        micro_batches = tu.chunk_tensordict(data, total_data_size // (micro_batch_size_per_gpu * force_group_size))
        batch_idx_list = None
    return micro_batches, batch_idx_list


def postprocess_batch_func(output_lst, indices, data: TensorDict):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """

    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    assert pad_mode == DatasetPadMode.NO_PADDING, "postprocess_batch_func only support NO_PADDING pad_mode"

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        if pad_mode == DatasetPadMode.NO_PADDING:
            tensors = [tensor for nt in model_output[key] for tensor in nt.unbind()]
            model_output[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # Restore the original row order after dynamic or precomputed batching.
        if indices is not None:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output
