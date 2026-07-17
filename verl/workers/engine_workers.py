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
import functools
import json
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Callable, Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.opsd_metrics import build_opsd_topk_metric_objects, compute_opsd_topk_masses
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric import AggregationType
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import ceildiv
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine.utils import (
    build_response_prediction_indices,
    build_synchronized_balanced_micro_batch_indices,
    compute_masked_loss_stats,
    raise_if_any_rank_has_errors,
    reduce_distributed_int,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.rollout.utils import get_max_position_embeddings
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td, loss_function=loss_function)
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> TensorDict:
        loss_function = loss_function or self.loss_fn
        assert loss_function is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=loss_function)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self,
        config: DictConfig,
        role: str,
        distillation_config: Optional[DistillationConfig] = None,
        opsd_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role

        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None

        # OPSD keeps a forward-only teacher inside the actor worker group.
        self.opsd_config = opsd_config
        self.opsd_enabled = bool(opsd_config is not None and opsd_config.get("enabled", False))
        self.opsd_teacher: TrainingWorker = None
        if self.opsd_enabled:
            teacher_cfg = self.opsd_config.get("teacher", {})
            teacher_update_cfg = teacher_cfg.get("update", {})
            teacher_privileged_input_cfg = teacher_cfg.get("privileged_input", {})
            loss_cfg = self.opsd_config.get("loss", {})
            legacy_loss_keys = [key for key in ("loss_mode", "topk_strategy") if key in loss_cfg]
            if legacy_loss_keys:
                raise ValueError(
                    "Deprecated OPSD loss config keys are not accepted because they can silently select the wrong "
                    f"vocabulary path: {legacy_loss_keys}. Use only opsd.loss.vocab_strategy."
                )

            self.opsd_teacher_model_path = teacher_cfg.get("model_path")
            self.opsd_teacher_share_actor_worker = teacher_cfg.get("share_actor_worker", True)
            self.opsd_teacher_use_dynamic_bsz = teacher_cfg.get("use_dynamic_bsz", True)
            self.opsd_teacher_max_token_len_per_gpu = teacher_cfg.get("max_token_len_per_gpu", None)
            self.opsd_teacher_micro_batch_size_per_gpu = teacher_cfg.get("micro_batch_size_per_gpu", None)
            self.opsd_teacher_use_remove_padding = teacher_cfg.get(
                "use_remove_padding", self.config.model.get("use_remove_padding", True)
            )
            self.opsd_teacher_use_fused_kernels = teacher_cfg.get("use_fused_kernels", False)
            teacher_fsdp_cfg = teacher_cfg.get("fsdp_config", {})
            self.opsd_teacher_param_offload = teacher_fsdp_cfg.get("param_offload", None)
            self.opsd_teacher_optimizer_offload = teacher_fsdp_cfg.get("optimizer_offload", False)

            self.opsd_teacher_update_mode = teacher_update_cfg.get("mode", "none")
            self.opsd_teacher_update_interval = teacher_update_cfg.get("interval", 1)
            self.opsd_teacher_update_ema_decay = teacher_update_cfg.get("ema_decay", 0.999)
            self.opsd_teacher_privileged_input_mode = teacher_privileged_input_cfg.get("mode", "answer")

            self.opsd_kl_mode = loss_cfg.get("kl_mode", "reverse_kl")
            self.opsd_rl_coupling = loss_cfg.get("rl_coupling", "none")
            self.opsd_vocab_strategy = loss_cfg.get("vocab_strategy", "full")
            self.opsd_student_topk = loss_cfg.get("student_topk", 8)
            self.opsd_chunked_topk_chunk_size = loss_cfg.get("chunked_topk_chunk_size", 4096)
            self.opsd_teacher_topk = loss_cfg.get("teacher_topk", 8)
            self.opsd_use_tail = loss_cfg.get(
                "use_tail", self.opsd_vocab_strategy.startswith(("union_", "teacher_"))
            )
            self.opsd_loss_coef = loss_cfg.get("loss_coef", 1.0)
            self.opsd_temperature = loss_cfg.get("temperature", 1.0)

            if self.opsd_teacher_model_path is None:
                raise ValueError("OPSD requires +opsd.teacher.model_path.")
            if self.config.actor.strategy not in {"fsdp", "fsdp2"}:
                raise NotImplementedError(
                    "OPSD currently supports only the FSDP actor engine, "
                    f"got actor.strategy={self.config.actor.strategy!r}."
                )
            actor_fsdp_cfg = self.config.actor.get("fsdp_config", {})
            actor_sp_size = max(
                int(self.config.actor.get("ulysses_sequence_parallel_size", 1)),
                int(actor_fsdp_cfg.get("ulysses_sequence_parallel_size", 1)),
            )
            if actor_sp_size != 1:
                raise NotImplementedError(
                    "OPSD selected-logits forward currently requires ulysses_sequence_parallel_size=1."
                )
            if not self.opsd_teacher_share_actor_worker:
                raise NotImplementedError(
                    "OPSD currently only supports +opsd.teacher.share_actor_worker=True."
                )
            valid_teacher_update_modes = {"none", "copy", "ema"}
            if self.opsd_teacher_update_mode not in valid_teacher_update_modes:
                raise ValueError(
                    f"Invalid OPSD teacher update mode: {self.opsd_teacher_update_mode}. "
                    f"Expected one of {sorted(valid_teacher_update_modes)}."
                )
            if self.opsd_teacher_update_mode != "none":
                raise NotImplementedError(
                    f"OPSD teacher update mode '{self.opsd_teacher_update_mode}' is reserved but not implemented yet."
                )
            if self.opsd_teacher_update_interval <= 0:
                raise ValueError("OPSD teacher.update.interval must be greater than 0.")
            if not 0 <= self.opsd_teacher_update_ema_decay < 1:
                raise ValueError("OPSD teacher.update.ema_decay must be in [0, 1).")
            valid_teacher_privileged_input_modes = {"answer", "reason", "cot_examples"}
            if self.opsd_teacher_privileged_input_mode not in valid_teacher_privileged_input_modes:
                raise ValueError(
                    f"Invalid OPSD teacher privileged input mode: {self.opsd_teacher_privileged_input_mode}. "
                    f"Expected one of {sorted(valid_teacher_privileged_input_modes)}."
                )
            if self.opsd_teacher_privileged_input_mode == "cot_examples":
                raise NotImplementedError(
                    "OPSD teacher privileged input mode "
                    f"'{self.opsd_teacher_privileged_input_mode}' is reserved but not implemented yet."
                )
            valid_rl_couplings = {"none", "grpo"}
            if self.opsd_rl_coupling not in valid_rl_couplings:
                raise ValueError(
                    f"Invalid OPSD rl_coupling: {self.opsd_rl_coupling}. "
                    f"Expected one of {sorted(valid_rl_couplings)}."
                )
            if self.opsd_rl_coupling != "none":
                raise NotImplementedError(
                    f"OPSD rl_coupling '{self.opsd_rl_coupling}' is reserved but not implemented yet."
                )
            valid_vocab_strategies = {
                "full",
                "student_renorm",
                "student_truncated",
                "teacher_renorm",
                "teacher_truncated",
                "union_renorm",
                "union_truncated",
            }
            if self.opsd_vocab_strategy not in valid_vocab_strategies:
                raise ValueError(
                    f"Invalid OPSD vocab_strategy: {self.opsd_vocab_strategy}. "
                    f"Expected one of {sorted(valid_vocab_strategies)}."
                )
            if self.opsd_vocab_strategy not in {"full", "student_renorm", "student_truncated"}:
                raise NotImplementedError(
                    f"OPSD vocab strategy '{self.opsd_vocab_strategy}' is reserved but not implemented yet."
                )
            valid_kl_modes = {"forward_kl", "reverse_kl", "dynamic_merge"}
            if self.opsd_kl_mode not in valid_kl_modes:
                raise ValueError(
                    f"Invalid OPSD kl_mode: {self.opsd_kl_mode}. "
                    f"Expected one of {sorted(valid_kl_modes)}."
                )
            if self.opsd_kl_mode != "reverse_kl":
                raise NotImplementedError(f"OPSD kl_mode '{self.opsd_kl_mode}' is reserved but not implemented yet.")
            if self.opsd_student_topk <= 0:
                raise ValueError("OPSD loss.student_topk must be greater than 0.")
            if self.opsd_chunked_topk_chunk_size <= 0:
                raise ValueError("OPSD loss.chunked_topk_chunk_size must be greater than 0.")
            if self.opsd_teacher_topk <= 0:
                raise ValueError("OPSD loss.teacher_topk must be greater than 0.")
            if self.opsd_temperature <= 0:
                raise ValueError("OPSD loss.temperature must be greater than 0.")
            if self.opsd_use_tail:
                raise NotImplementedError(
                    "OPSD loss.use_tail=True is reserved for future teacher/union vocabulary strategies."
                )
            if self.opsd_teacher_use_dynamic_bsz and self.opsd_teacher_max_token_len_per_gpu is None:
                raise ValueError(
                    "OPSD teacher.max_token_len_per_gpu must be set when teacher.use_dynamic_bsz=True."
                )
            if self.opsd_teacher_use_dynamic_bsz and self.opsd_teacher_max_token_len_per_gpu <= 0:
                raise ValueError("OPSD teacher.max_token_len_per_gpu must be greater than 0.")
            if not self.opsd_teacher_use_dynamic_bsz and self.opsd_teacher_micro_batch_size_per_gpu is None:
                raise ValueError(
                    "OPSD teacher.micro_batch_size_per_gpu must be set when teacher.use_dynamic_bsz=False."
                )
            if (
                not self.opsd_teacher_use_dynamic_bsz
                and self.opsd_teacher_micro_batch_size_per_gpu is not None
                and self.opsd_teacher_micro_batch_size_per_gpu <= 0
            ):
                raise ValueError("OPSD teacher.micro_batch_size_per_gpu must be greater than 0.")
        else:
            self.opsd_teacher_model_path = None
            self.opsd_teacher_share_actor_worker = True
            self.opsd_teacher_use_dynamic_bsz = True
            self.opsd_teacher_max_token_len_per_gpu = None
            self.opsd_teacher_micro_batch_size_per_gpu = None
            self.opsd_teacher_use_remove_padding = True
            self.opsd_teacher_use_fused_kernels = False
            self.opsd_teacher_param_offload = None
            self.opsd_teacher_optimizer_offload = False

            self.opsd_teacher_update_mode = "none"
            self.opsd_teacher_update_interval = 1
            self.opsd_teacher_update_ema_decay = 0.999
            self.opsd_teacher_privileged_input_mode = "answer"

            self.opsd_kl_mode = "reverse_kl"
            self.opsd_rl_coupling = "none"
            self.opsd_vocab_strategy = "full"
            self.opsd_student_topk = 8
            self.opsd_chunked_topk_chunk_size = 4096
            self.opsd_teacher_topk = 8
            self.opsd_use_tail = False
            self.opsd_loss_coef = 1.0
            self.opsd_temperature = 1.0


        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self.opsd_enabled and not self._is_actor:
            raise ValueError("OPSD requires actor role because OPSD loss is computed during actor update.")
        if self.opsd_enabled and self.distillation_enabled:
            raise ValueError("OPSD-v0 should not be enabled together with original distillation.enabled.")


        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        # Router replay is supported on the megatron engine and on the veomni
        # engine. Both expose `router_replay` on their per-strategy engine
        # config (the field lives on the shared `EngineConfig` base).
        actor_strategy = self.config.actor.strategy
        if actor_strategy == "megatron":
            rr_mode = self.config.actor.megatron.router_replay.mode
        elif actor_strategy == "veomni":
            rr_mode = self.config.actor.veomni.router_replay.mode
        else:
            rr_mode = "disabled"
        self.enable_routing_replay = rr_mode != "disabled"

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())
        # 2.5 build OPSD teacher model config
        if self.opsd_enabled:
            teacher_model_omega = deepcopy(self.config.model)
            with open_dict(teacher_model_omega):
                teacher_model_omega.path = self.opsd_teacher_model_path
                # teacher 的 hf config 和 tokenizer 默认也从 teacher model path 加载
                teacher_model_omega.hf_config_path = self.opsd_teacher_model_path
                teacher_model_omega.tokenizer_path = self.opsd_teacher_model_path
                teacher_model_omega.enable_gradient_checkpointing = False
                teacher_model_omega.use_remove_padding = self.opsd_teacher_use_remove_padding
                teacher_model_omega.use_fused_kernels = self.opsd_teacher_use_fused_kernels

            # Reuse ref's forward-only engine behavior, but load weights from
            # opsd.teacher.model_path instead of the student model path.
            opsd_teacher_model_config: HFModelConfig = omega_conf_to_dataclass(teacher_model_omega)

            opsd_teacher_ref_omega = deepcopy(self.config.ref)
            with open_dict(opsd_teacher_ref_omega):
                opsd_teacher_ref_omega.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                opsd_teacher_ref_omega.ppo_micro_batch_size = opsd_teacher_ref_omega.pop(
                    "log_prob_micro_batch_size", None
                )
                ref_micro_batch_size_per_gpu = opsd_teacher_ref_omega.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                opsd_teacher_ref_omega.pop("log_prob_use_dynamic_bsz", False)
                ref_max_token_len_per_gpu = opsd_teacher_ref_omega.pop("log_prob_max_token_len_per_gpu", None)
                opsd_teacher_ref_omega.use_dynamic_bsz = self.opsd_teacher_use_dynamic_bsz
                opsd_teacher_ref_omega.ppo_micro_batch_size_per_gpu = (
                    self.opsd_teacher_micro_batch_size_per_gpu
                    if self.opsd_teacher_micro_batch_size_per_gpu is not None
                    else ref_micro_batch_size_per_gpu
                )
                opsd_teacher_ref_omega.ppo_max_token_len_per_gpu = (
                    self.opsd_teacher_max_token_len_per_gpu
                    if self.opsd_teacher_max_token_len_per_gpu is not None
                    else ref_max_token_len_per_gpu
                )
                opsd_teacher_ref_omega.fsdp_config.forward_only = True
                opsd_teacher_ref_omega.fsdp_config.offload_policy = False
                if self.opsd_teacher_param_offload is not None:
                    opsd_teacher_ref_omega.fsdp_config.param_offload = self.opsd_teacher_param_offload
                opsd_teacher_ref_omega.fsdp_config.optimizer_offload = self.opsd_teacher_optimizer_offload

            opsd_teacher_config: ActorConfig = omega_conf_to_dataclass(opsd_teacher_ref_omega)
            opsd_teacher_config.model_config = opsd_teacher_model_config
            opsd_teacher_config.model_config.mtp = MtpConfig(enable=False)

            opsd_teacher_training_config = TrainingWorkerConfig(
                model_type=opsd_teacher_config.model_config.get("model_type", "language_model"),
                model_config=opsd_teacher_config.model_config,
                engine_config=opsd_teacher_config.engine,
                optimizer_config=opsd_teacher_config.optim,
                checkpoint_config=opsd_teacher_config.checkpoint,
            )

            opsd_teacher_training_config.engine_config.use_dynamic_bsz = self.opsd_teacher_use_dynamic_bsz
            opsd_teacher_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.opsd_teacher_max_token_len_per_gpu
            )
            opsd_teacher_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.opsd_teacher_micro_batch_size_per_gpu
            )
            opsd_teacher_training_config.engine_config.use_remove_padding = self.opsd_teacher_use_remove_padding
            opsd_teacher_training_config.engine_config.use_fused_kernels = self.opsd_teacher_use_fused_kernels
            opsd_teacher_training_config.engine_config.forward_only = True
            opsd_teacher_training_config.engine_config.forward_only_cpu_offload = (
                True if self.opsd_teacher_param_offload is None else self.opsd_teacher_param_offload
            )
            if self.opsd_teacher_use_dynamic_bsz:
                assert opsd_teacher_training_config.engine_config.infer_max_token_len_per_gpu is not None
            else:
                assert opsd_teacher_training_config.engine_config.infer_micro_batch_size_per_gpu is not None

            self.opsd_teacher = TrainingWorker(config=opsd_teacher_training_config)
            self.opsd_teacher.reset()
            self._validate_opsd_tokenizer_compatibility()
            self.set_dispatch_collect(mesh_name="opsd_teacher", **self.opsd_teacher.get_dispatch_collect())


        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    def _build_opsd_teacher_forward_batch(self, data: TensorDict) -> TensorDict:
        required_keys = (
            "responses",
            "response_mask",
            "opsd_teacher_input_ids",
            "opsd_teacher_attention_mask",
            "opsd_teacher_position_ids",
        )
        missing_keys = [key for key in required_keys if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"OPSD teacher forward requires teacher input tensors: {missing_keys}.")

        teacher_data = TensorDict(
            {
                "responses": data["responses"],
                "input_ids": data["opsd_teacher_input_ids"],
                "attention_mask": data["opsd_teacher_attention_mask"],
                "position_ids": data["opsd_teacher_position_ids"],
                "response_mask": data["response_mask"],
            },
            batch_size=data.batch_size,
        )
        return teacher_data

    def _validate_opsd_tokenizer_compatibility(self):
        if self.actor is None or self.opsd_teacher is None:
            raise RuntimeError("OPSD tokenizer validation requires initialized actor and teacher workers.")

        student_tokenizer = self.actor.model_config.tokenizer
        teacher_tokenizer = self.opsd_teacher.model_config.tokenizer
        if student_tokenizer is None or teacher_tokenizer is None:
            raise ValueError("OPSD requires both actor and teacher tokenizers to be loaded.")

        student_vocab = student_tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()
        mismatches = []
        for token, student_id in student_vocab.items():
            teacher_id = teacher_vocab.get(token)
            if teacher_id != student_id:
                mismatches.append((token, student_id, teacher_id))
                if len(mismatches) == 3:
                    break
        if len(student_vocab) != len(teacher_vocab) or mismatches:
            raise ValueError(
                "OPSD actor and teacher must use the same token-to-id vocabulary mapping. "
                f"Got actor_vocab_size={len(student_vocab)}, teacher_vocab_size={len(teacher_vocab)}, "
                f"first_mismatches={mismatches}."
            )

        special_id_names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        special_id_mismatches = {
            name: (getattr(student_tokenizer, name, None), getattr(teacher_tokenizer, name, None))
            for name in special_id_names
            if getattr(student_tokenizer, name, None) != getattr(teacher_tokenizer, name, None)
        }
        if special_id_mismatches:
            raise ValueError(
                "OPSD actor and teacher special token IDs must match. "
                f"Got mismatches={special_id_mismatches}."
            )

    def _build_opsd_joint_micro_batch_plan(self, data: TensorDict) -> list[list[int]]:
        input_ids = data["input_ids"]
        if not input_ids.is_nested:
            raise NotImplementedError("OPSD joint batching requires no-padding nested student input_ids.")

        teacher_attention_mask = data.get("opsd_teacher_attention_mask")
        if teacher_attention_mask is None or teacher_attention_mask.is_nested:
            raise ValueError("OPSD joint batching requires padded opsd_teacher_attention_mask.")

        student_lengths = input_ids.offsets().diff().to(torch.long).cpu().tolist()
        teacher_lengths = teacher_attention_mask.sum(dim=-1).to(torch.long).cpu().tolist()
        batch_size = len(student_lengths)

        dp_group = self.actor.engine.get_data_parallel_group()
        collective_device = get_device_name()

        validation_errors = []
        if batch_size == 0 or len(teacher_lengths) != batch_size:
            validation_errors.append(
                "OPSD student and teacher batches must be non-empty and row-aligned, "
                f"got student={batch_size}, teacher={len(teacher_lengths)}."
            )
        nonpositive_students = [(index, length) for index, length in enumerate(student_lengths) if length <= 0]
        nonpositive_teachers = [(index, length) for index, length in enumerate(teacher_lengths) if length <= 0]
        if nonpositive_students or nonpositive_teachers:
            validation_errors.append(
                "OPSD student and teacher sequence lengths must be positive. "
                f"student={nonpositive_students[:8]}, teacher={nonpositive_teachers[:8]}."
            )

        response_mask = data.get("response_mask")
        if response_mask is None:
            local_response_tokens = 0
            validation_errors.append("OPSD joint batching requires response_mask.")
        else:
            local_response_tokens = int(response_mask.sum().item())
            empty_response_samples = (response_mask.sum(dim=-1) == 0).nonzero(as_tuple=False).flatten().tolist()
            if empty_response_samples:
                validation_errors.append(
                    "Every OPSD sample must select at least one response token, "
                    f"empty_samples={empty_response_samples[:8]}."
                )
        global_response_tokens = reduce_distributed_int(
            local_response_tokens,
            op=torch.distributed.ReduceOp.SUM,
            dp_group=dp_group,
            collective_device=collective_device,
        )
        if global_response_tokens == 0:
            validation_errors.append("OPSD rollout batch must contain at least one response token selected for loss.")

        teacher_max_context = get_max_position_embeddings(self.opsd_teacher.model_config.hf_config)
        context_violations = [
            (index, length) for index, length in enumerate(teacher_lengths) if length > teacher_max_context
        ]
        if context_violations:
            validation_errors.append(
                "OPSD teacher input exceeds the model context window. "
                f"max_position_embeddings={teacher_max_context}, offending_samples={context_violations[:8]}."
            )

        actor_engine_config = self.actor.engine_config
        student_token_budget = None
        student_micro_batch_size = None
        if actor_engine_config.use_dynamic_bsz:
            student_token_budget = actor_engine_config.max_token_len_per_gpu
            if student_token_budget is None:
                raise ValueError("OPSD joint batching requires the actor token budget when dynamic batching is on.")
            student_token_budget = int(student_token_budget)
        else:
            student_micro_batch_size = actor_engine_config.micro_batch_size_per_gpu
            if student_micro_batch_size is None:
                raise ValueError(
                    "OPSD joint batching requires the actor micro-batch size when dynamic batching is off."
                )
            student_micro_batch_size = int(student_micro_batch_size)

        teacher_token_budget = None
        teacher_micro_batch_size = None
        if self.opsd_teacher_use_dynamic_bsz:
            teacher_token_budget = int(self.opsd_teacher_max_token_len_per_gpu)
        else:
            teacher_micro_batch_size = int(self.opsd_teacher_micro_batch_size_per_gpu)

        if student_token_budget is not None:
            oversized_students = [
                (index, length) for index, length in enumerate(student_lengths) if length > student_token_budget
            ]
            if oversized_students:
                validation_errors.append(
                    "An OPSD student sequence exceeds actor.ppo_max_token_len_per_gpu. "
                    f"budget={student_token_budget}, offending_samples={oversized_students[:8]}."
                )
        if teacher_token_budget is not None:
            oversized_teachers = [
                (index, length) for index, length in enumerate(teacher_lengths) if length > teacher_token_budget
            ]
            if oversized_teachers:
                validation_errors.append(
                    "An OPSD teacher sequence exceeds teacher.max_token_len_per_gpu and cannot be fixed by "
                    f"batch regrouping. budget={teacher_token_budget}, offending_samples={oversized_teachers[:8]}."
                )

        raise_if_any_rank_has_errors(
            validation_errors,
            dp_group=dp_group,
            collective_device=collective_device,
            error_prefix="Invalid OPSD joint micro-batch inputs across actor DP ranks",
        )

        required_group_counts = [1]
        if student_token_budget is not None:
            required_group_counts.append(ceildiv(sum(student_lengths), student_token_budget))
        else:
            required_group_counts.append(ceildiv(batch_size, student_micro_batch_size))
        if teacher_token_budget is not None:
            required_group_counts.append(ceildiv(sum(teacher_lengths), teacher_token_budget))
        else:
            required_group_counts.append(ceildiv(batch_size, teacher_micro_batch_size))
        local_group_count = max(required_group_counts)

        fixed_batch_limits = [
            limit for limit in (student_micro_batch_size, teacher_micro_batch_size) if limit is not None
        ]
        max_micro_batch_size = min(fixed_batch_limits) if fixed_batch_limits else None
        return build_synchronized_balanced_micro_batch_indices(
            primary_lengths=teacher_lengths,
            secondary_lengths=student_lengths,
            local_num_micro_batches=local_group_count,
            primary_token_budget=teacher_token_budget,
            secondary_token_budget=student_token_budget,
            max_micro_batch_size=max_micro_batch_size,
            dp_group=dp_group,
            collective_device=collective_device,
        )

    def _opsd_response_prediction_ranges(self, data: TensorDict) -> list[tuple[int, int]]:
        input_ids = data["input_ids"]
        if not input_ids.is_nested:
            raise NotImplementedError("OPSD full-vocab forward currently expects no-padding nested input_ids.")

        seq_lens = input_ids.offsets().diff().to(torch.long)
        attention_mask = data["attention_mask"]
        responses = data["responses"]
        if attention_mask.is_nested or responses.is_nested:
            raise NotImplementedError("OPSD response alignment currently expects padded attention/response tensors.")
        response_lens = attention_mask[:, -responses.shape[1] :].sum(dim=-1).to(torch.long)
        attention_seq_lens = attention_mask.sum(dim=-1).to(torch.long)
        if not torch.equal(seq_lens.cpu(), attention_seq_lens.cpu()):
            raise ValueError(
                "OPSD no-padding sequence lengths must match attention_mask lengths. "
                f"Got nested={seq_lens.tolist()}, attention={attention_seq_lens.tolist()}."
            )

        seq_offsets = input_ids.offsets()[1:].to(torch.long)
        ranges = []
        for seq_len, resp_len, seq_offset in zip(seq_lens, response_lens, seq_offsets, strict=True):
            seq_len_item = int(seq_len.item())
            resp_len_item = int(resp_len.item())
            seq_offset_item = int(seq_offset.item())
            if resp_len_item == 0:
                ranges.append((seq_offset_item, seq_offset_item))
                continue
            if seq_len_item <= resp_len_item:
                raise ValueError("OPSD response prediction needs at least one prompt token before the response.")
            # 对 response token 的预测来自：最后一个 prompt 位置，到倒数第二个 response 位置。
            ranges.append((seq_offset_item - resp_len_item - 1, seq_offset_item - 1))
        return ranges

    def _compute_opsd_teacher_selected_logits(
        self, data: TensorDict, student_logits: torch.Tensor, data_format: str = "thd"
    ) -> torch.Tensor:
        assert self.opsd_teacher is not None, "OPSD teacher is not initialized."
        if data_format != "thd":
            raise NotImplementedError(
                f"OPSD teacher selected-logits forward only supports data_format='thd', got {data_format!r}."
            )

        teacher_engine = self.opsd_teacher.engine
        if getattr(teacher_engine, "use_ulysses_sp", False):
            raise NotImplementedError("OPSD teacher selected-logits forward does not support Ulysses SP yet.")
        if not self.opsd_teacher_use_remove_padding:
            raise NotImplementedError(
                "OPSD teacher selected-logits forward currently requires teacher.use_remove_padding=True."
            )
        if self.opsd_teacher_use_fused_kernels:
            raise NotImplementedError(
                "OPSD teacher selected-logits forward requires a materialized LM head, not fused kernels."
            )
        test_enabled = bool(tu.get_non_tensor_data(data=data, key="opsd_test_enabled", default=False))

        has_leading_dim = student_logits.dim() == 3 and student_logits.shape[0] == 1
        student_logits_values = student_logits.squeeze(0) if has_leading_dim else student_logits
        student_prediction_indices = build_response_prediction_indices(data)
        if student_logits_values.shape[0] != student_prediction_indices.numel():
            raise ValueError(
                "OPSD student logits must contain exactly the selected response prediction positions, "
                f"got logits={student_logits_values.shape[0]}, selected={student_prediction_indices.numel()}."
            )
        response_width = data["responses"].shape[1]
        selected_response_mask = data["response_mask"].bool() & data["attention_mask"][:, -response_width:].bool()
        selected_rollout_ids = data["responses"][selected_response_mask]
        student_target_ids = data["input_ids"].values().index_select(0, student_prediction_indices + 1)
        student_targets_match_rollout = torch.equal(student_target_ids, selected_rollout_ids)
        if not student_targets_match_rollout:
            raise ValueError("OPSD student prediction indices do not point to the selected rollout tokens.")
        student_ranges = self._opsd_response_prediction_ranges(data)

        teacher_data = self._build_opsd_teacher_forward_batch(data)
        teacher_response_mask_matches_student = True
        if test_enabled:
            teacher_response_mask_matches_student = torch.equal(
                teacher_data["response_mask"], data["response_mask"]
            )
        if test_enabled and not teacher_response_mask_matches_student:
            raise ValueError("OPSD teacher forward must reuse the unchanged student response_mask.")
        teacher_data = left_right_2_no_padding(teacher_data)
        tu.assign_non_tensor(
            teacher_data,
            temperature=tu.get_non_tensor_data(data=data, key="temperature", default=self.opsd_temperature),
            use_remove_padding=True,
            use_dynamic_bsz=self.opsd_teacher_use_dynamic_bsz,
            max_token_len_per_gpu=self.opsd_teacher_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.opsd_teacher_micro_batch_size_per_gpu,
            use_fused_kernels=False,
            opsd_use_logits_processor=True,
        )

        teacher_group_lengths = teacher_data["input_ids"].offsets().diff().to(torch.long)
        teacher_group_tokens = int(teacher_group_lengths.sum().item())
        if self.opsd_teacher_use_dynamic_bsz:
            if teacher_group_tokens > self.opsd_teacher_max_token_len_per_gpu:
                raise RuntimeError(
                    "OPSD joint batching produced a teacher group over its token budget: "
                    f"tokens={teacher_group_tokens}, budget={self.opsd_teacher_max_token_len_per_gpu}."
                )
        elif len(teacher_data) > self.opsd_teacher_micro_batch_size_per_gpu:
            raise RuntimeError(
                "OPSD joint batching produced a teacher group over its fixed micro-batch size: "
                f"size={len(teacher_data)}, limit={self.opsd_teacher_micro_batch_size_per_gpu}."
            )
        tu.assign_non_tensor(
            data,
            opsd_teacher_micro_batch_count=1,
            opsd_teacher_group_tokens=teacher_group_tokens,
            opsd_teacher_group_max_sequence_len=int(teacher_group_lengths.max().item()),
        )

        teacher_data = teacher_data.to(get_device_id())
        with teacher_engine.eval_mode(), torch.no_grad():
            teacher_ranges = self._opsd_response_prediction_ranges(teacher_data)
            teacher_prediction_indices = build_response_prediction_indices(teacher_data)
            if teacher_prediction_indices.numel() != student_prediction_indices.numel():
                raise ValueError(
                    "OPSD teacher/student must select the same number of response tokens, "
                    f"got student={student_prediction_indices.numel()}, "
                    f"teacher={teacher_prediction_indices.numel()}."
                )
            teacher_target_ids = teacher_data["input_ids"].values().index_select(0, teacher_prediction_indices + 1)
            teacher_targets_match_rollout = torch.equal(teacher_target_ids, selected_rollout_ids)
            if not teacher_targets_match_rollout:
                raise ValueError("OPSD teacher prediction indices do not point to the selected rollout tokens.")
            for student_range, teacher_range in zip(student_ranges, teacher_ranges, strict=True):
                student_start, student_end = student_range
                teacher_start, teacher_end = teacher_range
                if student_end - student_start != teacher_end - teacher_start:
                    raise ValueError(
                        "OPSD teacher/student response lengths must match after privileged input construction. "
                        f"Got student={student_end - student_start}, teacher={teacher_end - teacher_start}."
                    )

            model_inputs, output_args = teacher_engine.prepare_model_inputs(micro_batch=teacher_data)
            autocast_dtype = getattr(teacher_engine, "_autocast_dtype", torch.bfloat16)
            autocast_ctx = (
                nullcontext()
                if autocast_dtype == torch.float32
                else torch.autocast(device_type=get_device_name(), dtype=autocast_dtype)
            )
            with autocast_ctx:
                raw_output = teacher_engine.module(**model_inputs, use_cache=False)

            teacher_logits_values = raw_output.logits.squeeze(0)
            if teacher_logits_values.shape[0] != teacher_prediction_indices.numel():
                raise ValueError(
                    "OPSD teacher output must contain exactly its selected prediction positions, "
                    f"got logits={teacher_logits_values.shape[0]}, selected={teacher_prediction_indices.numel()}."
                )
            selected_temperature = output_args["temperature_rmpad"].index_select(0, teacher_prediction_indices)
            teacher_logits_values = teacher_logits_values / selected_temperature.clamp(min=1e-8).unsqueeze(-1).to(
                teacher_logits_values.dtype
            )
            if teacher_logits_values.shape != student_logits_values.shape:
                raise ValueError(
                    "OPSD teacher/student selected logits must have the same shape, "
                    f"got student={tuple(student_logits_values.shape)}, "
                    f"teacher={tuple(teacher_logits_values.shape)}."
                )

            if test_enabled:
                self._opsd_test_alignment_evidence = {
                    "student_selected_count": student_prediction_indices.numel(),
                    "teacher_selected_count": teacher_prediction_indices.numel(),
                    "student_targets_match_rollout": student_targets_match_rollout,
                    "teacher_targets_match_rollout": teacher_targets_match_rollout,
                    "teacher_response_mask_matches_student": teacher_response_mask_matches_student,
                    "student_logits_finite": bool(torch.isfinite(student_logits_values).all()),
                    "teacher_logits_finite": bool(torch.isfinite(teacher_logits_values).all()),
                    "student_prediction_indices": student_prediction_indices.detach().cpu().tolist(),
                    "teacher_prediction_indices": teacher_prediction_indices.detach().cpu().tolist(),
                    "student_target_ids": student_target_ids.detach().cpu().tolist(),
                    "teacher_target_ids": teacher_target_ids.detach().cpu().tolist(),
                    "rollout_target_ids": selected_rollout_ids.detach().cpu().tolist(),
                }

        return teacher_logits_values.unsqueeze(0) if has_leading_dim else teacher_logits_values

    def _compute_opsd_student_selected_logits(
        self, student_logits: torch.Tensor, data: TensorDict, data_format: str = "thd"
    ) -> torch.Tensor:
        if student_logits is None:
            raise ValueError("OPSD student selected logits are required.")
        selected_count = build_response_prediction_indices(data).numel()
        logits_count = student_logits.shape[-2]
        if logits_count != selected_count:
            raise ValueError(
                "OPSD student logits must contain only selected response prediction positions, "
                f"got logits={logits_count}, selected={selected_count}."
            )
        return student_logits

    def _select_opsd_vocab_for_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_opsd_logits(student_logits=student_logits, teacher_logits=teacher_logits)
        strategy = tu.get_non_tensor_data(data=data, key="opsd_vocab_strategy", default=self.opsd_vocab_strategy)

        if strategy == "full":
            student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
            teacher_log_probs = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
            return student_log_probs.exp(), student_log_probs, teacher_log_probs

        if strategy in {"student_renorm", "student_truncated"}:
            topk = int(tu.get_non_tensor_data(data=data, key="opsd_student_topk", default=self.opsd_student_topk))
            vocab_size = student_logits.shape[-1]
            if topk <= 0:
                raise ValueError(f"OPSD student_topk must be greater than 0, got {topk}.")
            if topk > vocab_size:
                raise ValueError(f"OPSD student_topk ({topk}) cannot exceed vocab size ({vocab_size}).")
            use_tail = bool(tu.get_non_tensor_data(data=data, key="opsd_use_tail", default=self.opsd_use_tail))
            if use_tail:
                raise NotImplementedError(f"OPSD top-k strategy '{strategy}' currently requires use_tail=False.")

            if strategy == "student_renorm":
                student_topk_logits, student_topk_ids = torch.topk(
                    student_logits, k=topk, dim=-1, sorted=True
                )
                if tu.get_non_tensor_data(data=data, key="opsd_test_enabled", default=False):
                    self._opsd_test_loss_token_ids = student_topk_ids.detach()
                teacher_topk_logits = torch.gather(teacher_logits.detach(), dim=-1, index=student_topk_ids)
                student_log_probs = torch.log_softmax(student_topk_logits.float(), dim=-1)
                teacher_log_probs = torch.log_softmax(teacher_topk_logits.float(), dim=-1)
                return student_log_probs.exp(), student_log_probs, teacher_log_probs

            # softmax preserves logit ordering, so selecting IDs from logits is
            # equivalent to selecting them from the full-vocabulary probabilities.
            student_topk_ids = torch.topk(student_logits.detach(), k=topk, dim=-1, sorted=True).indices
            if tu.get_non_tensor_data(data=data, key="opsd_test_enabled", default=False):
                self._opsd_test_loss_token_ids = student_topk_ids.detach()
            chunk_size = int(
                tu.get_non_tensor_data(
                    data=data,
                    key="opsd_chunked_topk_chunk_size",
                    default=self.opsd_chunked_topk_chunk_size,
                )
            )
            student_topk_log_probs = _chunked_topk_log_probs(
                logits=student_logits,
                topk_ids=student_topk_ids,
                chunk_size=chunk_size,
            ).float()
            teacher_topk_log_probs = _chunked_topk_log_probs(
                logits=teacher_logits.detach(),
                topk_ids=student_topk_ids,
                chunk_size=chunk_size,
            ).float()
            student_topk_probs = student_topk_log_probs.exp()
            return student_topk_probs, student_topk_log_probs, teacher_topk_log_probs

        if strategy in {"teacher_renorm", "teacher_truncated", "union_renorm", "union_truncated"}:
            raise NotImplementedError(f"OPSD vocab strategy '{strategy}' is reserved but not implemented yet.")
        raise ValueError(f"Invalid OPSD vocab strategy: {strategy}.")

    def _validate_opsd_logits(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"OPSD student/teacher logits must have the same shape, "
                f"got student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}."
            )
        if student_logits.dim() < 2:
            raise ValueError(f"OPSD logits must include a vocab dimension, got shape={tuple(student_logits.shape)}.")
        if student_logits.shape[-1] <= 0:
            raise ValueError("OPSD logits must have a non-empty vocab dimension.")

    def _validate_opsd_distributions(
        self,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ):
        if student_probs.shape != student_log_probs.shape or student_probs.shape != teacher_log_probs.shape:
            raise ValueError(
                "OPSD student/teacher distributions must have the same shape, "
                f"got student_probs={tuple(student_probs.shape)}, "
                f"student_log_probs={tuple(student_log_probs.shape)}, "
                f"teacher_log_probs={tuple(teacher_log_probs.shape)}."
            )
        if student_probs.dim() < 2:
            raise ValueError(
                f"OPSD distributions must include a vocab dimension, got shape={tuple(student_probs.shape)}."
            )
        if student_probs.shape[-1] <= 0:
            raise ValueError("OPSD distributions must have a non-empty vocab dimension.")

    def _compute_opsd_reverse_kl(
        self,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    def _compute_opsd_forward_kl(
        self,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("OPSD forward_kl is reserved but not implemented yet.")

    def _compute_opsd_dynamic_merge_kl(
        self,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        data: TensorDict,
    ) -> torch.Tensor:
        raise NotImplementedError("OPSD dynamic_merge KL is reserved but not implemented yet.")

    def _compute_opsd_kl_by_mode(
        self,
        kl_mode: str,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        data: TensorDict,
    ) -> torch.Tensor:
        if kl_mode == "reverse_kl":
            return self._compute_opsd_reverse_kl(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
            )
        if kl_mode == "forward_kl":
            return self._compute_opsd_forward_kl(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
            )
        if kl_mode == "dynamic_merge":
            return self._compute_opsd_dynamic_merge_kl(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                data=data,
            )
        raise ValueError(f"Invalid OPSD kl_mode: {kl_mode}.")

    def _compute_opsd_token_loss(
        self,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        data: TensorDict,
    ) -> torch.Tensor:
        self._validate_opsd_distributions(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
        )
        kl_mode = tu.get_non_tensor_data(data=data, key="opsd_kl_mode", default=self.opsd_kl_mode)
        return self._compute_opsd_kl_by_mode(
            kl_mode=kl_mode,
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            data=data,
        )

    def _opsd_test_token_text(self, token_id: int) -> str:
        tokenizer = self.actor.model_config.tokenizer
        try:
            return str(tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def _build_opsd_test_records(
        self,
        *,
        data: TensorDict,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        token_losses: torch.Tensor,
    ) -> list[dict]:
        """Build bounded, JSON-safe evidence from the tensors used by the real OPSD loss."""
        if not tu.get_non_tensor_data(data=data, key="opsd_test_enabled", default=False):
            return []

        student_logits = student_logits.squeeze(0) if student_logits.dim() == 3 else student_logits
        teacher_logits = teacher_logits.squeeze(0) if teacher_logits.dim() == 3 else teacher_logits
        student_probs = student_probs.squeeze(0) if student_probs.dim() == 3 else student_probs
        student_log_probs = student_log_probs.squeeze(0) if student_log_probs.dim() == 3 else student_log_probs
        teacher_log_probs = teacher_log_probs.squeeze(0) if teacher_log_probs.dim() == 3 else teacher_log_probs
        token_losses = token_losses.squeeze(0) if token_losses.dim() == 2 else token_losses
        loss_token_ids = getattr(self, "_opsd_test_loss_token_ids", None)
        if loss_token_ids is not None and loss_token_ids.dim() == 3:
            loss_token_ids = loss_token_ids.squeeze(0)

        response_ranges = self._opsd_response_prediction_ranges(data)
        selected_prediction_indices = build_response_prediction_indices(data)
        alignment_evidence = getattr(self, "_opsd_test_alignment_evidence", None)
        if alignment_evidence is None:
            raise RuntimeError("OPSD test records require teacher/student alignment evidence from the real forward.")
        if alignment_evidence["student_prediction_indices"] != selected_prediction_indices.detach().cpu().tolist():
            raise ValueError("OPSD test alignment evidence does not match the selected student prediction indices.")
        selected_row_by_prediction_index = {
            int(prediction_index): selected_row
            for selected_row, prediction_index in enumerate(selected_prediction_indices.tolist())
        }
        if student_logits.shape[0] != selected_prediction_indices.numel():
            raise ValueError(
                "OPSD diagnostics require one logits row per selected response token, "
                f"got logits={student_logits.shape[0]}, selected={selected_prediction_indices.numel()}."
            )
        responses = data["responses"]
        response_mask = data["response_mask"].bool()
        strategy = tu.get_non_tensor_data(data=data, key="opsd_vocab_strategy", default=self.opsd_vocab_strategy)
        diagnostic_topk = min(
            int(tu.get_non_tensor_data(data=data, key="opsd_test_topk", default=5)), student_logits.shape[-1]
        )
        max_samples = int(tu.get_non_tensor_data(data=data, key="opsd_test_max_samples", default=2))
        max_response_tokens = int(
            tu.get_non_tensor_data(data=data, key="opsd_test_max_response_tokens", default=32)
        )
        max_loss_vocab_tokens = int(
            tu.get_non_tensor_data(data=data, key="opsd_test_max_loss_vocab_tokens", default=32)
        )
        loss_coef = float(tu.get_non_tensor_data(data=data, key="opsd_loss_coef", default=self.opsd_loss_coef))
        global_step = int(tu.get_non_tensor_data(data=data, key="opsd_test_global_step", default=-1))

        uid_values = data.get("uid")
        records = []
        for sample_index, (response_start, response_end) in enumerate(response_ranges[:max_samples]):
            uid = None
            if uid_values is not None:
                uid = tu.unwrap_non_tensor_data(uid_values[sample_index])
            sample_record = {
                "global_step": global_step,
                "worker_rank": int(self.rank),
                "sample_index_on_worker": sample_index,
                "sample_uid": None if uid is None else str(uid),
                "vocab_strategy": str(strategy),
                "kl_mode": str(
                    tu.get_non_tensor_data(data=data, key="opsd_kl_mode", default=self.opsd_kl_mode)
                ),
                "loss_coef": loss_coef,
                "response_tokens_selected": int(response_mask[sample_index].sum().item()),
                "alignment_evidence": {
                    key: alignment_evidence[key]
                    for key in (
                        "student_selected_count",
                        "teacher_selected_count",
                        "student_targets_match_rollout",
                        "teacher_targets_match_rollout",
                        "teacher_response_mask_matches_student",
                        "student_logits_finite",
                        "teacher_logits_finite",
                    )
                },
                "tokens": [],
            }

            recorded_tokens = 0
            for response_index, prediction_index in enumerate(range(response_start, response_end)):
                selected_row = selected_row_by_prediction_index.get(prediction_index)
                if selected_row is None:
                    continue
                if recorded_tokens >= max_response_tokens:
                    break

                student_row = student_logits[selected_row].float()
                teacher_row = teacher_logits[selected_row].float()
                if strategy == "full" or loss_token_ids is None:
                    student_topk_logits, student_topk_ids = torch.topk(
                        student_row, k=diagnostic_topk, sorted=True
                    )
                else:
                    student_topk_ids = loss_token_ids[selected_row, :diagnostic_topk]
                    student_topk_logits = torch.gather(student_row, dim=-1, index=student_topk_ids)
                teacher_topk_logits, teacher_topk_ids = torch.topk(teacher_row, k=diagnostic_topk, sorted=True)
                student_topk_log_probs = student_topk_logits - torch.logsumexp(student_row, dim=-1)
                teacher_topk_log_probs_full = teacher_topk_logits - torch.logsumexp(teacher_row, dim=-1)

                def topk_entries(ids: torch.Tensor, log_probs: torch.Tensor) -> list[dict]:
                    return [
                        {
                            "token_id": int(token_id),
                            "token": self._opsd_test_token_text(int(token_id)),
                            "log_prob": float(log_prob),
                            "prob": float(log_prob.exp()),
                        }
                        for token_id, log_prob in zip(
                            ids.detach().cpu().tolist(), log_probs.detach().cpu(), strict=True
                        )
                    ]

                raw_token_loss = float(token_losses[selected_row].detach().float().cpu())
                loss_calculation = {
                    "raw_token_loss": raw_token_loss,
                    "weighted_token_loss": raw_token_loss * loss_coef,
                    "vocabulary_mode": "full" if strategy == "full" else "student_topk",
                    "vocabulary_size": int(student_logits.shape[-1]),
                    "terms": [],
                }

                if strategy == "full":
                    loss_ids = student_topk_ids[:max_loss_vocab_tokens]
                    full_student_log_probs = student_topk_log_probs[:max_loss_vocab_tokens]
                    full_teacher_log_probs = (
                        torch.gather(teacher_row, dim=-1, index=loss_ids) - torch.logsumexp(teacher_row, dim=-1)
                    )
                    full_student_probs = full_student_log_probs.exp()
                    loss_calculation["reported_terms_are_diagnostic_subset"] = True
                else:
                    loss_width = student_log_probs.shape[-1]
                    if loss_token_ids is None:
                        raise RuntimeError("OPSD test records require the exact token IDs selected by the loss path.")
                    reported_width = min(loss_width, max_loss_vocab_tokens)
                    loss_ids = loss_token_ids[selected_row, :reported_width]
                    full_student_probs = student_probs[selected_row, :reported_width]
                    full_student_log_probs = student_log_probs[selected_row, :reported_width]
                    full_teacher_log_probs = teacher_log_probs[selected_row, :reported_width]
                    loss_calculation["selected_vocabulary_size"] = int(loss_width)
                    loss_calculation["reported_terms_are_diagnostic_subset"] = reported_width < loss_width

                contributions = full_student_probs * (full_student_log_probs - full_teacher_log_probs)
                for token_id, prob, student_lp, teacher_lp, contribution in zip(
                    loss_ids.detach().cpu().tolist(),
                    full_student_probs.detach().float().cpu().tolist(),
                    full_student_log_probs.detach().float().cpu().tolist(),
                    full_teacher_log_probs.detach().float().cpu().tolist(),
                    contributions.detach().float().cpu().tolist(),
                    strict=True,
                ):
                    loss_calculation["terms"].append(
                        {
                            "token_id": int(token_id),
                            "token": self._opsd_test_token_text(int(token_id)),
                            "student_prob": prob,
                            "student_log_prob": student_lp,
                            "teacher_log_prob": teacher_lp,
                            "reverse_kl_contribution": contribution,
                        }
                    )
                reported_sum = float(contributions.sum().detach().float().cpu())
                loss_calculation["reported_contribution_sum"] = reported_sum
                loss_calculation["unreported_contribution"] = raw_token_loss - reported_sum

                rollout_token_id = int(responses[sample_index, response_index].item())
                student_target_token_id = int(alignment_evidence["student_target_ids"][selected_row])
                teacher_target_token_id = int(alignment_evidence["teacher_target_ids"][selected_row])
                aligned_rollout_token_id = int(alignment_evidence["rollout_target_ids"][selected_row])
                sample_record["tokens"].append(
                    {
                        "response_index": response_index,
                        "prediction_index_in_unpadded_batch": prediction_index,
                        "teacher_prediction_index_in_unpadded_batch": int(
                            alignment_evidence["teacher_prediction_indices"][selected_row]
                        ),
                        "selected_logits_row": selected_row,
                        "rollout_token_id": rollout_token_id,
                        "rollout_token": self._opsd_test_token_text(rollout_token_id),
                        "student_target_token_id": student_target_token_id,
                        "teacher_target_token_id": teacher_target_token_id,
                        "target_ids_match_rollout": (
                            student_target_token_id
                            == teacher_target_token_id
                            == aligned_rollout_token_id
                            == rollout_token_id
                        ),
                        "included_in_loss": True,
                        "student_topk": topk_entries(student_topk_ids, student_topk_log_probs),
                        "teacher_topk": topk_entries(teacher_topk_ids, teacher_topk_log_probs_full),
                        "loss_calculation": loss_calculation,
                    }
                )
                recorded_tokens += 1

            sample_record["response_tokens_recorded"] = recorded_tokens
            sample_record["response_tokens_truncated_in_report"] = (
                sample_record["response_tokens_selected"] > recorded_tokens
            )
            records.append(sample_record)
        self._opsd_test_alignment_evidence = None
        return records

    def _aggregate_opsd_loss(self, model_output: dict, data: TensorDict, dp_group=None):
        if "opsd_losses" not in model_output:
            raise KeyError("OPSD aggregation requires model_output['opsd_losses'].")

        opsd_losses = no_padding_2_padding(model_output["opsd_losses"], data)
        response_mask = data["response_mask"]
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        response_mask = response_mask.bool()

        # 对应 1/B * sum_i [1/T_i * sum_t KL_{i,t}]，先 response 内平均，再 batch 内平均。
        global_batch_info = {
            "dp_size": data["dp_size"],
            "batch_num_tokens": data["batch_num_tokens"],
            "global_batch_size": data["global_batch_size"],
            "loss_scale_factor": self.config.actor.get("loss_scale_factor", None),
        }
        opsd_loss = agg_loss(
            loss_mat=opsd_losses,
            loss_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            **global_batch_info,
        )
        loss_coef = tu.get_non_tensor_data(data=data, key="opsd_loss_coef", default=self.opsd_loss_coef)
        opsd_loss = opsd_loss * loss_coef

        valid_loss_sum, _, valid_loss_min, valid_loss_max = compute_masked_loss_stats(
            losses=opsd_losses,
            loss_mask=response_mask,
        )
        metrics = {
            "opsd_loss": Metric(value=opsd_loss.detach(), aggregation=AggregationType.SUM),
            "opsd_token_loss_sum": Metric(value=valid_loss_sum.detach(), aggregation=AggregationType.SUM),
            "opsd_token_loss_min": Metric(value=valid_loss_min.detach(), aggregation=AggregationType.MIN),
            "opsd_token_loss_max": Metric(value=valid_loss_max.detach(), aggregation=AggregationType.MAX),
            "opsd_response_tokens": Metric(value=response_mask.sum().detach(), aggregation=AggregationType.SUM),
            "opsd_teacher_micro_batches": Metric(
                value=opsd_losses.new_tensor(
                    tu.get_non_tensor_data(data=data, key="opsd_teacher_micro_batch_count", default=1)
                ),
                aggregation=AggregationType.SUM,
            ),
            "opsd_teacher_tokens_per_micro_batch": Metric(
                value=opsd_losses.new_tensor(
                    tu.get_non_tensor_data(data=data, key="opsd_teacher_group_tokens", default=0)
                ),
                aggregation=AggregationType.MEAN,
            ),
            "opsd_teacher_tokens_per_micro_batch_max": Metric(
                value=opsd_losses.new_tensor(
                    tu.get_non_tensor_data(data=data, key="opsd_teacher_group_tokens", default=0)
                ),
                aggregation=AggregationType.MAX,
            ),
            "opsd_teacher_max_sequence_len": Metric(
                value=opsd_losses.new_tensor(
                    tu.get_non_tensor_data(data=data, key="opsd_teacher_group_max_sequence_len", default=0)
                ),
                aggregation=AggregationType.MAX,
            ),
        }
        topk_mass_keys = ("opsd_student_topk_mass", "opsd_teacher_mass_on_student_topk")
        present_topk_mass_keys = [key for key in topk_mass_keys if key in model_output]
        if present_topk_mass_keys and len(present_topk_mass_keys) != len(topk_mass_keys):
            raise KeyError("OPSD top-k coverage requires both student and teacher mass tensors.")
        if present_topk_mass_keys:
            metrics.update(
                build_opsd_topk_metric_objects(
                    student_topk_mass=no_padding_2_padding(model_output[topk_mass_keys[0]], data),
                    teacher_mass_on_student_topk=no_padding_2_padding(model_output[topk_mass_keys[1]], data),
                    response_mask=response_mask,
                )
            )
        test_records_json = getattr(self, "_opsd_test_pending_records_json", None)
        if test_records_json is not None:
            metrics["opsd_test_records_json"] = test_records_json
            self._opsd_test_pending_records_json = None
        return opsd_loss, metrics

    def _opsd_actor_loss(
        self,
        model_output: dict = None,
        data: TensorDict = None,
        dp_group=None,
        student_logits: torch.Tensor = None,
        data_format: str = "thd",
    ):
        if student_logits is not None:
            # Selected full-vocabulary rows are consumed inside this micro-batch;
            # only scalar token losses are returned to the engine.
            student_logits = self._compute_opsd_student_selected_logits(
                student_logits=student_logits, data=data, data_format=data_format
            )
            teacher_logits = self._compute_opsd_teacher_selected_logits(
                data=data, student_logits=student_logits, data_format=data_format
            )
            student_probs, student_log_probs, teacher_log_probs = self._select_opsd_vocab_for_loss(
                student_logits=student_logits, teacher_logits=teacher_logits, data=data
            )
            opsd_losses = self._compute_opsd_token_loss(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                data=data,
            )
            output = {"opsd_losses": opsd_losses}
            strategy = tu.get_non_tensor_data(data=data, key="opsd_vocab_strategy", default=self.opsd_vocab_strategy)
            if strategy == "student_truncated":
                student_topk_mass, teacher_mass_on_student_topk = compute_opsd_topk_masses(
                    student_topk_log_probs=student_log_probs,
                    teacher_log_probs_on_student_topk=teacher_log_probs,
                )
                output.update(
                    {
                        "opsd_student_topk_mass": student_topk_mass,
                        "opsd_teacher_mass_on_student_topk": teacher_mass_on_student_topk,
                    }
                )
            test_records = self._build_opsd_test_records(
                data=data,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                token_losses=opsd_losses,
            )
            self._opsd_test_loss_token_ids = None
            if test_records:
                self._opsd_test_pending_records_json = json.dumps(test_records, ensure_ascii=False)
            del teacher_logits
            return output
        return self._aggregate_opsd_loss(model_output=model_output, data=data, dp_group=dp_group)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update_opsd")
    @_with_routing_replay_flag(enabled=True)
    def update_actor_opsd(self, data: TensorDict) -> TensorDict:
        assert self.opsd_enabled, "update_actor_opsd is only valid when OPSD is enabled."
        dataloader_kwargs = tu.get_non_tensor_data(data=data, key="dataloader_kwargs", default={})
        if dataloader_kwargs.get("shuffle", False):
            raise ValueError("OPSD joint micro-batch indices require actor_rollout_ref.actor.shuffle=False.")

        partitions = self._build_opsd_joint_micro_batch_plan(data)
        self._opsd_test_pending_records_json = None
        self._opsd_test_loss_token_ids = None
        self._opsd_test_alignment_evidence = None
        tu.assign_non_tensor_data(data, "precomputed_micro_batch_indices", partitions)
        tu.assign_non_tensor(data, opsd_use_logits_processor=True)
        output = self.actor.train_mini_batch(data=data, loss_function=self._opsd_actor_loss)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
