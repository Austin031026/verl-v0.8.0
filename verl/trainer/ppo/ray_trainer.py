# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.opsd_metrics import finalize_opsd_actor_metrics
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        # 加入对 opsd 的支持
        self.opsd_config = self.config.get("opsd", None)
        self.use_opsd = bool(self.opsd_config is not None and self.opsd_config.get("enabled", False))
        self.opsd_rl_coupling = (
            self.opsd_config.get("loss", {}).get("rl_coupling", "none")
            if self.use_opsd
            else "none"
        )
        self.pure_opsd = self.use_opsd and self.opsd_rl_coupling == "none"
        self.reward_loop_manager = None
        self.opsd_teacher_privileged_input_mode = (
            self.opsd_config.get("teacher", {}).get("privileged_input", {}).get("mode", "answer")
            if self.use_opsd
            else "answer"
        )
        opsd_teacher_config = self.opsd_config.get("teacher", {}) if self.use_opsd else {}
        self.opsd_teacher_max_prompt_length = int(opsd_teacher_config.get("max_prompt_length", 12288))
        self.opsd_teacher_max_context_length = opsd_teacher_config.get("max_context_length", None)
        if self.use_opsd and self.opsd_teacher_max_prompt_length <= 0:
            raise ValueError("OPSD teacher.max_prompt_length must be greater than 0.")
        self.opsd_test_config = self.opsd_config.get("test", {}) if self.use_opsd else {}
        self.opsd_test_enabled = bool(self.opsd_test_config.get("enabled", False))
        self.opsd_test_steps = {int(step) for step in self.opsd_test_config.get("steps", [])}
        self.opsd_test_output_path = None
        self._opsd_test_rollout_projection_evidence = None
        self._opsd_test_privileged_input_evidence = None
        self._opsd_test_actor_transport_evidence = None
        self._opsd_test_report = None
        if self.opsd_test_enabled:
            output_path = self.opsd_test_config.get("output_path", "opsd_test_result.json")
            self.opsd_test_output_path = os.path.abspath(os.path.expanduser(str(output_path)))
            student_model_path = str(self.config.actor_rollout_ref.model.path)
            teacher_model_path = str(self.opsd_config.get("teacher", {}).get("model_path", ""))
            teacher_update_mode = str(self.opsd_config.get("teacher", {}).get("update", {}).get("mode", "none"))
            self._opsd_test_report = {
                "schema_version": "1.0",
                "status": "running",
                "test_mode": "real_verl_runtime",
                "output_path": self.opsd_test_output_path,
                "requested_steps": sorted(self.opsd_test_steps),
                "opsd_config": OmegaConf.to_container(self.opsd_config, resolve=True),
                "modules": {
                    "config": {"status": "pass"},
                    "model_identity": {
                        "status": "pass" if student_model_path == teacher_model_path else "fail",
                        "student_model_path": student_model_path,
                        "teacher_model_path": teacher_model_path,
                    },
                    "teacher_update_mode": {
                        "status": "pass" if teacher_update_mode == "none" else "fail",
                        "expected": "none",
                        "actual": teacher_update_mode,
                    },
                    "rollout_lifecycle": {"status": "pending"},
                    "privileged_input": {"status": "pending"},
                    "teacher_batch": {"status": "pending"},
                    "actor_transport": {"status": "pending"},
                    "teacher_student_alignment": {"status": "pending"},
                    "joint_micro_batching": {"status": "pending"},
                    "vocab_selection": {"status": "pending"},
                    "kl_loss": {"status": "pending"},
                    "loss_aggregation": {"status": "pending"},
                    "optimizer_update": {"status": "pending"},
                    "memory": {"status": "pending"},
                },
                "steps": [],
            }
            self._write_opsd_test_report()

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _write_opsd_test_report(self):
        if not self.opsd_test_enabled:
            return
        output_dir = os.path.dirname(self.opsd_test_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        temp_path = f"{self.opsd_test_output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._opsd_test_report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, self.opsd_test_output_path)

    def record_opsd_test_failure(self, stage: str, error: Exception):
        if not self.opsd_test_enabled:
            return
        self._opsd_test_report["status"] = "fail"
        self._opsd_test_report["failure"] = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "global_step": getattr(self, "global_steps", None),
        }
        self._write_opsd_test_report()

    def _opsd_test_is_enabled_for_step(self) -> bool:
        return self.opsd_test_enabled and (not self.opsd_test_steps or self.global_steps in self.opsd_test_steps)

    @staticmethod
    def _opsd_test_text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _opsd_test_token_ids_sha256(token_ids: list[int]) -> str:
        payload = ",".join(str(int(token_id)) for token_id in token_ids)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _capture_opsd_test_rollout_projection(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        reason_hashes_before: Optional[list[str]] = None,
    ):
        if not self._opsd_test_is_enabled_for_step():
            return

        controller_reward_models = batch.non_tensor_batch.get("reward_model")
        rollout_reward_models = gen_batch.non_tensor_batch.get("reward_model")
        controller_reason_hashes = []
        if controller_reward_models is not None:
            for reward_model in controller_reward_models:
                reason = reward_model.get("reason") if isinstance(reward_model, dict) else None
                controller_reason_hashes.append(
                    self._opsd_test_text_sha256(reason) if isinstance(reason, str) else None
                )

        reason_absent_from_rollout_reward_model = bool(
            rollout_reward_models is None
            or all(
                isinstance(reward_model, dict) and "reason" not in reward_model
                for reward_model in rollout_reward_models
            )
        )
        reason_mode = self.opsd_teacher_privileged_input_mode == "reason"
        self._opsd_test_rollout_projection_evidence = {
            "privileged_input_mode": self.opsd_teacher_privileged_input_mode,
            "reason_absent_from_rollout_reward_model": reason_absent_from_rollout_reward_model,
            "legacy_top_level_reason_absent_from_rollout": "reason" not in gen_batch.non_tensor_batch,
            "controller_reason_preserved": bool(
                not reason_mode
                or (reason_hashes_before is not None and reason_hashes_before == controller_reason_hashes)
            ),
            "num_samples": len(batch),
        }

    @staticmethod
    def _decode_opsd_test_records(values) -> list[dict]:
        records = []
        if values is None:
            return records
        if isinstance(values, str):
            decoded = json.loads(values)
            return decoded if isinstance(decoded, list) else [decoded]
        if isinstance(values, (list, tuple)):
            for value in values:
                records.extend(RayPPOTrainer._decode_opsd_test_records(value))
            return records
        raise TypeError(f"Unsupported OPSD test record container: {type(values)}")

    def _record_opsd_test_step(self, records: list[dict], actor_metrics: dict, is_last_step: bool):
        if not self._opsd_test_is_enabled_for_step():
            return

        max_samples = int(self.opsd_test_config.get("max_samples_per_step", 2))
        records = records[:max_samples]
        token_records = [token for sample in records for token in sample.get("tokens", [])]
        finite_losses = all(
            np.isfinite(token["loss_calculation"]["raw_token_loss"]) for token in token_records
        )
        contribution_matches = all(
            token["loss_calculation"].get("reported_terms_are_diagnostic_subset", False)
            or abs(token["loss_calculation"]["unreported_contribution"]) <= 1e-5
            for token in token_records
        )
        rollout_projection_evidence = self._opsd_test_rollout_projection_evidence or {}
        rollout_projection_ok = bool(
            rollout_projection_evidence.get("reason_absent_from_rollout_reward_model")
            and rollout_projection_evidence.get("legacy_top_level_reason_absent_from_rollout")
            and rollout_projection_evidence.get("controller_reason_preserved")
        )
        privileged_evidence = self._opsd_test_privileged_input_evidence or {}
        privileged_ok = bool(
            privileged_evidence.get("student_input_ids_unchanged")
            and privileged_evidence.get("teacher_response_ids_match_student")
            and privileged_evidence.get("selected_text_matches_source")
            and privileged_evidence.get("teacher_prompt_matches_expected")
        )
        actor_transport_evidence = self._opsd_test_actor_transport_evidence or {}
        actor_transport_ok = bool(
            actor_transport_evidence.get("raw_privileged_text_absent")
            and actor_transport_evidence.get("reason_absent")
            and actor_transport_evidence.get("teacher_tensor_inputs_present")
        )
        alignment_evidence = [
            record.get("alignment_evidence", {}) for record in records if record.get("alignment_evidence")
        ]
        alignment_ok = bool(
            records
            and token_records
            and alignment_evidence
            and all(
                evidence.get("teacher_response_mask_matches_student")
                and evidence.get("student_selected_count") == evidence.get("teacher_selected_count")
                and evidence.get("student_targets_match_rollout")
                and evidence.get("teacher_targets_match_rollout")
                and evidence.get("student_logits_finite")
                and evidence.get("teacher_logits_finite")
                for evidence in alignment_evidence
            )
            and all(token.get("target_ids_match_rollout", False) for token in token_records)
        )
        grad_norm = actor_metrics.get("actor/grad_norm")
        optimizer_ok = grad_norm is not None and np.isfinite(grad_norm) and grad_norm > 0
        current_uids = {record.get("sample_uid") for record in records if record.get("sample_uid") is not None}
        previous_uids = {
            uid
            for step in self._opsd_test_report["steps"]
            for uid in step["modules"]["rollout_lifecycle"].get("rollout_sample_uids", [])
            if uid is not None
        }
        fresh_rollout = bool(current_uids) and current_uids.isdisjoint(previous_uids)
        token_loss_sum = actor_metrics.get("actor/opsd_token_loss_sum")
        response_tokens = actor_metrics.get("actor/opsd_response_tokens")
        token_loss_mean = actor_metrics.get("actor/opsd_token_loss_mean")
        aggregation_ok = bool(
            token_loss_sum is not None
            and response_tokens is not None
            and response_tokens > 0
            and token_loss_mean is not None
            and np.isfinite(token_loss_mean)
            and abs(token_loss_mean - token_loss_sum / response_tokens) <= 1e-8
        )
        memory_allocated = actor_metrics.get("actor/perf/max_memory_allocated_gb")
        memory_reserved = actor_metrics.get("actor/perf/max_memory_reserved_gb")
        memory_ok = bool(
            memory_allocated is not None
            and memory_reserved is not None
            and np.isfinite(memory_allocated)
            and np.isfinite(memory_reserved)
        )

        step_modules = {
            "rollout_lifecycle": {
                "status": "pass" if records and token_records and fresh_rollout and rollout_projection_ok else "fail",
                "fresh_rollout": fresh_rollout,
                "rollout_sample_uids": sorted(current_uids),
                "projection_evidence": rollout_projection_evidence,
            },
            "privileged_input": {
                "status": "pass" if privileged_ok else "fail",
                "evidence": privileged_evidence,
            },
            "teacher_batch": {
                "status": "pass"
                if privileged_ok
                and privileged_evidence.get("teacher_prompt_within_limit")
                and privileged_evidence.get("teacher_context_within_limit")
                and alignment_evidence
                and all(evidence.get("teacher_response_mask_matches_student") for evidence in alignment_evidence)
                else "fail",
                "response_ids_match": privileged_evidence.get("teacher_response_ids_match_student", False),
                "response_mask_matches": bool(
                    alignment_evidence
                    and all(evidence.get("teacher_response_mask_matches_student") for evidence in alignment_evidence)
                ),
                "teacher_prompt_matches_expected": privileged_evidence.get(
                    "teacher_prompt_matches_expected", False
                ),
                "prompt_within_limit": privileged_evidence.get("teacher_prompt_within_limit", False),
                "context_within_limit": privileged_evidence.get("teacher_context_within_limit", False),
            },
            "actor_transport": {
                "status": "pass" if actor_transport_ok else "fail",
                "evidence": actor_transport_evidence,
            },
            "teacher_student_alignment": {
                "status": "pass" if alignment_ok else "fail",
                "checked_response_tokens": len(token_records),
                "evidence": alignment_evidence,
            },
            "joint_micro_batching": {
                "status": "pass" if actor_metrics.get("actor/opsd_teacher_micro_batches", 0) > 0 else "fail",
                "teacher_micro_batches": actor_metrics.get("actor/opsd_teacher_micro_batches"),
                "teacher_tokens_per_micro_batch_max": actor_metrics.get(
                    "actor/opsd_teacher_tokens_per_micro_batch_max"
                ),
                "teacher_max_sequence_len": actor_metrics.get("actor/opsd_teacher_max_sequence_len"),
            },
            "vocab_selection": {
                "status": "pass" if records and contribution_matches else "fail",
                "strategy": self.opsd_config.get("loss", {}).get("vocab_strategy", "full"),
                "reported_contributions_match_token_loss": contribution_matches,
            },
            "kl_loss": {
                "status": "pass" if token_records and finite_losses else "fail",
                "finite_token_losses": finite_losses,
                "opsd_loss": actor_metrics.get("actor/opsd_loss"),
                "opsd_token_loss_mean": actor_metrics.get("actor/opsd_token_loss_mean"),
            },
            "loss_aggregation": {
                "status": "pass" if aggregation_ok else "fail",
                "token_loss_sum": token_loss_sum,
                "response_tokens": response_tokens,
                "token_loss_mean": token_loss_mean,
            },
            "optimizer_update": {
                "status": "pass" if optimizer_ok else "fail",
                "grad_norm": grad_norm,
                "learning_rate": actor_metrics.get("actor/lr"),
            },
            "memory": {
                "status": "pass" if memory_ok else "fail",
                "max_memory_allocated_gb": memory_allocated,
                "max_memory_reserved_gb": memory_reserved,
                "cpu_memory_used_gb": actor_metrics.get("actor/perf/cpu_memory_used_gb"),
            },
        }
        self._opsd_test_report["steps"].append(
            {
                "global_step": self.global_steps,
                "modules": step_modules,
                "samples": records,
            }
        )
        for module_name, result in step_modules.items():
            previous = self._opsd_test_report["modules"][module_name]["status"]
            self._opsd_test_report["modules"][module_name]["status"] = (
                "fail" if previous == "fail" or result["status"] == "fail" else "pass"
            )
        if is_last_step:
            statuses = [module["status"] for module in self._opsd_test_report["modules"].values()]
            self._opsd_test_report["status"] = "pass" if all(status == "pass" for status in statuses) else "fail"
        self._write_opsd_test_report()

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        if self.use_opsd:
            # Keep the complete controller batch for privileged teacher input
            # construction. AgentLoop receives a projection without teacher-only
            # reason text.
            gen_batch = batch.select(
                batch_keys=[],
                non_tensor_batch_keys=list(batch.non_tensor_batch.keys()),
                meta_info_keys=list(batch.meta_info.keys()),
            )
            gen_batch.non_tensor_batch.pop("reason", None)

            if self.pure_opsd:
                gen_batch.non_tensor_batch.pop("reward_model", None)
                gen_batch.non_tensor_batch.pop("data_source", None)
                return gen_batch

            reward_models = batch.non_tensor_batch.get("reward_model")
            if reward_models is not None:
                rollout_reward_models = np.empty(reward_models.shape, dtype=object)
                for i, reward_model in enumerate(reward_models):
                    if not isinstance(reward_model, dict):
                        raise TypeError(
                            "OPSD requires each reward_model to be a dict; "
                            f"sample index {i} has type {type(reward_model).__name__}."
                        )
                    rollout_reward_model = dict(reward_model)
                    rollout_reward_model.pop("reason", None)
                    rollout_reward_models[i] = rollout_reward_model
                gen_batch.non_tensor_batch["reward_model"] = rollout_reward_models
            return gen_batch

        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _merge_rollout_output(self, batch: DataProto, rollout_output: DataProto) -> DataProto:
        """Merge rollout results while keeping the complete OPSD reward_model authoritative."""
        if self.use_opsd:
            rollout_output.non_tensor_batch.pop("reward_model", None)
        return batch.union(rollout_output)

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = self._merge_rollout_output(test_batch, test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _init_reward_loop_manager(self) -> None:
        """Create reward workers only for training modes that consume reward."""
        if self.pure_opsd:
            self.reward_loop_manager = None
            return

        from verl.experimental.reward_loop import RewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """

        """
        第一句：创建 RayResourcePool
        第二句：以这些 RayResourcePool 为 key 建表
        后面：把不同 worker 分配/登记到对应 resource pool 中
        最后：根据这张表创建 worker group
        """
        """对于每个定义的 resource pool创建可以被使用的 ray resource pool"""
        self.resource_pool_manager.create_resource_pool()
        """创造一个全新的dict 去记录每个ray resource pool里面应该有哪些 worker，因为后面同一个资源池里的worker class 要汇总成一个RayWorkerGroup"""
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        """
        要把对应的worker以及这个 worker 的对应的 init 参数打包成一个 RayClassWithInitArgs（本质是一个装饰器），
        目的打包Ray actor class + 构造参数，在后续 RayWorkerGroup 知道资源位置后再创建对应的ray actor
        """
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                opsd_config=self.config.get("opsd"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        """Ray worker group 的初始化，或者是Ray actor 进程被确定了，但是每个 worker group 里面所需要的模型参数啥的还没进行加载"""
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            """根据 resource_pool 创建 Ray actor"""
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)


        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor: # 和lora有关系
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        """从所有的 worker group/或者是ray actor 中找到 actor rollout worker group，并且初始化模型参数"""
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self._init_reward_loop_manager()

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True
        """
        teacher_model_manager 负责创建和持有 teacher server；
        teacher server 负责加载模型和算 logprob；
        teacher client 是给 AgentLoopWorker 用的请求入口，用它异步请求 teacher server。
        """
        # initialize teacher loop manager
        if self.use_teacher_policy:
            """ 多/单teacher的management """
            from verl.experimental.teacher_loop import MultiTeacherModelManager
            """获得teacher的resource pool"""
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            """创建teacher的管理器, 负责输出teacher 的 token logprob/logits"""
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        """
        AgentLoopManager 只负责rollout 以及 调 teacher server 算 teacher_logprobs，最后返回一批训练用数据
        """
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = self.reward_loop_manager is not None and (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        )
        """
        student actor/rollout worker group 包装成可请求的 rollout 推理服务，
        并创建负载均衡 client，供后面的 AgentLoopManager 生成训练样本用。"""
        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        """
        创建 AgentLoopManager
        - 拿到 student llm_client
        - 拿到 teacher_client
        - 拿到 reward_loop_worker_handles
        """
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        """ 设置的是如何在 actor/trainer 最新权重  ->  rollout server 权重同步，重点是studentnt的权重 """
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _build_opsd_answer_privileged_texts(self, batch: DataProto) -> list[str]:
        reward_models = batch.non_tensor_batch.get("reward_model")
        if reward_models is None:
            raise KeyError("OPSD privileged input mode 'answer' requires batch.non_tensor_batch['reward_model'].")

        privileged_texts = []
        for i, reward_model in enumerate(reward_models):
            answer = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            if answer is None:
                raise ValueError(
                    "OPSD privileged input mode 'answer' requires reward_model.ground_truth "
                    f"for sample index {i}."
                )
            privileged_texts.append(str(answer))

        return privileged_texts

    def _build_opsd_reason_privileged_texts(self, batch: DataProto) -> list[str]:
        reward_models = batch.non_tensor_batch.get("reward_model")
        if reward_models is None:
            raise KeyError("OPSD privileged input mode 'reason' requires batch.non_tensor_batch['reward_model'].")

        privileged_texts = []
        for i, reward_model in enumerate(reward_models):
            if not isinstance(reward_model, dict):
                raise TypeError(
                    "OPSD privileged input mode 'reason' requires each reward_model "
                    f"to be a dict; sample index {i} has type {type(reward_model).__name__}."
                )

            reason = reward_model.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "OPSD privileged input mode 'reason' requires a non-empty reward_model.reason "
                    f"string for sample index {i}."
                )

            privileged_texts.append(reason)

        return privileged_texts

    def _attach_opsd_teacher_privileged_info(self, batch: DataProto) -> tuple[DataProto, dict]:
        mode = self.opsd_teacher_privileged_input_mode
        if mode == "answer":
            privileged_texts = self._build_opsd_answer_privileged_texts(batch)
        elif mode == "reason":
            privileged_texts = self._build_opsd_reason_privileged_texts(batch)
        elif mode == "cot_examples":
            raise NotImplementedError("OPSD teacher privileged input mode 'cot_examples' is not implemented yet.")
        else:
            raise ValueError(f"Invalid OPSD teacher privileged input mode: {mode}.")

        batch.non_tensor_batch["opsd_teacher_privileged_text"] = np.array(privileged_texts, dtype=object)
        metrics = {"opsd/teacher_privileged/num_samples": len(privileged_texts)}
        return batch, metrics

    def _build_opsd_teacher_messages(self, raw_prompt: list[dict], privileged_text: str) -> list[dict]:
        if not isinstance(raw_prompt, list):
            raise TypeError("OPSD teacher input requires raw_prompt to be a list of chat messages.")

        teacher_messages = [dict(message) for message in raw_prompt]
        if not teacher_messages:
            raise ValueError("OPSD teacher input requires a non-empty raw_prompt.")

        privileged_label = {"answer": "Answer", "reason": "Reason"}.get(self.opsd_teacher_privileged_input_mode)
        if privileged_label is None:
            raise ValueError(
                "OPSD teacher message construction requires privileged input mode 'answer' or 'reason', "
                f"got {self.opsd_teacher_privileged_input_mode!r}."
            )
        privileged_block = f"\n\nTeacher privileged information:\n{privileged_label}: {privileged_text}"
        if self.opsd_teacher_privileged_input_mode == "reason":
            privileged_block += (
                "\n\nAfter understanding the reference solution, please try to solve this problem "
                "using your own approach below:\n\nAnswer:"
            )
        target_idx = next(
            (idx for idx in range(len(teacher_messages) - 1, -1, -1) if teacher_messages[idx].get("role") == "user"),
            len(teacher_messages) - 1,
        )

        target_message = dict(teacher_messages[target_idx])
        content = target_message.get("content", "")
        if isinstance(content, str):
            target_message["content"] = content + privileged_block
        elif isinstance(content, list):
            # 多模态 content 用 list 表示；这里先只追加文本特权信息，不改已有多模态段。
            target_message["content"] = list(content) + [{"type": "text", "text": privileged_block}]
        else:
            raise TypeError(f"Unsupported raw_prompt content type for OPSD teacher input: {type(content)}.")

        teacher_messages[target_idx] = target_message
        return teacher_messages

    def _tokenize_opsd_teacher_prompt(self, messages: list[dict]) -> list[int]:
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)
        tokenized_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **apply_kwargs,
        )
        return normalize_token_ids(tokenized_prompt)

    def _resolve_opsd_teacher_max_context_length(self) -> Optional[int]:
        max_context_length = getattr(self, "opsd_teacher_max_context_length", None)
        if max_context_length is None:
            return None
        if hasattr(max_context_length, "get"):
            apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            enable_thinking = apply_kwargs.get("enable_thinking", None)
            if enable_thinking is None:
                raise ValueError(
                    "OPSD teacher.max_context_length has per-mode values, but "
                    "data.apply_chat_template_kwargs.enable_thinking is not set."
                )
            mode = "thinking" if bool(enable_thinking) else "no_think"
            max_context_length = max_context_length.get(mode, None)
            if max_context_length is None:
                raise ValueError(f"OPSD teacher.max_context_length is missing mode {mode!r}.")

        max_context_length = int(max_context_length)
        if max_context_length <= 0:
            raise ValueError("OPSD teacher.max_context_length must be greater than 0.")
        return max_context_length

    def _build_opsd_teacher_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        privileged_texts = batch.non_tensor_batch.get("opsd_teacher_privileged_text")
        if raw_prompts is None:
            raise KeyError("OPSD teacher input requires batch.non_tensor_batch['raw_prompt'].")
        if privileged_texts is None:
            raise KeyError("OPSD teacher input requires batch.non_tensor_batch['opsd_teacher_privileged_text'].")

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        response_width = responses.size(1)
        response_attention_mask = batch.batch["attention_mask"][:, -response_width:]
        if response_mask.shape != responses.shape:
            raise ValueError(f"response_mask shape {response_mask.shape} must match responses shape {responses.shape}.")
        if response_attention_mask.shape != responses.shape:
            raise ValueError(
                f"response attention mask shape {response_attention_mask.shape} "
                f"must match responses shape {responses.shape}."
            )

        teacher_prompt_ids = []
        for raw_prompt, privileged_text in zip(raw_prompts, privileged_texts, strict=True):
            teacher_messages = self._build_opsd_teacher_messages(raw_prompt, str(privileged_text))
            teacher_prompt_ids.append(self._tokenize_opsd_teacher_prompt(teacher_messages))

        max_prompt_length = getattr(self, "opsd_teacher_max_prompt_length", 12288)
        prompt_violations = [
            (i, len(prompt_ids))
            for i, prompt_ids in enumerate(teacher_prompt_ids)
            if len(prompt_ids) > max_prompt_length
        ]
        if prompt_violations:
            raise ValueError(
                "OPSD tokenized teacher prompt exceeds teacher.max_prompt_length. "
                f"max_prompt_length={max_prompt_length}, offending_samples={prompt_violations[:8]}."
            )

        response_lens = response_attention_mask.sum(dim=-1).to(torch.long)
        max_context_length = self._resolve_opsd_teacher_max_context_length()
        if max_context_length is not None:
            context_violations = [
                (i, len(prompt_ids), int(response_lens[i].item()))
                for i, prompt_ids in enumerate(teacher_prompt_ids)
                if len(prompt_ids) + int(response_lens[i].item()) > max_context_length
            ]
            if context_violations:
                raise ValueError(
                    "OPSD teacher input exceeds the configured context length before tensor construction. "
                    f"max_context_length={max_context_length}, "
                    f"offending_samples(prompt_len,response_len)={context_violations[:8]}."
                )

        device = responses.device
        dtype = responses.dtype
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("OPSD teacher input requires tokenizer.pad_token_id or tokenizer.eos_token_id.")

        teacher_prompt_width = max(len(prompt_ids) for prompt_ids in teacher_prompt_ids)
        teacher_prompts = torch.full(
            (len(teacher_prompt_ids), teacher_prompt_width),
            fill_value=pad_token_id,
            dtype=dtype,
            device=device,
        )
        teacher_prompt_attention_mask = torch.zeros(
            (len(teacher_prompt_ids), teacher_prompt_width),
            dtype=response_attention_mask.dtype,
            device=device,
        )
        for i, prompt_ids in enumerate(teacher_prompt_ids):
            prompt_tensor = torch.tensor(prompt_ids, dtype=dtype, device=device)
            teacher_prompts[i, -prompt_tensor.numel() :] = prompt_tensor
            teacher_prompt_attention_mask[i, -prompt_tensor.numel() :] = 1

        teacher_attention_mask = torch.cat([teacher_prompt_attention_mask, response_attention_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompts, responses], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        if not torch.equal(teacher_input_ids[:, -response_width:], responses):
            raise ValueError("OPSD teacher input must end with the unchanged student rollout responses.")

        teacher_data = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
        }
        teacher_meta_info = dict(batch.meta_info)
        teacher_meta_info["global_token_num"] = torch.sum(teacher_attention_mask, dim=-1).tolist()
        teacher_batch = DataProto.from_single_dict(teacher_data, meta_info=teacher_meta_info)

        teacher_prompt_lens = torch.tensor([len(prompt_ids) for prompt_ids in teacher_prompt_ids], dtype=torch.float32)
        response_lens_float = response_lens.float()
        metrics = {
            "opsd/teacher_prompt/mean_len": teacher_prompt_lens.mean().item(),
            "opsd/teacher_prompt/max_len": teacher_prompt_lens.max().item(),
            "opsd/teacher_response/mean_len": response_lens_float.mean().item(),
            "opsd/teacher_total/max_len": (teacher_prompt_lens + response_lens_float).max().item(),
        }
        return teacher_batch, metrics

    def _prepare_opsd_teacher_inputs(self, batch: DataProto):
        # OPSD worker 内部会消费这些 teacher_* 张量；这里不在 RayTrainer 里算 teacher logits。
        test_enabled = self._opsd_test_is_enabled_for_step()
        student_input_ids_before = batch.batch["input_ids"].clone() if test_enabled else None
        student_responses_before = batch.batch["responses"].clone() if test_enabled else None
        expected_privileged_texts = None
        privileged_source_field = None
        if test_enabled:
            if self.opsd_teacher_privileged_input_mode == "answer":
                expected_privileged_texts = self._build_opsd_answer_privileged_texts(batch)
                privileged_source_field = "reward_model.ground_truth"
            elif self.opsd_teacher_privileged_input_mode == "reason":
                expected_privileged_texts = self._build_opsd_reason_privileged_texts(batch)
                privileged_source_field = "reward_model.reason"

        batch, metrics = self._attach_opsd_teacher_privileged_info(batch)
        teacher_batch, teacher_batch_metrics = self._build_opsd_teacher_batch(batch)
        metrics.update(teacher_batch_metrics)
        teacher_input_data = {
            "opsd_teacher_input_ids": teacher_batch.batch["input_ids"],
            "opsd_teacher_attention_mask": teacher_batch.batch["attention_mask"],
            "opsd_teacher_position_ids": teacher_batch.batch["position_ids"],
        }
        batch = batch.union(DataProto.from_single_dict(teacher_input_data))
        if test_enabled:
            selected_privileged_texts = [str(text) for text in batch.non_tensor_batch["opsd_teacher_privileged_text"]]
            selected_text_matches_source = selected_privileged_texts == expected_privileged_texts
            raw_prompts = batch.non_tensor_batch["raw_prompt"]
            expected_teacher_prompt_ids = [
                self._tokenize_opsd_teacher_prompt(
                    self._build_opsd_teacher_messages(raw_prompt, privileged_text)
                )
                for raw_prompt, privileged_text in zip(
                    raw_prompts,
                    expected_privileged_texts,
                    strict=True,
                )
            ]
            response_width = student_responses_before.size(1)
            teacher_prompt_width = teacher_batch.batch["input_ids"].size(1) - response_width
            actual_teacher_prompt_ids = []
            for input_ids, attention_mask in zip(
                teacher_batch.batch["input_ids"][:, :teacher_prompt_width],
                teacher_batch.batch["attention_mask"][:, :teacher_prompt_width],
                strict=True,
            ):
                actual_teacher_prompt_ids.append(input_ids[attention_mask.bool()].tolist())
            teacher_prompt_matches_expected = actual_teacher_prompt_ids == expected_teacher_prompt_ids

            reward_models = batch.non_tensor_batch["reward_model"]
            uids = batch.non_tensor_batch.get("uid")
            max_samples = int(self.opsd_test_config.get("max_samples_per_step", 2))
            sample_evidence = []
            for sample_index, (reward_model, selected_text, expected_ids, actual_ids) in enumerate(
                zip(
                    reward_models,
                    selected_privileged_texts,
                    expected_teacher_prompt_ids,
                    actual_teacher_prompt_ids,
                    strict=True,
                )
            ):
                if sample_index >= max_samples:
                    break
                ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
                reason = reward_model.get("reason") if isinstance(reward_model, dict) else None
                sample_evidence.append(
                    {
                        "sample_uid": None if uids is None else str(uids[sample_index]),
                        "selected_text_sha256": self._opsd_test_text_sha256(selected_text),
                        "selected_text_chars": len(selected_text),
                        "ground_truth_sha256": (
                            self._opsd_test_text_sha256(str(ground_truth)) if ground_truth is not None else None
                        ),
                        "reason_sha256": self._opsd_test_text_sha256(reason) if isinstance(reason, str) else None,
                        "reason_chars": len(reason) if isinstance(reason, str) else None,
                        "selected_differs_from_ground_truth": (
                            selected_text != str(ground_truth) if ground_truth is not None else None
                        ),
                        "expected_teacher_prompt_sha256": self._opsd_test_token_ids_sha256(expected_ids),
                        "actual_teacher_prompt_sha256": self._opsd_test_token_ids_sha256(actual_ids),
                        "teacher_prompt_matches_expected": actual_ids == expected_ids,
                        "teacher_prompt_tokens": len(actual_ids),
                    }
                )

            max_context_length = self._resolve_opsd_teacher_max_context_length()
            teacher_total_max_len = teacher_batch_metrics["opsd/teacher_total/max_len"]
            self._opsd_test_privileged_input_evidence = {
                "num_samples": len(batch.batch),
                "student_input_ids_unchanged": bool(
                    torch.equal(student_input_ids_before, batch.batch["input_ids"])
                ),
                "teacher_response_ids_match_student": bool(
                    torch.equal(
                        student_responses_before,
                        teacher_batch.batch["input_ids"][:, -student_responses_before.size(1) :],
                    )
                ),
                "source_field": privileged_source_field,
                "selected_text_matches_source": selected_text_matches_source,
                "teacher_prompt_matches_expected": teacher_prompt_matches_expected,
                "teacher_prompt_mean_len": teacher_batch_metrics["opsd/teacher_prompt/mean_len"],
                "teacher_prompt_max_len": teacher_batch_metrics["opsd/teacher_prompt/max_len"],
                "teacher_total_max_len": teacher_total_max_len,
                "teacher_prompt_within_limit": (
                    teacher_batch_metrics["opsd/teacher_prompt/max_len"] <= self.opsd_teacher_max_prompt_length
                ),
                "teacher_context_within_limit": (
                    max_context_length is None or teacher_total_max_len <= max_context_length
                ),
                "teacher_max_prompt_length": self.opsd_teacher_max_prompt_length,
                "teacher_max_context_length": max_context_length,
                "privileged_input_mode": self.opsd_teacher_privileged_input_mode,
                "sample_evidence": sample_evidence,
            }
        return batch, metrics

    def _update_actor_opsd(self, batch: DataProto) -> tuple[DataProto, list[dict]]:
        # OPSD 专用 actor 更新入口：选中 response 位置的 full-vocab logits
        # 只能在 worker 的 microbatch 内部被消费。
        required_teacher_keys = (
            "opsd_teacher_input_ids",
            "opsd_teacher_attention_mask",
            "opsd_teacher_position_ids",
        )
        missing_teacher_keys = [key for key in required_teacher_keys if key not in batch.batch.keys()]
        if missing_teacher_keys:
            raise KeyError(f"OPSD actor update requires teacher input tensors: {missing_teacher_keys}.")

        # The teacher prompt has already been tokenized into opsd_teacher_* tensors.
        # Do not serialize the raw answer/reason a second time to the actor worker.
        actor_non_tensor_keys = set(batch.non_tensor_batch) - {
            "reward_model",
            "opsd_teacher_privileged_text",
        }
        actor_batch = batch.select(non_tensor_batch_keys=actor_non_tensor_keys)
        if self._opsd_test_is_enabled_for_step():
            self._opsd_test_actor_transport_evidence = {
                "raw_privileged_text_absent": (
                    "reward_model" not in actor_batch.non_tensor_batch
                    and "opsd_teacher_privileged_text" not in actor_batch.non_tensor_batch
                ),
                "reason_absent": (
                    "reason" not in actor_batch.non_tensor_batch
                    and "reward_model" not in actor_batch.non_tensor_batch
                ),
                "teacher_tensor_inputs_present": all(key in actor_batch.batch for key in required_teacher_keys),
                "num_samples": len(actor_batch),
            }
        batch_td = actor_batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        actor_config = self.config.actor_rollout_ref.actor
        rollout_config = self.config.actor_rollout_ref.rollout
        loss_cfg = self.opsd_config.get("loss", {})
        ppo_mini_batch_size = actor_config.ppo_mini_batch_size * rollout_config.n
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            distillation_use_topk=False,
            opsd_use_logits_processor=True,
            opsd_kl_mode=loss_cfg.get("kl_mode", "reverse_kl"),
            opsd_rl_coupling=loss_cfg.get("rl_coupling", "none"),
            opsd_vocab_strategy=loss_cfg.get("vocab_strategy", "full"),
            opsd_student_topk=loss_cfg.get("student_topk", 8),
            opsd_chunked_topk_chunk_size=loss_cfg.get("chunked_topk_chunk_size", 4096),
            opsd_loss_coef=loss_cfg.get("loss_coef", 1.0),
            opsd_temperature=loss_cfg.get("temperature", 1.0),
            temperature=loss_cfg.get("temperature", 1.0),
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=actor_config.ppo_epochs,
            seed=actor_config.data_loader_seed,
            dataloader_kwargs={"shuffle": actor_config.shuffle},
            compute_loss=True,
            opsd_test_enabled=self._opsd_test_is_enabled_for_step(),
            opsd_test_global_step=self.global_steps,
            opsd_test_topk=int(self.opsd_test_config.get("topk", 5)),
            opsd_test_max_samples=int(self.opsd_test_config.get("max_samples_per_worker_micro_batch", 2)),
            opsd_test_max_response_tokens=int(self.opsd_test_config.get("max_response_tokens_per_sample", 32)),
            opsd_test_max_loss_vocab_tokens=int(self.opsd_test_config.get("max_loss_vocab_tokens", 32)),
        )
        actor_output = self.actor_rollout_wg.update_actor_opsd(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        test_records = self._decode_opsd_test_records(actor_output.pop("opsd_test_records_json", None))
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        return actor_output, test_records

    def _post_actor_update(self, timing_raw: dict, is_last_step: bool):
        # actor 参数更新完成后，保存 checkpoint 并把新权重同步给 rollout replica。
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if self.config.trainer.save_freq > 0 and (
            is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

        with marked_timer("update_weights", timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        """涉及到负责恢复训练 checkpoint，或者从头开始训练，来决定如何加载checking point"""
        self._load_checkpoint()
        """_load_checkpoint() 加载的是 trainer/actor 侧权重, rollout server 侧用于生成 response 的权重，还需要同步一次。"""
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function. 就是得有groud truth，然后也得有从rollout中提取答案的func
        """训练开始前，是否先用当前 student 模型跑一次验证集。"""
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            """看是否是不进行训练，只进行验证的模式"""
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return
        """这部分暂时不管"""
        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        """历史上的最慢的一个步骤"""
        last_val_metrics = None
        self.max_steps_duration = 0
        """这一步骤是不是特殊步骤，这一个步骤需不需要记录实验结果"""
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                """dataloader 给出的普通 batch dict 转成 verl 内部统一数据结构 DataProto"""
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                """记录temperature的温度到meta data上去"""
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                """给每一个prompt添加一个独立的uid，对后面的柜组有意义"""
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                """
                把完整训练 batch 整理成 rollout 用 batch，
                保留 prompt tensor 和 reward/分组所需字段，
                去掉无关 non-tensor 字段。
                """
                reason_hashes_before = None
                if self.use_opsd and self.opsd_teacher_privileged_input_mode == "reason":
                    # Training requires reason before rollout; validation only needs ground_truth.
                    reason_texts = self._build_opsd_reason_privileged_texts(batch)
                    if self._opsd_test_is_enabled_for_step():
                        reason_hashes_before = [self._opsd_test_text_sha256(reason) for reason in reason_texts]
                gen_batch = self._get_gen_batch(batch)
                if self.use_opsd:
                    self._capture_opsd_test_rollout_projection(
                        batch=batch,
                        gen_batch=gen_batch,
                        reason_hashes_before=reason_hashes_before,
                    )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                """不管这个分支"""
                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)


                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        """真正开始生成 rollout 数据。去看ray_trainer.py的对应文件，去看rollout的逻辑"""
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = self._merge_rollout_output(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    reward_extra_infos_dict = {}
                    if self.use_opsd and self.opsd_rl_coupling == "none":
                        # Pure OPSD 不走 reward/old_log_prob/advantage；teacher/student logits 在 worker 内部消费。
                        with marked_timer("opsd_teacher_input", timing_raw, color="purple"):
                            batch, opsd_teacher_metrics = self._prepare_opsd_teacher_inputs(batch)
                            metrics.update(opsd_teacher_metrics)

                        with marked_timer("update_actor_opsd", timing_raw, color="red"):
                            actor_output, opsd_test_records = self._update_actor_opsd(batch)

                        self._post_actor_update(timing_raw=timing_raw, is_last_step=is_last_step)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        actor_output_metrics.update(finalize_opsd_actor_metrics(actor_output_metrics))
                        self._record_opsd_test_step(
                            records=opsd_test_records,
                            actor_metrics=actor_output_metrics,
                            is_last_step=is_last_step,
                        )
                        metrics.update(actor_output_metrics)
                    else:
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    raise ValueError(
                                        "Detected conflicting router replay configuration: "
                                        "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                        "cannot be enabled simultaneously. "
                                        "The enable_rollout_routing_replay option is only used in R3 mode; "
                                        "it should not be set when using R2 mode."
                                    )
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup > self.global_steps:
                            # Still in critic warmup, only update weights to wake up rollout replicas.
                            self.checkpoint_manager.update_weights(self.global_steps)
                        else:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)

                            self._post_actor_update(timing_raw=timing_raw, is_last_step=is_last_step)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if not (self.use_opsd and self.opsd_rl_coupling == "none"):
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
