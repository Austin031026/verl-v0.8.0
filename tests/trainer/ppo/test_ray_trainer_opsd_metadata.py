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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": list(range(len(text)))}


def _make_trainer(*, use_opsd: bool, privileged_input_mode: str = "reason") -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_opsd = use_opsd
    trainer.opsd_teacher_privileged_input_mode = privileged_input_mode
    trainer.tokenizer = _CharacterTokenizer()
    trainer.opsd_teacher_max_prompt_length = 12288
    return trainer


def _make_batch() -> DataProto:
    raw_prompts = np.empty(1, dtype=object)
    raw_prompts[0] = [{"role": "user", "content": "What is 6 * 7?"}]
    reward_models = np.empty(1, dtype=object)
    reward_models[0] = {"ground_truth": "42", "reason": "Multiply six by seven."}
    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8)},
        non_tensors={
            "raw_prompt": raw_prompts,
            "reward_model": reward_models,
            "reason": np.array(["legacy top-level reason"], dtype=object),
            "uid": np.array(["sample-0"], dtype=object),
        },
        meta_info={"temperature": 1.0},
    )


def test_opsd_gen_batch_is_non_destructive_and_excludes_reason():
    trainer = _make_trainer(use_opsd=True)
    batch = _make_batch()

    gen_batch = trainer._get_gen_batch(batch)

    assert "raw_prompt" in batch.non_tensor_batch
    assert batch.non_tensor_batch["reward_model"][0]["reason"] == "Multiply six by seven."
    assert batch.non_tensor_batch["reason"][0] == "legacy top-level reason"
    assert "reason" not in gen_batch.non_tensor_batch
    assert "reason" not in gen_batch.non_tensor_batch["reward_model"][0]
    assert gen_batch.non_tensor_batch["reward_model"][0]["ground_truth"] == "42"
    assert gen_batch.non_tensor_batch["reward_model"][0] is not batch.non_tensor_batch["reward_model"][0]


def test_reason_mode_test_evidence_proves_rollout_projection_isolated_reason():
    trainer = _make_trainer(use_opsd=True, privileged_input_mode="reason")
    trainer.opsd_test_enabled = True
    trainer.opsd_test_steps = {1}
    trainer.global_steps = 1
    batch = _make_batch()
    reason_hashes_before = [
        trainer._opsd_test_text_sha256(batch.non_tensor_batch["reward_model"][0]["reason"])
    ]

    gen_batch = trainer._get_gen_batch(batch)
    trainer._capture_opsd_test_rollout_projection(batch, gen_batch, reason_hashes_before)

    evidence = trainer._opsd_test_rollout_projection_evidence
    assert evidence["privileged_input_mode"] == "reason"
    assert evidence["reason_absent_from_rollout_reward_model"] is True
    assert evidence["legacy_top_level_reason_absent_from_rollout"] is True
    assert evidence["controller_reason_preserved"] is True


def test_reason_mode_gen_batch_does_not_require_validation_reason():
    trainer = _make_trainer(use_opsd=True, privileged_input_mode="reason")
    batch = _make_batch()
    batch.non_tensor_batch["reward_model"][0] = {"ground_truth": "42"}
    batch.non_tensor_batch.pop("reason")

    gen_batch = trainer._get_gen_batch(batch)

    assert gen_batch.non_tensor_batch["reward_model"][0] == {"ground_truth": "42"}


def test_opsd_merge_keeps_controller_metadata_and_adds_rollout_results():
    trainer = _make_trainer(use_opsd=True)
    batch = _make_batch()
    projected_reward_models = np.empty(1, dtype=object)
    projected_reward_models[0] = {"ground_truth": "42"}
    rollout_output = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2]], dtype=torch.long)},
        non_tensors={
            "raw_prompt": batch.non_tensor_batch["raw_prompt"],
            "reward_model": projected_reward_models,
            "__num_turns__": np.array([2], dtype=np.int32),
        },
    )

    merged = trainer._merge_rollout_output(batch, rollout_output)

    assert merged.non_tensor_batch["reward_model"][0]["reason"] == "Multiply six by seven."
    assert merged.non_tensor_batch["raw_prompt"][0][0]["content"] == "What is 6 * 7?"
    assert merged.non_tensor_batch["__num_turns__"].tolist() == [2]
    torch.testing.assert_close(merged.batch["responses"], torch.tensor([[1, 2]], dtype=torch.long))


def test_reason_mode_reads_only_reason_without_a_separate_reason_length_limit():
    trainer = _make_trainer(use_opsd=True)
    reward_models = np.empty(1, dtype=object)
    reward_models[0] = {"reason": "reason may use the available teacher prompt budget"}
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8)},
        non_tensors={"reward_model": reward_models},
    )

    assert trainer._build_opsd_reason_privileged_texts(batch) == [
        "reason may use the available teacher prompt budget"
    ]


@pytest.mark.parametrize("invalid_reason", [None, "", "   ", 42])
def test_reason_mode_rejects_missing_empty_or_non_string_reason(invalid_reason):
    trainer = _make_trainer(use_opsd=True, privileged_input_mode="reason")
    reward_models = np.empty(1, dtype=object)
    reward_models[0] = {"ground_truth": "42", "reason": invalid_reason}
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8)},
        non_tensors={"reward_model": reward_models},
    )

    with pytest.raises(ValueError, match="non-empty reward_model.reason"):
        trainer._build_opsd_reason_privileged_texts(batch)


@pytest.mark.parametrize(
    ("mode", "expected_label", "expected_text"),
    [("answer", "Answer", "42"), ("reason", "Reason", "Multiply six by seven.")],
)
def test_teacher_message_uses_the_configured_privileged_source(mode, expected_label, expected_text):
    trainer = _make_trainer(use_opsd=True, privileged_input_mode=mode)
    batch = _make_batch()
    batch, _ = trainer._attach_opsd_teacher_privileged_info(batch)

    messages = trainer._build_opsd_teacher_messages(
        batch.non_tensor_batch["raw_prompt"][0],
        batch.non_tensor_batch["opsd_teacher_privileged_text"][0],
    )

    assert messages[0]["content"] == (
        f"What is 6 * 7?\n\nTeacher privileged information:\n{expected_label}: {expected_text}"
    )
    assert batch.non_tensor_batch["raw_prompt"][0][0]["content"] == "What is 6 * 7?"


def test_teacher_prompt_limit_applies_after_question_reason_and_template_are_combined():
    trainer = _make_trainer(use_opsd=True)
    trainer.opsd_teacher_max_prompt_length = 4
    trainer.opsd_teacher_max_context_length = None
    trainer._tokenize_opsd_teacher_prompt = lambda messages: [1, 2, 3, 4, 5]
    raw_prompts = np.empty(1, dtype=object)
    raw_prompts[0] = [{"role": "user", "content": "question"}]
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[31, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        },
        non_tensors={
            "raw_prompt": raw_prompts,
            "opsd_teacher_privileged_text": np.array(["reason"], dtype=object),
        },
    )

    with pytest.raises(ValueError, match="teacher.max_prompt_length"):
        trainer._build_opsd_teacher_batch(batch)


@pytest.mark.parametrize(("enable_thinking", "expected"), [(False, 16000), (True, 32000)])
def test_teacher_context_limit_is_selected_from_chat_template_mode(enable_thinking: bool, expected: int):
    trainer = _make_trainer(use_opsd=True)
    trainer.config = SimpleNamespace(
        data={"apply_chat_template_kwargs": {"enable_thinking": enable_thinking}}
    )
    trainer.opsd_teacher_max_context_length = {"no_think": 16000, "thinking": 32000}

    assert trainer._resolve_opsd_teacher_max_context_length() == expected
