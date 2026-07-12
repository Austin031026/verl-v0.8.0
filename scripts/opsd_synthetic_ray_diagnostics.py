#!/usr/bin/env python3
"""Synthetic OPSD diagnostics that exercise verl data, Ray, and loss paths.

This script deliberately does not load a real model and does not start vLLM.
It builds a tiny verl-style DataProto, simulates generated responses and
student/teacher logits, then calls the OPSD helper methods that are used by the
training path. It also creates the test runner through RayResourcePool and
RayWorkerGroup so worker allocation is covered.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str = ""


class ExpectedReserved(Exception):
    """Raised when a reserved OPSD feature correctly reports NotImplemented."""


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def convert_ids_to_tokens(self, token_id):
        return f"token_{int(token_id)}"

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(self.convert_ids_to_tokens(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        if not tokenize:
            raise ValueError("FakeTokenizer only supports tokenize=True.")

        ids = []
        role_offsets = {"system": 3, "user": 5, "assistant": 7}
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content
                )
            else:
                text = str(content)

            ids.append(role_offsets.get(role, 9))
            ids.extend(10 + (ord(ch) % 80) for ch in text[:96])

        if add_generation_prompt:
            ids.append(99)
        return ids


class FakeVocabTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)


def _make_synthetic_batch() -> DataProto:
    prompts = torch.tensor(
        [
            [11, 12, 13],
            [0, 21, 22],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [31, 32],
            [41, 0],
        ],
        dtype=torch.long,
    )
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    response_mask = torch.tensor(
        [
            [1, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    raw_prompt = [
        [{"role": "user", "content": "What is 1+1?"}],
        [{"role": "user", "content": "What is 2+3?"}],
    ]
    reward_model = [{"ground_truth": "2"}, {"ground_truth": "5"}]

    raw_prompt_array = np.empty(len(raw_prompt), dtype=object)
    raw_prompt_array[:] = raw_prompt
    reward_model_array = np.empty(len(reward_model), dtype=object)
    reward_model_array[:] = reward_model

    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        non_tensors={
            "raw_prompt": raw_prompt_array,
            "reward_model": reward_model_array,
        },
        meta_info={"global_token_num": attention_mask.sum(dim=-1).tolist()},
    )


def _make_trainer_like() -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(data={})
    trainer.tokenizer = FakeTokenizer()
    trainer.opsd_teacher_privileged_input_mode = "answer"
    return trainer


def _make_opsd_worker_like() -> ActorRolloutRefWorker:
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(actor={})
    worker.opsd_vocab_strategy = "full"
    worker.opsd_kl_mode = "reverse_kl"
    worker.opsd_loss_coef = 1.0
    worker.opsd_student_topk = 8
    worker.opsd_chunked_topk_chunk_size = 4096
    worker.opsd_use_tail = False
    worker._rank = 0
    worker.actor = SimpleNamespace(model_config=SimpleNamespace(tokenizer=FakeTokenizer()))
    return worker


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str, atol: float = 1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch, max_abs_diff={diff}")


def _expect_not_implemented(fn: Callable[[], object]):
    try:
        fn()
    except NotImplementedError as exc:
        raise ExpectedReserved(str(exc)) from exc
    raise AssertionError("Expected NotImplementedError, but the call returned normally.")


def _make_logits(num_tokens: int, vocab_size: int, offset: float = 0.0) -> torch.Tensor:
    logits = torch.linspace(-1.2, 1.4, steps=num_tokens * vocab_size, dtype=torch.float32)
    logits = logits.reshape(num_tokens, vocab_size)
    return logits + offset


def _make_teacher_batch() -> tuple[DataProto, DataProto, dict]:
    batch = _make_synthetic_batch()
    trainer = _make_trainer_like()
    batch, privileged_metrics = trainer._attach_opsd_teacher_privileged_info(batch)
    teacher_batch, teacher_metrics = trainer._build_opsd_teacher_batch(batch)
    metrics = {**privileged_metrics, **teacher_metrics}
    return batch, teacher_batch, metrics


def _make_aligned_logits(worker: ActorRolloutRefWorker):
    batch, teacher_batch, _ = _make_teacher_batch()
    prefixed_data = batch.batch.clone()
    for key, value in teacher_batch.batch.items():
        prefixed_data[f"opsd_teacher_{key}"] = value

    teacher_forward = worker._build_opsd_teacher_forward_batch(prefixed_data)
    student_nopad = left_right_2_no_padding(batch.batch.clone())
    teacher_nopad = left_right_2_no_padding(teacher_forward.clone())

    student_ranges = worker._opsd_response_prediction_ranges(student_nopad)
    teacher_ranges = worker._opsd_response_prediction_ranges(teacher_nopad)

    vocab_size = 7
    student_token_count = student_nopad["input_ids"].values().shape[0]
    teacher_token_count = teacher_nopad["input_ids"].values().shape[0]
    student_logits = _make_logits(student_token_count, vocab_size, offset=0.0)
    teacher_logits_full = _make_logits(teacher_token_count, vocab_size, offset=0.35)
    teacher_logits_aligned = torch.zeros_like(student_logits)

    for student_range, teacher_range in zip(student_ranges, teacher_ranges, strict=True):
        student_start, student_end = student_range
        teacher_start, teacher_end = teacher_range
        if student_end - student_start != teacher_end - teacher_start:
            raise AssertionError(
                "Student/teacher response prediction ranges have different lengths: "
                f"student={student_range}, teacher={teacher_range}"
            )
        teacher_logits_aligned[student_start:student_end] = teacher_logits_full[teacher_start:teacher_end]

    return batch, student_logits, teacher_logits_aligned, student_ranges, teacher_ranges


class SyntheticOpsdRayWorker(Worker):
    def __init__(self):
        super().__init__()
        self.opsd_worker = _make_opsd_worker_like()

    def _run_case(self, name: str, fn: Callable[[], str | None], include_traceback: bool) -> CaseResult:
        try:
            detail = fn() or ""
            return CaseResult(name=name, status="PASS", detail=detail)
        except ExpectedReserved as exc:
            return CaseResult(name=name, status="EXPECTED", detail=str(exc))
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if include_traceback:
                detail += "\n" + traceback.format_exc()
            return CaseResult(name=name, status="FAIL", detail=detail)

    def _case_worker_metadata(self):
        if self.world_size < 1:
            raise AssertionError(f"Invalid world_size={self.world_size}")
        if self.rank < 0:
            raise AssertionError(f"Invalid rank={self.rank}")
        return f"rank={self.rank}, world_size={self.world_size}, master={self.get_master_addr_port()}"

    def _case_dataproto_batch(self):
        batch = _make_synthetic_batch()
        if len(batch) != 2:
            raise AssertionError(f"Expected batch size 2, got {len(batch)}")
        if tuple(batch.batch["input_ids"].shape) != (2, 5):
            raise AssertionError(f"Unexpected input_ids shape {tuple(batch.batch['input_ids'].shape)}")
        if batch.non_tensor_batch["reward_model"][0]["ground_truth"] != "2":
            raise AssertionError("reward_model.ground_truth was not preserved")
        return "DataProto keeps prompts/responses/masks/raw_prompt/reward_model."

    def _case_teacher_privileged_answer(self):
        batch = _make_synthetic_batch()
        trainer = _make_trainer_like()
        privileged_texts = trainer._build_opsd_answer_privileged_texts(batch)
        if privileged_texts != ["2", "5"]:
            raise AssertionError(f"Unexpected privileged_texts={privileged_texts}")
        batch, metrics = trainer._attach_opsd_teacher_privileged_info(batch)
        if metrics["opsd/teacher_privileged/num_samples"] != 2:
            raise AssertionError(f"Unexpected metrics={metrics}")
        if list(batch.non_tensor_batch["opsd_teacher_privileged_text"]) != ["2", "5"]:
            raise AssertionError("Privileged answer text was not attached to batch.")
        return "answer mode reads reward_model.ground_truth and attaches teacher text."

    def _case_teacher_message_injection(self):
        trainer = _make_trainer_like()
        messages = trainer._build_opsd_teacher_messages(
            [{"role": "user", "content": "Solve x."}],
            "42",
        )
        content = messages[-1]["content"]
        if "Teacher privileged information" not in content or "Answer: 42" not in content:
            raise AssertionError(f"Privileged block missing from teacher message: {content!r}")
        return "privileged answer is appended to the last user message."

    def _case_teacher_batch(self):
        batch, teacher_batch, metrics = _make_teacher_batch()
        if tuple(teacher_batch.batch["responses"].shape) != tuple(batch.batch["responses"].shape):
            raise AssertionError("Teacher responses shape changed.")
        if not torch.equal(teacher_batch.batch["response_mask"], batch.batch["response_mask"]):
            raise AssertionError("Teacher response_mask differs from student response_mask.")
        if teacher_batch.batch["prompts"].shape[1] <= batch.batch["prompts"].shape[1]:
            raise AssertionError("Teacher prompt did not grow after privileged answer injection.")
        required = {"opsd/teacher_prompt/mean_len", "opsd/teacher_prompt/max_len", "opsd/teacher_response/mean_len"}
        if not required.issubset(metrics):
            raise AssertionError(f"Missing teacher metrics: {required - set(metrics)}")
        return f"teacher_prompt_width={teacher_batch.batch['prompts'].shape[1]}"

    def _case_teacher_forward_batch_keys(self):
        batch, teacher_batch, _ = _make_teacher_batch()
        prefixed_data = batch.batch.clone()
        for key, value in teacher_batch.batch.items():
            prefixed_data[f"opsd_teacher_{key}"] = value
        teacher_forward = self.opsd_worker._build_opsd_teacher_forward_batch(prefixed_data)
        expected_keys = {
            "prompts",
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "response_mask",
        }
        if set(teacher_forward.keys()) != expected_keys:
            raise AssertionError(f"Unexpected teacher forward keys: {set(teacher_forward.keys())}")
        if not torch.equal(teacher_forward["responses"], batch.batch["responses"]):
            raise AssertionError("Teacher forward batch did not preserve responses.")
        return "prefixed opsd_teacher_* tensors are converted to a forward batch."

    def _case_response_prediction_ranges(self):
        batch = _make_synthetic_batch()
        nopad = left_right_2_no_padding(batch.batch.clone())
        ranges = self.opsd_worker._opsd_response_prediction_ranges(nopad)
        expected = [(2, 4), (6, 7)]
        if ranges != expected:
            raise AssertionError(f"Expected ranges={expected}, got {ranges}")
        return f"ranges={ranges}"

    def _case_response_loss_mask_does_not_change_ranges(self):
        batch = _make_synthetic_batch()
        batch.batch["response_mask"][0, 0] = 0
        nopad = left_right_2_no_padding(batch.batch.clone())
        ranges = self.opsd_worker._opsd_response_prediction_ranges(nopad)
        expected = [(2, 4), (6, 7)]
        if ranges != expected:
            raise AssertionError(f"Expected physical response ranges={expected}, got {ranges}")
        return "response_mask changes loss selection without changing teacher/student token alignment."

    def _case_tokenizer_compatibility(self):
        worker = _make_opsd_worker_like()
        student_tokenizer = FakeVocabTokenizer({"<pad>": 0, "a": 1, "b": 2})
        teacher_tokenizer = FakeVocabTokenizer({"<pad>": 0, "a": 1, "b": 2})
        worker.actor = SimpleNamespace(model_config=SimpleNamespace(tokenizer=student_tokenizer))
        worker.opsd_teacher = SimpleNamespace(model_config=SimpleNamespace(tokenizer=teacher_tokenizer))
        worker._validate_opsd_tokenizer_compatibility()

        worker.opsd_teacher.model_config.tokenizer = FakeVocabTokenizer({"<pad>": 0, "a": 2, "b": 1})
        try:
            worker._validate_opsd_tokenizer_compatibility()
        except ValueError:
            return "matching token IDs pass and a permuted teacher vocabulary is rejected."
        raise AssertionError("A mismatched teacher vocabulary was not rejected.")

    def _case_teacher_student_logit_alignment(self):
        _, _, teacher_logits_aligned, student_ranges, teacher_ranges = _make_aligned_logits(self.opsd_worker)
        if not torch.isfinite(teacher_logits_aligned).all():
            raise AssertionError("Aligned teacher logits contain non-finite values.")
        nonzero_rows = torch.nonzero(teacher_logits_aligned.abs().sum(dim=-1), as_tuple=False).flatten().tolist()
        expected_rows = []
        for start, end in student_ranges:
            expected_rows.extend(range(start, end))
        if nonzero_rows != expected_rows:
            raise AssertionError(f"Expected nonzero rows {expected_rows}, got {nonzero_rows}")
        return f"student_ranges={student_ranges}, teacher_ranges={teacher_ranges}"

    def _case_select_full_vocab(self):
        data = TensorDict({}, batch_size=[])
        student = torch.randn(3, 5)
        teacher = torch.randn(3, 5)
        student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
            student, teacher, data
        )
        _assert_close(student_log_probs, torch.log_softmax(student, dim=-1), "full student log probabilities")
        _assert_close(teacher_log_probs, torch.log_softmax(teacher, dim=-1), "full teacher log probabilities")
        _assert_close(student_probs, student_log_probs.exp(), "full student probabilities")
        return "full strategy returns full-vocab probability distributions."

    def _case_student_topk_renorm(self):
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(
            data,
            opsd_vocab_strategy="student_renorm",
            opsd_student_topk=2,
            opsd_use_tail=False,
        )
        student = torch.tensor([[3.0, 0.5, 2.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[0.1, 1.0, 2.5, -0.5]], requires_grad=True)
        student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
            student, teacher, data
        )

        expected_ids = torch.tensor([[0, 2]])
        expected_student_logits = torch.gather(student, dim=-1, index=expected_ids)
        expected_teacher_logits = torch.gather(teacher.detach(), dim=-1, index=expected_ids)
        expected_student_log_probs = torch.log_softmax(expected_student_logits, dim=-1)
        expected_teacher_log_probs = torch.log_softmax(expected_teacher_logits, dim=-1)
        _assert_close(student_probs, expected_student_log_probs.exp(), "student_renorm probabilities")
        _assert_close(student_log_probs, expected_student_log_probs, "student_renorm student log probabilities")
        _assert_close(teacher_log_probs, expected_teacher_log_probs, "student_renorm teacher log probabilities")
        _assert_close(student_probs.sum(dim=-1), torch.ones(1), "student_renorm probability mass")

        loss = self.opsd_worker._compute_opsd_reverse_kl(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
        ).sum()
        loss.backward()
        if student.grad is None or not torch.isfinite(student.grad).all() or student.grad.abs().sum() == 0:
            raise AssertionError(f"student_renorm did not produce a valid student gradient: {student.grad}")
        if teacher.grad is not None:
            raise AssertionError("student_renorm must not backpropagate into teacher logits.")
        return f"loss={loss.item():.8f}, student_grad_norm={student.grad.norm().item():.8f}"

    def _case_student_topk_truncated(self):
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(
            data,
            opsd_vocab_strategy="student_truncated",
            opsd_student_topk=2,
            opsd_chunked_topk_chunk_size=1,
            opsd_use_tail=False,
        )
        student = torch.tensor([[3.0, 0.5, 2.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[0.1, 1.0, 2.5, -0.5]], requires_grad=True)
        student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
            student, teacher, data
        )

        expected_ids = torch.tensor([[0, 2]])
        expected_student_log_probs_full = torch.log_softmax(student, dim=-1)
        expected_teacher_log_probs_full = torch.log_softmax(teacher.detach(), dim=-1)
        expected_student_log_probs = torch.gather(expected_student_log_probs_full, dim=-1, index=expected_ids)
        expected_teacher_log_probs = torch.gather(expected_teacher_log_probs_full, dim=-1, index=expected_ids)
        _assert_close(student_probs, expected_student_log_probs.exp(), "student_truncated probabilities")
        _assert_close(student_log_probs, expected_student_log_probs, "student_truncated student log probabilities")
        _assert_close(teacher_log_probs, expected_teacher_log_probs, "student_truncated teacher log probabilities")
        if not torch.all(student_probs.sum(dim=-1) < 1):
            raise AssertionError("student_truncated must preserve top-k mass from the full-vocab distribution.")

        loss = self.opsd_worker._compute_opsd_reverse_kl(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
        ).sum()
        loss.backward()
        if student.grad is None or not torch.isfinite(student.grad).all() or student.grad.abs().sum() == 0:
            raise AssertionError(f"student_truncated did not produce a valid student gradient: {student.grad}")
        if teacher.grad is not None:
            raise AssertionError("student_truncated must not backpropagate into teacher logits.")

        reference_student = student.detach().clone().requires_grad_(True)
        reference_student_log_probs = torch.log_softmax(reference_student, dim=-1)
        reference_student_probs = reference_student_log_probs.exp()
        reference_topk_probs, reference_topk_ids = torch.topk(
            reference_student_probs, k=2, dim=-1, sorted=True
        )
        reference_topk_log_probs = torch.gather(
            reference_student_log_probs, dim=-1, index=reference_topk_ids
        )
        reference_teacher_log_probs = torch.gather(
            torch.log_softmax(teacher.detach(), dim=-1), dim=-1, index=reference_topk_ids
        )
        reference_loss = self.opsd_worker._compute_opsd_reverse_kl(
            student_probs=reference_topk_probs,
            student_log_probs=reference_topk_log_probs,
            teacher_log_probs=reference_teacher_log_probs,
        ).sum()
        reference_loss.backward()
        _assert_close(loss.detach(), reference_loss.detach(), "student_truncated reference loss")
        _assert_close(student.grad, reference_student.grad, "student_truncated reference gradient")
        return (
            f"loss={loss.item():.8f}, topk_mass={student_probs.sum().item():.8f}, "
            f"student_grad_norm={student.grad.norm().item():.8f}"
        )

    def _case_chunked_topk_log_probs(self):
        logits = torch.tensor(
            [[3.0, 0.5, 2.0, -1.0], [0.1, 1.0, 2.5, -0.5], [-0.2, 0.3, 0.7, 1.4]],
            requires_grad=True,
        )
        topk_ids = torch.topk(logits.detach(), k=2, dim=-1).indices
        actual = _chunked_topk_log_probs(logits=logits, topk_ids=topk_ids, chunk_size=1).float()
        expected = torch.gather(torch.log_softmax(logits.float(), dim=-1), dim=-1, index=topk_ids)
        _assert_close(actual, expected, "chunked top-k log probabilities")

        actual.sum().backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all() or logits.grad.abs().sum() == 0:
            raise AssertionError(f"chunked top-k did not preserve student gradients: {logits.grad}")
        return f"chunks=3, student_grad_norm={logits.grad.norm().item():.8f}"

    def _case_student_topk_full_vocab_equivalence(self):
        student = torch.tensor([[0.4, -0.2, 1.1, 0.3]], dtype=torch.float32)
        teacher = torch.tensor([[0.1, 0.8, -0.5, 0.2]], dtype=torch.float32)
        losses = {}
        for strategy in ("full", "student_renorm", "student_truncated"):
            data = TensorDict({}, batch_size=[])
            tu.assign_non_tensor(
                data,
                opsd_vocab_strategy=strategy,
                opsd_student_topk=student.shape[-1],
                opsd_use_tail=False,
            )
            student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
                student, teacher, data
            )
            losses[strategy] = self.opsd_worker._compute_opsd_reverse_kl(
                student_probs=student_probs,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
            )
        _assert_close(losses["student_renorm"], losses["full"], "student_renorm K=V loss")
        _assert_close(losses["student_truncated"], losses["full"], "student_truncated K=V loss")
        return f"full_vocab_loss={losses['full'].item():.8f}"

    def _case_reverse_kl_matches_manual(self):
        student = torch.tensor([[0.2, -0.1, 0.7], [1.0, 0.0, -0.5]], dtype=torch.float32)
        teacher = torch.tensor([[0.0, 0.4, 0.1], [-0.2, 0.6, 0.3]], dtype=torch.float32)
        student_log_probs = torch.log_softmax(student, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        student_probs = student_log_probs.exp()
        actual = self.opsd_worker._compute_opsd_reverse_kl(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
        )
        expected = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        _assert_close(actual, expected, "reverse KL")
        return f"loss={actual.tolist()}"

    def _case_token_loss_and_aggregate(self):
        batch, student_logits, teacher_logits, _, _ = _make_aligned_logits(self.opsd_worker)
        data = TensorDict({}, batch_size=[])
        student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
            student_logits, teacher_logits, data
        )
        token_losses = self.opsd_worker._compute_opsd_token_loss(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            data=data,
        )

        aggregate_data = batch.batch.clone()
        tu.assign_non_tensor(
            aggregate_data,
            dp_size=1,
            batch_num_tokens=int(aggregate_data["response_mask"].sum().item()),
            global_batch_size=aggregate_data.batch_size[0],
        )
        actual_loss, metrics = self.opsd_worker._aggregate_opsd_loss(
            {"opsd_losses": token_losses},
            aggregate_data,
            dp_group=None,
        )
        padded_losses = no_padding_2_padding(token_losses, aggregate_data)
        response_mask = aggregate_data["response_mask"].bool()
        seq_token_counts = response_mask.sum(dim=-1)
        seq_losses = (padded_losses * response_mask).sum(dim=-1) / (seq_token_counts + 1e-8)
        seq_mask = (seq_token_counts > 0).float()
        expected_loss = (seq_losses * seq_mask).sum() / aggregate_data.batch_size[0]
        _assert_close(actual_loss, expected_loss, "aggregated OPSD loss")

        response_tokens = metrics["opsd_response_tokens"].aggregate()
        if response_tokens != 3:
            raise AssertionError(f"Expected 3 response tokens, got {response_tokens}")
        return f"opsd_loss={actual_loss.item():.8f}, response_tokens={response_tokens}"

    def _case_real_loss_json_evidence(self):
        batch, student_logits, teacher_logits, _, _ = _make_aligned_logits(self.opsd_worker)
        data = left_right_2_no_padding(batch.batch.clone())
        tu.assign_non_tensor(
            data,
            opsd_vocab_strategy="student_truncated",
            opsd_student_topk=2,
            opsd_chunked_topk_chunk_size=2,
            opsd_use_tail=False,
            opsd_kl_mode="reverse_kl",
            opsd_loss_coef=1.0,
            opsd_test_enabled=True,
            opsd_test_global_step=1,
            opsd_test_topk=3,
            opsd_test_max_samples=2,
            opsd_test_max_response_tokens=4,
            opsd_test_max_loss_vocab_tokens=2,
        )
        student_probs, student_log_probs, teacher_log_probs = self.opsd_worker._select_opsd_vocab_for_loss(
            student_logits, teacher_logits, data
        )
        token_losses = self.opsd_worker._compute_opsd_token_loss(
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            data=data,
        )
        records = self.opsd_worker._build_opsd_test_records(
            data=data,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_probs=student_probs,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            token_losses=token_losses,
        )
        encoded = json.dumps(records, ensure_ascii=False)
        decoded = json.loads(encoded)
        tokens = [token for sample in decoded for token in sample["tokens"]]
        if len(tokens) != 3:
            raise AssertionError(f"Expected three masked rollout tokens, got {len(tokens)}")
        for token in tokens:
            calculation = token["loss_calculation"]
            if len(token["student_topk"]) != 3 or len(token["teacher_topk"]) != 3:
                raise AssertionError("Diagnostic top-k records are incomplete.")
            if len(calculation["terms"]) != 2:
                raise AssertionError("Loss vocabulary terms do not match student_topk=2.")
            if abs(calculation["unreported_contribution"]) > 1e-6:
                raise AssertionError(f"Loss terms do not reconstruct token loss: {calculation}")
        return f"samples={len(decoded)}, tokens={len(tokens)}, json_bytes={len(encoded)}"

    def _case_forward_kl_reserved(self):
        data = TensorDict({}, batch_size=[])
        student = torch.randn(2, 3)
        teacher = torch.randn(2, 3)
        student_log_probs = torch.log_softmax(student, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        _expect_not_implemented(
            lambda: self.opsd_worker._compute_opsd_kl_by_mode(
                "forward_kl",
                student_log_probs.exp(),
                student_log_probs,
                teacher_log_probs,
                data,
            )
        )

    def _case_topk_union_reserved(self):
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(data, opsd_vocab_strategy="union_renorm")
        student = torch.randn(2, 3)
        teacher = torch.randn(2, 3)
        _expect_not_implemented(lambda: self.opsd_worker._select_opsd_vocab_for_loss(student, teacher, data))

    def run_all(self, include_traceback: bool = False):
        cases = [
            ("ray.worker_metadata", self._case_worker_metadata),
            ("data.dataproto_synthetic_batch", self._case_dataproto_batch),
            ("teacher.privileged_answer", self._case_teacher_privileged_answer),
            ("teacher.message_injection", self._case_teacher_message_injection),
            ("teacher.batch_builds_prefixed_inputs", self._case_teacher_batch),
            ("teacher.forward_batch_keys", self._case_teacher_forward_batch_keys),
            ("alignment.response_prediction_ranges", self._case_response_prediction_ranges),
            ("alignment.response_loss_mask_independence", self._case_response_loss_mask_does_not_change_ranges),
            ("alignment.teacher_student_logits", self._case_teacher_student_logit_alignment),
            ("compatibility.tokenizer_id_mapping", self._case_tokenizer_compatibility),
            ("loss.select_full_vocab", self._case_select_full_vocab),
            ("loss.student_topk_renorm", self._case_student_topk_renorm),
            ("loss.student_topk_truncated", self._case_student_topk_truncated),
            ("loss.chunked_topk_log_probs", self._case_chunked_topk_log_probs),
            ("loss.student_topk_full_vocab_equivalence", self._case_student_topk_full_vocab_equivalence),
            ("loss.reverse_kl_matches_manual", self._case_reverse_kl_matches_manual),
            ("loss.token_loss_and_seq_mean_token_mean", self._case_token_loss_and_aggregate),
            ("test_json.real_loss_evidence", self._case_real_loss_json_evidence),
            ("reserved.forward_kl", self._case_forward_kl_reserved),
            ("reserved.topk_union", self._case_topk_union_reserved),
        ]
        return [asdict(self._run_case(name, fn, include_traceback)) for name, fn in cases]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic OPSD diagnostics through a RayWorkerGroup.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of CPU Ray workers to create.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write the result JSON.")
    parser.add_argument("--traceback", action="store_true", help="Include Python tracebacks in failed case details.")
    return parser.parse_args()


def _print_results(results: list[dict]):
    width = max(len(item["name"]) for item in results)
    for item in results:
        line = f"{item['status']:<8} {item['name']:<{width}}"
        if item["detail"]:
            line += f"  {item['detail']}"
        print(line)

    failures = [item for item in results if item["status"] == "FAIL"]
    expected = [item for item in results if item["status"] == "EXPECTED"]
    passed = [item for item in results if item["status"] == "PASS"]
    print()
    print(f"Summary: PASS={len(passed)} EXPECTED={len(expected)} FAIL={len(failures)}")


def main() -> int:
    args = _parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    import ray

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=max(args.num_workers, 2),
            log_to_driver=False,
        )

    name_prefix = f"opsd_synth_{os.getpid()}_"
    resource_pool = RayResourcePool(
        process_on_nodes=[args.num_workers],
        use_gpu=False,
        name_prefix=name_prefix,
        max_colocate_count=1,
    )
    ray_worker_cls = RayClassWithInitArgs(ray.remote(SyntheticOpsdRayWorker))
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_worker_cls,
        use_gpu=False,
        name_prefix=name_prefix,
    )

    nested_results = worker_group.execute_all_sync("run_all", include_traceback=args.traceback)
    results = [item for worker_results in nested_results for item in worker_results]
    _print_results(results)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")

    ray.shutdown()
    return 1 if any(item["status"] == "FAIL" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
