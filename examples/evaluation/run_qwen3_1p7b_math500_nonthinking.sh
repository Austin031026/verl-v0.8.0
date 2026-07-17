#!/usr/bin/env bash
# Evaluate Qwen3-1.7B in non-thinking mode on MATH-500.

set -euo pipefail

FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J
CONDA_ENV=$FENG_J/conda/envs/verl_v080_official_script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_PATH=Qwen/Qwen3-1.7B
DATA_DIR=$FENG_J/data/MATH-500
RAW_DATA=$DATA_DIR/raw
TEST_FILE=$DATA_DIR/test_verl.parquet
RESULTS_ROOT=$FENG_J/benchmark_results/math500

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=10240
VAL_BATCH_SIZE=80
VAL_N=4

NGPUS_PER_NODE=4
MAX_NUM_SEQS_PER_GPU=60
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.9

TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

# Planning assumption only; the post-run summary reports observed lengths.
EXPECTED_RESPONSE_LENGTH=1024
PHYSICAL_VRAM_GIB=79.26
RUNTIME_RESERVE_GIB=6.0

RUN_ID=qwen3_1p7b_math500_nothinking_pass4_8k_seq60_mem09_$(date +%Y%m%d_%H%M%S)
RESULT_DIR=$RESULTS_ROOT/$RUN_ID

source "$FENG_J/conda/miniconda3/etc/profile.d/conda.sh"
export CONDARC=$FENG_J/conda/.condarc
conda activate "$CONDA_ENV"

export HF_HOME=$FENG_J/hf
export HF_HUB_CACHE=$FENG_J/hf/hub
export HUGGINGFACE_HUB_CACHE=$FENG_J/hf/hub
export HF_DATASETS_CACHE=$FENG_J/hf/datasets
export TRANSFORMERS_CACHE=$FENG_J/hf/hub
export HF_XET_CACHE=$FENG_J/hf/xet
export PIP_CACHE_DIR=$FENG_J/pip
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_jf42bamu

mkdir -p "$DATA_DIR" "$RESULT_DIR/validation" "$RESULT_DIR/checkpoints" "$RAY_TMPDIR"
cd "$REPO_ROOT"

# Build the verl-format parquet when it is absent or does not match this prompt.
if ! python3 - "$TEST_FILE" <<'PY'
import os
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
required_text = r"Please reason step by step, and put your final answer within \boxed{}."
if not os.path.isfile(path):
    raise SystemExit(1)
table = pq.read_table(path, columns=["data_source", "prompt"])
if table.num_rows != 500:
    raise SystemExit(1)
if set(table.column("data_source").to_pylist()) != {"HuggingFaceH4/MATH-500"}:
    raise SystemExit(1)
prompt = table.column("prompt")[0].as_py()[0]["content"]
if required_text not in prompt:
    raise SystemExit(1)
PY
then
    python3 - "$RAW_DATA" "$TEST_FILE" <<'PY'
import os
import sys
from collections import Counter

from datasets import load_dataset, load_from_disk

raw_path, output_path = sys.argv[1:]
loaded = load_from_disk(raw_path) if os.path.exists(raw_path) else load_dataset("HuggingFaceH4/MATH-500")
dataset = loaded["test"]

problem_counts = Counter(" ".join(example["problem"].split()) for example in dataset)
if len(dataset) != 500 or len(problem_counts) != 500:
    raise RuntimeError(f"Expected 500 unique problems, got {len(dataset)} rows and {len(problem_counts)} unique")

instruction = r"Please reason step by step, and put your final answer within \boxed{}."


def process(example, index):
    return {
        "data_source": "HuggingFaceH4/MATH-500",
        "prompt": [{"role": "user", "content": f'{example["problem"]}\n\n{instruction}'}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["answer"]},
        "extra_info": {
            "split": "test",
            "index": index,
            "unique_id": example["unique_id"],
            "subject": example["subject"],
            "level": example["level"],
        },
    }


processed = dataset.map(
    process,
    with_indices=True,
    remove_columns=dataset.column_names,
    load_from_cache_file=False,
)
processed.to_parquet(output_path)
print(f"Prepared {len(processed)} MATH-500 rows at {output_path}")
PY
fi

python3 - <<PY
import math

max_prompt = $MAX_PROMPT_LENGTH
max_response = $MAX_RESPONSE_LENGTH
expected_response = $EXPECTED_RESPONSE_LENGTH
val_batch = $VAL_BATCH_SIZE
n = $VAL_N
gpus = $NGPUS_PER_NODE
max_num_seqs = $MAX_NUM_SEQS_PER_GPU
gpu_util = $GPU_MEMORY_UTILIZATION
physical_vram_gib = $PHYSICAL_VRAM_GIB
runtime_reserve_gib = $RUNTIME_RESERVE_GIB

kv_bytes_per_token = 2 * 28 * 8 * 128 * 2
expected_sequence_gib = (max_prompt + expected_response) * kv_bytes_per_token / 1024**3
worst_sequence_gib = (max_prompt + max_response) * kv_bytes_per_token / 1024**3
expected_kv_gib_per_gpu = expected_sequence_gib * max_num_seqs
worst_kv_gib_per_gpu = worst_sequence_gib * max_num_seqs
model_weight_gib = 1.7e9 * 2 / 1024**3
vllm_budget_gib = physical_vram_gib * gpu_util
estimated_usable_kv_gib = max(0.0, vllm_budget_gib - model_weight_gib - runtime_reserve_gib)
estimated_full_length_seqs = math.floor(estimated_usable_kv_gib / worst_sequence_gib)

print("========== VALIDATION PREFLIGHT ==========")
print(f"model                              : $MODEL_PATH")
print("mode                               : non-thinking")
print(f"validation file                    : $TEST_FILE")
print(f"max prompt length                  : {max_prompt} tokens")
print(f"max response length                : {max_response} tokens")
print(f"validation prompt batch            : {val_batch}")
print(f"samples per prompt                 : {n}")
print(f"queued responses per batch         : {val_batch * n}")
print(f"max_num_seqs per GPU               : {max_num_seqs}")
print(f"cluster active-sequence cap        : {max_num_seqs * gpus}")
print(f"vLLM gpu_memory_utilization        : {gpu_util}")
print(f"nominal vLLM budget per GPU        : {vllm_budget_gib:.2f} GiB")
print(f"expected response assumption       : {expected_response} tokens")
print(f"expected KV cache per sequence     : {expected_sequence_gib:.3f} GiB")
print(f"expected active KV cache per GPU   : {expected_kv_gib_per_gpu:.2f} GiB")
print(f"worst KV cache per sequence        : {worst_sequence_gib:.3f} GiB")
print(f"worst active KV cache per GPU      : {worst_kv_gib_per_gpu:.2f} GiB")
print(f"estimated BF16 model weights/GPU   : {model_weight_gib:.2f} GiB")
print(f"assumed runtime/headroom reserve   : {runtime_reserve_gib:.2f} GiB")
print(f"estimated full-length seq capacity : {estimated_full_length_seqs}/GPU")
print("note                               : vLLM startup KV report is authoritative")
print("sampling                           : temperature=0.7, top_p=0.8, top_k=20")
print("==========================================")
PY

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TEST_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=1 \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples=-1 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.validation_shuffle=False \
    data.dataloader_num_workers=0 \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.actor.fsdp_config.forward_only=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    critic.enable=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TOP_P" \
    actor_rollout_ref.rollout.top_k="$TOP_K" \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS_PER_GPU" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    'trainer.logger=["console"]' \
    trainer.project_name=qwen3_1p7b_math500_eval \
    trainer.experiment_name="$RUN_ID" \
    trainer.validation_data_dir="$RESULT_DIR/validation" \
    trainer.default_local_dir="$RESULT_DIR/checkpoints" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    2>&1 | tee "$RESULT_DIR/launcher.log"

export GENERATION_FILE=$RESULT_DIR/validation/0.jsonl
export RESULT_DIR
export MODEL_PATH
export MAX_RESPONSE_LENGTH

python3 - <<'PY'
import json
import os
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer

from verl.utils.reward_score.math_reward import last_boxed_only_string

generation_file = os.environ["GENERATION_FILE"]
result_dir = os.environ["RESULT_DIR"]
max_response_length = int(os.environ["MAX_RESPONSE_LENGTH"])

with open(generation_file, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle]

groups = defaultdict(list)
for row in rows:
    groups[row["input"]].append(float(row["score"]) > 0.5)

sample_counts = Counter(map(len, groups.values()))
if len(groups) != 500 or sample_counts != {4: 500}:
    raise RuntimeError(f"Expected 500 prompts x 4 samples, got {len(groups)} prompts and {sample_counts}")

total_correct = sum(sum(values) for values in groups.values())
pass1 = total_correct / len(rows)
first_draw_pass1 = sum(values[0] for values in groups.values()) / len(groups)
pass4 = sum(any(values) for values in groups.values()) / len(groups)

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
lengths = np.array(
    [len(tokenizer.encode(row["output"], add_special_tokens=False)) for row in rows],
    dtype=np.int64,
)
unextracted = sum(last_boxed_only_string(row["output"]) is None for row in rows)

summary = {
    "problems": len(groups),
    "samples_per_problem": 4,
    "generations": len(rows),
    "correct_generations": total_correct,
    "pass1_estimator": pass1,
    "first_draw_pass1": first_draw_pass1,
    "empirical_pass4": pass4,
    "mean_output_tokens": float(lengths.mean()),
    "p50_output_tokens": float(np.percentile(lengths, 50)),
    "p90_output_tokens": float(np.percentile(lengths, 90)),
    "p95_output_tokens": float(np.percentile(lengths, 95)),
    "p99_output_tokens": float(np.percentile(lengths, 99)),
    "max_output_tokens": int(lengths.max()),
    "at_output_limit": int((lengths >= max_response_length).sum()),
    "over_2048_tokens": int((lengths > 2048).sum()),
    "unextracted_answers": int(unextracted),
}

with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

print("========== MATH-500 RESULTS ==========")
for key, value in summary.items():
    print(f"{key:24}: {value}")
print(f"results directory       : {result_dir}")
print("======================================")
PY
