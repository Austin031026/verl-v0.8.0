#!/usr/bin/env bash
# Evaluate Qwen3-1.7B in non-thinking mode on the full GSM8K test split.

set -euo pipefail

FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J
CONDA_ENV=$FENG_J/conda/envs/verl_v080_official_script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_PATH=Qwen/Qwen3-1.7B
DATA_DIR=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/gsm8k
TRAIN_FILE=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/gsm8k/train.parquet
TEST_FILE=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/gsm8k/test.parquet
RESULTS_ROOT=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/benchmark_results/gsm8k

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=6144
VAL_BATCH_SIZE=80
VAL_N=4

NGPUS_PER_NODE=6
MAX_NUM_SEQS_PER_GPU=20
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.8

TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

RUN_ID=qwen3_1p7b_gsm8k_nothinking_pass4_4k_batch80_$(date +%Y%m%d_%H%M%S)
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

# Rebuild stale parquet files so validation uses the current boxed-answer prompt.
if ! python3 - "$TRAIN_FILE" "$TEST_FILE" <<'PY'
import os
import sys

import pyarrow.parquet as pq

required_text = r"\boxed{18}"
for path in sys.argv[1:]:
    if not os.path.isfile(path):
        raise SystemExit(1)
    table = pq.read_table(path, columns=["prompt"])
    prompt = table.column("prompt")[0].as_py()[0]["content"]
    if required_text not in prompt:
        raise SystemExit(1)
PY
then
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi

python3 - <<PY
max_prompt = $MAX_PROMPT_LENGTH
max_response = $MAX_RESPONSE_LENGTH
val_batch = $VAL_BATCH_SIZE
n = $VAL_N
gpus = $NGPUS_PER_NODE
max_num_seqs = $MAX_NUM_SEQS_PER_GPU
gpu_util = $GPU_MEMORY_UTILIZATION

kv_bytes_per_token = 2 * 28 * 8 * 128 * 2
sequence_gib = (max_prompt + max_response) * kv_bytes_per_token / 1024**3
kv_gib_per_gpu = sequence_gib * max_num_seqs

print("========== VALIDATION PREFLIGHT ==========")
print(f"model                         : $MODEL_PATH")
print("mode                          : non-thinking")
print(f"train file                    : $TRAIN_FILE")
print(f"validation file               : $TEST_FILE")
print(f"max prompt length             : {max_prompt} tokens")
print(f"max response length           : {max_response} tokens")
print(f"validation prompt batch       : {val_batch}")
print(f"samples per prompt            : {n}")
print(f"queued responses per batch    : {val_batch * n}")
print(f"max_num_seqs per GPU          : {max_num_seqs}")
print(f"cluster active-sequence cap   : {max_num_seqs * gpus}")
print(f"vLLM gpu_memory_utilization   : {gpu_util}")
print(f"nominal vLLM budget per GPU   : {80 * gpu_util:.1f} GB")
print(f"worst KV cache per sequence   : {sequence_gib:.3f} GiB")
print(f"worst active KV cache per GPU : {kv_gib_per_gpu:.2f} GiB")
print("sampling                      : temperature=0.7, top_p=0.8, top_k=20")
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
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=1 \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples=-1 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
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
    trainer.project_name=qwen3_1p7b_gsm8k_eval \
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

from verl.utils.reward_score.gsm8k import extract_solution

generation_file = os.environ["GENERATION_FILE"]
result_dir = os.environ["RESULT_DIR"]
max_response_length = int(os.environ["MAX_RESPONSE_LENGTH"])

with open(generation_file, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle]

groups = defaultdict(list)
for row in rows:
    groups[row["input"]].append(float(row["score"]) > 0.5)

sample_counts = Counter(map(len, groups.values()))
if len(groups) != 1319 or sample_counts != {4: 1319}:
    raise RuntimeError(f"Expected 1319 prompts x 4 samples, got {len(groups)} prompts and {sample_counts}")

total_correct = sum(sum(values) for values in groups.values())
pass1 = total_correct / len(rows)
first_draw_pass1 = sum(values[0] for values in groups.values()) / len(groups)
pass4 = sum(any(values) for values in groups.values()) / len(groups)

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
lengths = np.array(
    [len(tokenizer.encode(row["output"], add_special_tokens=False)) for row in rows],
    dtype=np.int64,
)
unextracted = sum(extract_solution(row["output"], method="boxed") is None for row in rows)

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

print("========== GSM8K RESULTS ==========")
for key, value in summary.items():
    print(f"{key:24}: {value}")
print(f"results directory       : {result_dir}")
print("===================================")
PY
