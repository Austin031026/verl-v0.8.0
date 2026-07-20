#!/usr/bin/env bash
set -euo pipefail
umask 027

FENG_J=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J

BUNDLE_DIR="$FENG_J/opsd_training/Qwen3-1.7B/qwen3_1p7b_openthought_math_reason_4k_reverse_kl_b8_n4_full_vocab"
VERL_ROOT="$FENG_J/verl-v0.8.0-opsd-test"

CONDA_SH="$FENG_J/conda/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="$FENG_J/conda/envs/verl_v080_official_script"

export HF_HOME="$FENG_J/hf"
export HF_HUB_CACHE="$FENG_J/hf/hub"
export HUGGINGFACE_HUB_CACHE="$FENG_J/hf/hub"
export HF_DATASETS_CACHE="$FENG_J/hf/datasets"
export TRANSFORMERS_CACHE="$FENG_J/hf/hub"
export HF_XET_CACHE="$FENG_J/hf/xet"

MODEL_ID=Qwen/Qwen3-1.7B
PYTHON="$CONDA_ENV/bin/python"

STUDENT_MODEL="$("$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="Qwen/Qwen3-1.7B",
        local_files_only=True,
    )
)
PY
)"
TEACHER_MODEL="$STUDENT_MODEL"

TRAIN_FILE="$FENG_J/data/Openthought-math-open-ri/train_10000_seed42_verl_opsd.parquet"
CHECKPOINT_DIR="$FENG_J/checkpoints/qwen3_1p7b_openthought_math_reason_4k_reverse_kl_b8_n4_full_vocab"

# ---------- 实验配置 ----------

EXPERIMENT_NAME=qwen3_1p7b_openthought_math_reason_4k_reverse_kl_b8_n4_full_vocab
TRAIN_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
ENABLE_THINKING=False

ROLLOUT_N=4
ROLLOUT_DO_SAMPLE=True
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.8
ROLLOUT_TOP_K=20

OPSD_PRIVILEGED_INPUT_MODE=reason
OPSD_KL_MODE=reverse_kl
OPSD_RL_COUPLING=none
OPSD_VOCAB_STRATEGY=full

RAY_TMPDIR=/tmp/ray_jf42bamu

# ---------- 启动前检查 ----------

for required_dir in \
    "$VERL_ROOT" \
    "$CONDA_ENV" \
    "$STUDENT_MODEL" \
    "$TEACHER_MODEL"
do
    if [[ ! -d "$required_dir" ]]; then
        echo "Missing required directory: $required_dir" >&2
        exit 2
    fi
done

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda activation script: $CONDA_SH" >&2
    exit 2
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Missing training file: $TRAIN_FILE" >&2
    exit 2
fi

# ---------- 运行环境 ----------

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$VERL_ROOT"

export PIP_CACHE_DIR="$FENG_J/pip"

export TMPDIR=/tmp
export RAY_TMPDIR
export PYTHONUNBUFFERED=1

mkdir -p "$RAY_TMPDIR"
mkdir -p "$BUNDLE_DIR/runs"
mkdir -p "$CHECKPOINT_DIR"

# ---------- 本次运行日志目录 ----------

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="$BUNDLE_DIR/runs/$RUN_ID"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$BUNDLE_DIR/runs/latest"

TRAIN_LOG="$RUN_DIR/train.log"
METRICS_JSONL="$RUN_DIR/metrics.jsonl"
LOSS_JSONL="$RUN_DIR/loss_metrics.jsonl"
STATUS_JSON="$RUN_DIR/status.json"

export VERL_FILE_LOGGER_PATH="$METRICS_JSONL"

# ---------- 退出时提取 loss ----------

finalize() {
    exit_code=$?
    set +e

    python - "$METRICS_JSONL" "$LOSS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with destination.open("w", encoding="utf-8") as output:
    if source.exists():
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue

            record = json.loads(raw_line)
            losses = {
                key: value
                for key, value in record.get("data", {}).items()
                if "loss" in key.lower()
            }

            if losses:
                output.write(
                    json.dumps(
                        {
                            "step": record.get("step"),
                            "losses": losses,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
PY

    printf '{"exit_code":%d,"finished_at_utc":"%s"}\n' \
        "$exit_code" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$STATUS_JSON"

    trap - EXIT
    exit "$exit_code"
}

trap finalize EXIT

echo "Run directory:        $RUN_DIR"
echo "Full training log:    $TRAIN_LOG"
echo "Structured metrics:   $METRICS_JSONL"
echo "Loss-only metrics:    $LOSS_JSONL"
echo "Checkpoint directory: $CHECKPOINT_DIR"

# ---------- OPSD 训练 ----------

{
printf '%s\n' \
    '===== OPSD experiment configuration =====' \
    "experiment_name=$EXPERIMENT_NAME" \
    "train_batch_size=$TRAIN_BATCH_SIZE" \
    "rollout.n=$ROLLOUT_N" \
    "trajectories_per_update=$((TRAIN_BATCH_SIZE * ROLLOUT_N))" \
    "rollout.do_sample=$ROLLOUT_DO_SAMPLE" \
    "rollout.temperature=$ROLLOUT_TEMPERATURE" \
    "rollout.top_p=$ROLLOUT_TOP_P" \
    "rollout.top_k=$ROLLOUT_TOP_K" \
    "student.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "student.max_response_length=$MAX_RESPONSE_LENGTH" \
    "student.enable_thinking=$ENABLE_THINKING" \
    "opsd.privileged_input.mode=$OPSD_PRIVILEGED_INPUT_MODE" \
    "opsd.kl_mode=$OPSD_KL_MODE" \
    "opsd.rl_coupling=$OPSD_RL_COUPLING" \
    "opsd.vocab_strategy=$OPSD_VOCAB_STRATEGY" \
    '========================================='

env \
    STUDENT_MODEL="$STUDENT_MODEL" \
    OPSD_TEACHER_MODEL="$TEACHER_MODEL" \
    OPSD_TEACHER_PRIVILEGED_INPUT_MODE="$OPSD_PRIVILEGED_INPUT_MODE" \
    NNODES=1 \
    NGPUS_PER_NODE=4 \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    PPO_MINI_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    ENABLE_THINKING="$ENABLE_THINKING" \
    ACTOR_LR=1e-6 \
    ROLLOUT_TP=1 \
    ROLLOUT_GPU_MEM_UTIL=0.60 \
    OPSD_KL_MODE="$OPSD_KL_MODE" \
    OPSD_RL_COUPLING="$OPSD_RL_COUPLING" \
    OPSD_VOCAB_STRATEGY="$OPSD_VOCAB_STRATEGY" \
    OPSD_LOSS_COEF=1.0 \
    OPSD_TEMPERATURE=1.0 \
    OPSD_TEACHER_MAX_PROMPT_LENGTH=12288 \
    OPSD_TEACHER_MAX_CONTEXT_NO_THINK=16384 \
    OPSD_TEACHER_MAX_CONTEXT_THINKING=16384 \
    OPSD_TEACHER_MAX_TOKEN_LEN_PER_GPU=65536 \
    TOTAL_TRAINING_STEPS=1250 \
    TOTAL_EPOCHS=1 \
    SAVE_FREQ=200 \
    TEST_FREQ=-1 \
    LOGGER='["console","file"]' \
    PROJECT_NAME=verl_opsd \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    TRAIN_FILES="$TRAIN_FILE" \
    VAL_FILES="$TRAIN_FILE" \
    bash examples/on_policy_distillation_trainer/run_qwen3_2b_opsd_fsdp.sh \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    opsd.teacher.model_path="$TEACHER_MODEL" \
    actor_rollout_ref.rollout.do_sample="$ROLLOUT_DO_SAMPLE" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.skip.enable=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    opsd.teacher.use_dynamic_bsz=True \
    opsd.teacher.micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    reward.custom_reward_function.path=null \
    data.seed=42 \
    data.val_max_samples=1 \
    data.val_batch_size=1 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode=disable \
    trainer.resume_from_path=null \
    trainer.val_only=False \
    trainer.val_before_train=False
} 2>&1 | tee "$TRAIN_LOG"
