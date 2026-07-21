#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <checkpoint_step: 200|400|600|800|1000|1200|1250>" >&2
    exit 2
fi

CHECKPOINT_STEP=$1
case "$CHECKPOINT_STEP" in
    200|400|600|800|1000|1200|1250) ;;
    *)
        echo "Unsupported checkpoint step: $CHECKPOINT_STEP" >&2
        echo "Expected one of: 200, 400, 600, 800, 1000, 1200, 1250" >&2
        exit 2
        ;;
esac

FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENERATION_RUNNER=$(cd "$SCRIPT_DIR/../../../../evaluation" && pwd)/run_checkpoint_generation.sh

# Model and experiment identity. Only CHECKPOINT_STEP changes between evaluations.
export ALGORITHM_ID=opsd_reason
export MODEL_ID=qwen3_1.7b
export TRAINING_RUN_ID=qwen3_1p7b_openthought_math_reason_4k_reverse_kl_b8_n4_full_vocab
export MODEL_LABEL="step${CHECKPOINT_STEP}"
export MODEL_BACKEND=fsdp
export MODEL_PATH="$FENG_J/checkpoints/$TRAINING_RUN_ID/global_step_${CHECKPOINT_STEP}/actor"

# Dataset and offline-rescoring contract.
export BENCHMARK_ID=math500
export BENCHMARK_FILE="$FENG_J/data/MATH-500/test_verl.parquet"
export EXPECTED_PROMPTS=500
export PROMPT_KEY=prompt
export GROUND_TRUTH_FIELD=reward_model.ground_truth

# Generation protocol: 250 prompts x 4 samples = 1000 requests per loader batch.
export N_SAMPLES=4
export GENERATION_BATCH_SIZE=250
export GENERATION_DO_SAMPLE=True
export ENABLE_THINKING=False
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export EXPECTED_RESPONSE_LENGTH=1024
export MAX_MODEL_LEN=6145
export GENERATION_TEMPERATURE=0.7
export GENERATION_TOP_P=0.8
export GENERATION_TOP_K=20

# Four TP=1 replicas, with at most 128 resident sequences requested per replica.
export NNODES=1
export NGPUS_PER_NODE=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.90
export MAX_NUM_SEQS=128
export MAX_NUM_BATCHED_TOKENS=32768
export PHYSICAL_VRAM_GIB=80
export RUNTIME_RESERVE_GIB=10

export EVAL_ID="${MODEL_LABEL}_n${N_SAMPLES}_$(date +%Y%m%d_%H%M%S)"

bash "$GENERATION_RUNNER"
