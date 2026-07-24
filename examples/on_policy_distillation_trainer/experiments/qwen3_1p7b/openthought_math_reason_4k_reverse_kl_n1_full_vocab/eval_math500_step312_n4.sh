#!/usr/bin/env bash
set -euo pipefail

FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENERATION_RUNNER=$(cd "$SCRIPT_DIR/../../../../evaluation" && pwd)/run_checkpoint_generation.sh

# Model and experiment identity.
export ALGORITHM_ID=opsd_reason
export MODEL_ID=qwen3_1.7b
export TRAINING_RUN_ID=qwen3_1p7b_openthought_math_reason_4k
export MODEL_LABEL=step312
export MODEL_PATH="$FENG_J/checkpoints/qwen3_1p7b_openthought_math_reason_4k/global_step_312/actor_huggingface"

# Dataset and offline-rescoring contract.
export BENCHMARK_ID=math500
export BENCHMARK_FILE="$FENG_J/data/MATH-500/test_verl.parquet"
export EXPECTED_PROMPTS=500
export PROMPT_KEY=prompt
export GROUND_TRUTH_FIELD=reward_model.ground_truth

# Generation protocol.
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

# Standalone vLLM resources.
export NNODES=1
export NGPUS_PER_NODE=6
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.90
export MAX_NUM_SEQS=128
export MAX_NUM_BATCHED_TOKENS=32768
export PHYSICAL_VRAM_GIB=80
export RUNTIME_RESERVE_GIB=10

export EVAL_ID="${MODEL_LABEL}_n${N_SAMPLES}_$(date +%Y%m%d_%H%M%S)"

bash "$GENERATION_RUNNER"
