#!/usr/bin/env bash
# Current OPSD-answer Qwen3-1.7B checkpoint comparison on AIME 2024.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}

export FENG_J
export ALGORITHM_ID=opsd_answer
export MODEL_ID=qwen3_1.7b
export TRAINING_RUN_ID=dapo_top16_8k_100step_20260713_021308
export BENCHMARK_ID=aime24
export BENCHMARK_FILE=${AIME_FILE:-$FENG_J/data/aime-2024/aime-2024-unique-30.parquet}
export MODEL_MANIFEST=${MODEL_MANIFEST:-$SCRIPT_DIR/configs/opsd_answer_qwen3_1_7b_dapo100.tsv}

export EXPECTED_PROMPTS=30
export VAL_N=32
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=8192
export TOKENIZER_PATH=Qwen/Qwen3-1.7B
export ENABLE_THINKING=True

export NGPUS_PER_NODE=6
export ROLLOUT_TP=1
export ROLLOUT_INTERNAL_DP=1
export ROLLOUT_GPU_MEM_UTIL=0.80
export MAX_NUM_SEQS=48
export MAX_NUM_BATCHED_TOKENS=32768
export AGENT_NUM_WORKERS=32

export VAL_TEMPERATURE=1.0
export VAL_TOP_P=0.7
export VAL_TOP_K=-1

bash "$SCRIPT_DIR/run_checkpoint_benchmark.sh" "$@"
