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
        exit 2
        ;;
esac

FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_RUNNER=$(cd "$SCRIPT_DIR/../../../../evaluation" && pwd)/run_checkpoint_benchmark.sh
CONDA_SH="$FENG_J/conda/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="$FENG_J/conda/envs/verl_v080_official_script"

export ALGORITHM_ID=opsd_reason_official_solution
export MODEL_ID=qwen3_1.7b
export TRAINING_RUN_ID=qwen3_1p7b_openthought_math_official_solution_reason_4k_reverse_kl_b8_n4_full_vocab
export BENCHMARK_ID=math500
export BENCHMARK_FILE="$FENG_J/data/MATH-500/test_verl.parquet"
export EXPECTED_PROMPTS=500

CHECKPOINT_DIR="$FENG_J/checkpoints/$TRAINING_RUN_ID/global_step_${CHECKPOINT_STEP}/actor"
for rank in 0 1 2 3; do
    for kind in model optim extra_state; do
        file="$CHECKPOINT_DIR/${kind}_world_size_4_rank_${rank}.pt"
        if [[ ! -s "$file" ]]; then
            echo "Missing or empty checkpoint shard: $file" >&2
            exit 2
        fi
    done
done

if [[ ! -f "$BENCHMARK_FILE" ]]; then
    echo "Missing MATH-500 parquet: $BENCHMARK_FILE" >&2
    exit 2
fi
if [[ ! -f "$BENCHMARK_RUNNER" ]]; then
    echo "Missing benchmark runner: $BENCHMARK_RUNNER" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" || ! -d "$CONDA_ENV" ]]; then
    echo "Missing configured conda environment" >&2
    exit 2
fi

source "$CONDA_SH"
export CONDARC="$FENG_J/conda/.condarc"
conda activate "$CONDA_ENV"

export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_jf42bamu
mkdir -p "$RAY_TMPDIR"

MODEL_MANIFEST=$(mktemp "$RAY_TMPDIR/math500_step${CHECKPOINT_STEP}.XXXXXX.tsv")
trap 'rm -f "$MODEL_MANIFEST"' EXIT
printf 'step%s\tfsdp\t%s\n' "$CHECKPOINT_STEP" "$CHECKPOINT_DIR" > "$MODEL_MANIFEST"
export MODEL_MANIFEST

# Same MATH-500 generation protocol used for the conversations-reason experiment.
export VAL_N=4
export VAL_BATCH_SIZE=250
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export MAX_MODEL_LEN=6145
export TOKENIZER_PATH=Qwen/Qwen3-1.7B
export ENABLE_THINKING=False

export NGPUS_PER_NODE=6
export ROLLOUT_TP=1
export ROLLOUT_INTERNAL_DP=1
export ROLLOUT_GPU_MEM_UTIL=0.90
export MAX_NUM_SEQS=128
export MAX_NUM_BATCHED_TOKENS=32768
export AGENT_NUM_WORKERS=32

export VAL_DO_SAMPLE=True
export VAL_TEMPERATURE=0.7
export VAL_TOP_P=0.8
export VAL_TOP_K=20
export EVAL_ID="step${CHECKPOINT_STEP}_n${VAL_N}_$(date +%Y%m%d_%H%M%S)"

printf '%s\n' \
    '===== MATH-500 validation preflight =====' \
    "checkpoint=$CHECKPOINT_DIR" \
    "max_prompt_length=$MAX_PROMPT_LENGTH" \
    "max_response_length=$MAX_RESPONSE_LENGTH" \
    "val_batch_size=$VAL_BATCH_SIZE prompts" \
    "samples_per_prompt=$VAL_N" \
    "queued_requests_per_batch=$((VAL_BATCH_SIZE * VAL_N))" \
    "max_num_seqs_per_replica=$MAX_NUM_SEQS" \
    "replicas=$((NGPUS_PER_NODE / ROLLOUT_TP))" \
    "configured_cluster_active_cap=$((MAX_NUM_SEQS * NGPUS_PER_NODE / ROLLOUT_TP))" \
    "gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL" \
    '========================================='

bash "$BENCHMARK_RUNNER"
