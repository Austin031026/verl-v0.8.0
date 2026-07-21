#!/usr/bin/env bash
# Scorer-free checkpoint generation: HF/verl weights + parquet prompts -> validation/0.jsonl.

set -euo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}
CONDA_ENV=${CONDA_ENV:-$FENG_J/conda/envs/verl_v080_official_script}
PYTHON=${PYTHON:-$CONDA_ENV/bin/python}
RAY_BIN=${RAY_BIN:-$CONDA_ENV/bin/ray}
RESULTS_ROOT=${RESULTS_ROOT:-$FENG_J/benchmark_results}

: "${ALGORITHM_ID:?Set ALGORITHM_ID}"
: "${MODEL_ID:?Set MODEL_ID}"
: "${TRAINING_RUN_ID:?Set TRAINING_RUN_ID}"
: "${MODEL_LABEL:?Set MODEL_LABEL}"
: "${MODEL_PATH:?Set MODEL_PATH to Hugging Face weights or a verl checkpoint}"
: "${BENCHMARK_ID:?Set BENCHMARK_ID}"
: "${BENCHMARK_FILE:?Set BENCHMARK_FILE to a parquet file}"
: "${EXPECTED_PROMPTS:?Set EXPECTED_PROMPTS}"
: "${N_SAMPLES:?Set N_SAMPLES}"
: "${GENERATION_BATCH_SIZE:?Set GENERATION_BATCH_SIZE}"
: "${ENABLE_THINKING:?Set ENABLE_THINKING=True or False}"
: "${MAX_PROMPT_LENGTH:?Set MAX_PROMPT_LENGTH}"
: "${MAX_RESPONSE_LENGTH:?Set MAX_RESPONSE_LENGTH}"
: "${GENERATION_TEMPERATURE:?Set GENERATION_TEMPERATURE}"
: "${GENERATION_TOP_P:?Set GENERATION_TOP_P}"
: "${GENERATION_TOP_K:?Set GENERATION_TOP_K}"
: "${NGPUS_PER_NODE:?Set NGPUS_PER_NODE}"
: "${ROLLOUT_TP:?Set ROLLOUT_TP}"
: "${ROLLOUT_GPU_MEM_UTIL:?Set ROLLOUT_GPU_MEM_UTIL}"
: "${MAX_NUM_SEQS:?Set MAX_NUM_SEQS}"
: "${MAX_NUM_BATCHED_TOKENS:?Set MAX_NUM_BATCHED_TOKENS}"

NNODES=${NNODES:-1}
PROMPT_KEY=${PROMPT_KEY:-prompt}
RESPONSES_KEY=${RESPONSES_KEY:-responses}
GROUND_TRUTH_FIELD=${GROUND_TRUTH_FIELD:-reward_model.ground_truth}
GENERATION_DO_SAMPLE=${GENERATION_DO_SAMPLE:-True}
EXPECTED_RESPONSE_LENGTH=${EXPECTED_RESPONSE_LENGTH:-$MAX_RESPONSE_LENGTH}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
PHYSICAL_VRAM_GIB=${PHYSICAL_VRAM_GIB:-80}
RUNTIME_RESERVE_GIB=${RUNTIME_RESERVE_GIB:-10}
EVAL_ID=${EVAL_ID:-${MODEL_LABEL}_n${N_SAMPLES}_$(date +%Y%m%d_%H%M%S)}
MODEL_BACKEND=${MODEL_BACKEND:-hf}

for value in "$ALGORITHM_ID" "$MODEL_ID" "$TRAINING_RUN_ID" "$MODEL_LABEL" "$BENCHMARK_ID" "$EVAL_ID"; do
    if [[ ! "$value" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Identifiers may contain only letters, digits, dot, underscore, and hyphen: $value" >&2
        exit 2
    fi
done

case "${GENERATION_DO_SAMPLE,,}" in
    true|1|yes|on) ;;
    false|0|no|off)
        if [[ "$N_SAMPLES" -ne 1 ]]; then
            echo "Greedy generation requires N_SAMPLES=1" >&2
            exit 2
        fi
        GENERATION_TEMPERATURE=0.0
        ;;
    *)
        echo "GENERATION_DO_SAMPLE must be True or False" >&2
        exit 2
        ;;
esac

for required_dir in "$REPO" "$CONDA_ENV"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Missing required directory: $required_dir" >&2
        exit 2
    fi
done

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python executable: $PYTHON" >&2
    exit 2
fi

case "$MODEL_BACKEND" in
    hf) ;;
    fsdp|megatron)
        if [[ ! -d "$MODEL_PATH" ]]; then
            echo "Missing checkpoint directory: $MODEL_PATH" >&2
            exit 2
        fi
        MERGED_MODEL_PATH="$(dirname "$MODEL_PATH")/actor_huggingface"
        if [[ ! -s "$MERGED_MODEL_PATH/config.json" ]]; then
            mkdir -p "$MERGED_MODEL_PATH"
            (
                cd "$REPO"
                "$PYTHON" -m verl.model_merger merge \
                    --backend "$MODEL_BACKEND" \
                    --local_dir "$MODEL_PATH" \
                    --target_dir "$MERGED_MODEL_PATH"
            )
        fi
        MODEL_PATH="$MERGED_MODEL_PATH"
        ;;
    *)
        echo "Unsupported MODEL_BACKEND: $MODEL_BACKEND" >&2
        exit 2
        ;;
esac

for required_file in "$MODEL_PATH/config.json" "$BENCHMARK_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 2
    fi
done

EVAL_ROOT="$RESULTS_ROOT/$ALGORITHM_ID/$MODEL_ID/runs/$TRAINING_RUN_ID/$BENCHMARK_ID/$EVAL_ID"
MODEL_ROOT="$EVAL_ROOT/models/$MODEL_LABEL"
GENERATION_PARQUET="$MODEL_ROOT/generation.parquet"
GENERATION_JSONL="$MODEL_ROOT/validation/0.jsonl"
PREFLIGHT_JSON="$EVAL_ROOT/config_snapshot.json"
mkdir -p "$MODEL_ROOT/validation"

export HF_HOME=${HF_HOME:-$FENG_J/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$FENG_J/hf/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$FENG_J/hf/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$FENG_J/hf/datasets}
export HF_XET_CACHE=${HF_XET_CACHE:-$FENG_J/hf/xet}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_jf42bamu
mkdir -p "$RAY_TMPDIR"

"$PYTHON" "$SCRIPT_DIR/checkpoint_generation_preflight.py" \
    --model-path "$MODEL_PATH" \
    --data-path "$BENCHMARK_FILE" \
    --prompt-key "$PROMPT_KEY" \
    --ground-truth-field "$GROUND_TRUTH_FIELD" \
    --expected-prompts "$EXPECTED_PROMPTS" \
    --samples-per-prompt "$N_SAMPLES" \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --enable-thinking "$ENABLE_THINKING" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --expected-response-length "$EXPECTED_RESPONSE_LENGTH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --num-gpus "$((NNODES * NGPUS_PER_NODE))" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEM_UTIL" \
    --physical-vram-gib "$PHYSICAL_VRAM_GIB" \
    --runtime-reserve-gib "$RUNTIME_RESERVE_GIB" \
    --output-json "$PREFLIGHT_JSON" \
    2>&1 | tee "$MODEL_ROOT/preflight.log"

cleanup() {
    if [[ -x "$RAY_BIN" ]]; then
        "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
cleanup

cd "$REPO"
set +e
"$PYTHON" -m verl.trainer.main_generation_server \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    data.train_files="$BENCHMARK_FILE" \
    data.prompt_key="$PROMPT_KEY" \
    +data.output_path="$GENERATION_PARQUET" \
    +data.generation_batch_size="$GENERATION_BATCH_SIZE" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="$GENERATION_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$GENERATION_TOP_P" \
    actor_rollout_ref.rollout.top_k="$GENERATION_TOP_K" \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n="$N_SAMPLES" \
    2>&1 | tee "$MODEL_ROOT/generation.log"
generation_status=${PIPESTATUS[0]}
set -e
if [[ "$generation_status" -ne 0 ]]; then
    echo "$generation_status" > "$MODEL_ROOT/exit_status.txt"
    exit "$generation_status"
fi

"$PYTHON" "$SCRIPT_DIR/export_generation_jsonl.py" \
    --input-parquet "$GENERATION_PARQUET" \
    --output-jsonl "$GENERATION_JSONL" \
    --benchmark-id "$BENCHMARK_ID" \
    --prompt-key "$PROMPT_KEY" \
    --responses-key "$RESPONSES_KEY" \
    --ground-truth-field "$GROUND_TRUTH_FIELD" \
    --expected-prompts "$EXPECTED_PROMPTS" \
    --samples-per-prompt "$N_SAMPLES" \
    2>&1 | tee "$MODEL_ROOT/export.log"

echo 0 > "$MODEL_ROOT/exit_status.txt"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "GENERATION_PARQUET=$GENERATION_PARQUET"
echo "GENERATION_JSONL=$GENERATION_JSONL"
