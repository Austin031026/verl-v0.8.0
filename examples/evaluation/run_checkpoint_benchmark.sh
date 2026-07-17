#!/usr/bin/env bash
# Run one benchmark against a manifest of checkpoints and update a shared registry.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
FENG_J=${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}
RESULTS_ROOT=${RESULTS_ROOT:-$FENG_J/benchmark_results}

: "${ALGORITHM_ID:?Set ALGORITHM_ID, for example opsd_answer}"
: "${MODEL_ID:?Set MODEL_ID, for example qwen3_1.7b}"
: "${TRAINING_RUN_ID:?Set TRAINING_RUN_ID}"
: "${BENCHMARK_ID:?Set BENCHMARK_ID, for example aime24}"
: "${BENCHMARK_FILE:?Set BENCHMARK_FILE to a verl parquet file}"
: "${MODEL_MANIFEST:?Set MODEL_MANIFEST to a tab-separated model manifest}"

EXPECTED_PROMPTS=${EXPECTED_PROMPTS:-30}
VAL_N=${VAL_N:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
TOKENIZER_PATH=${TOKENIZER_PATH:-Qwen/Qwen3-1.7B}
ENABLE_THINKING=${ENABLE_THINKING:-True}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_INTERNAL_DP=${ROLLOUT_INTERNAL_DP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.80}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-48}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-32}

VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.7}
VAL_TOP_K=${VAL_TOP_K:--1}
EVAL_ID=${EVAL_ID:-$(date +%Y%m%d_%H%M%S)}
CUSTOM_REWARD_FUNCTION_PATH=${CUSTOM_REWARD_FUNCTION_PATH:-}
CUSTOM_REWARD_FUNCTION_NAME=${CUSTOM_REWARD_FUNCTION_NAME:-compute_score}

for value in "$ALGORITHM_ID" "$MODEL_ID" "$TRAINING_RUN_ID" "$BENCHMARK_ID" "$EVAL_ID"; do
    if [[ ! "$value" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Identifiers may contain only letters, digits, dot, underscore, and hyphen: $value" >&2
        exit 2
    fi
done

if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "Benchmark file not found: $BENCHMARK_FILE" >&2
    exit 2
fi
if [ ! -f "$MODEL_MANIFEST" ]; then
    echo "Model manifest not found: $MODEL_MANIFEST" >&2
    exit 2
fi
if [ -n "$CUSTOM_REWARD_FUNCTION_PATH" ] && [ ! -f "$CUSTOM_REWARD_FUNCTION_PATH" ]; then
    echo "Custom reward function not found: $CUSTOM_REWARD_FUNCTION_PATH" >&2
    exit 2
fi
cd "$REPO"

GROUP_ROOT="$RESULTS_ROOT/$ALGORITHM_ID/$MODEL_ID"
EVAL_ROOT="$GROUP_ROOT/runs/$TRAINING_RUN_ID/$BENCHMARK_ID/$EVAL_ID"
REGISTRY_PATH="$GROUP_ROOT/benchmark_registry.json"
RESOLVED_MODELS="$EVAL_ROOT/resolved_models.tsv"
mkdir -p "$EVAL_ROOT/models"
cp "$MODEL_MANIFEST" "$EVAL_ROOT/model_manifest.tsv"
: > "$RESOLVED_MODELS"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$FENG_J/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$FENG_J/hf/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$FENG_J/hf/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$FENG_J/hf/datasets}
export HF_XET_CACHE=${HF_XET_CACHE:-$FENG_J/hf/xet}
export TMPDIR=/tmp
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_jf42bamu}
mkdir -p "$RAY_TMPDIR"

python3 - "$EVAL_ROOT/config_snapshot.json" <<PY
import json
import sys

config = {
    "algorithm_id": "$ALGORITHM_ID",
    "model_id": "$MODEL_ID",
    "training_run_id": "$TRAINING_RUN_ID",
    "benchmark_id": "$BENCHMARK_ID",
    "eval_id": "$EVAL_ID",
    "benchmark_file": "$BENCHMARK_FILE",
    "model_manifest": "$MODEL_MANIFEST",
    "expected_prompts": $EXPECTED_PROMPTS,
    "val_n": $VAL_N,
    "max_prompt_length": $MAX_PROMPT_LENGTH,
    "max_response_length": $MAX_RESPONSE_LENGTH,
    "max_model_len": $MAX_MODEL_LEN,
    "tokenizer_path": "$TOKENIZER_PATH",
    "enable_thinking": "$ENABLE_THINKING",
    "n_gpus_per_node": $NGPUS_PER_NODE,
    "rollout_tp": $ROLLOUT_TP,
    "rollout_internal_dp": $ROLLOUT_INTERNAL_DP,
    "rollout_gpu_memory_utilization": $ROLLOUT_GPU_MEM_UTIL,
    "max_num_seqs": $MAX_NUM_SEQS,
    "max_num_batched_tokens": $MAX_NUM_BATCHED_TOKENS,
    "agent_num_workers": $AGENT_NUM_WORKERS,
    "temperature": $VAL_TEMPERATURE,
    "top_p": $VAL_TOP_P,
    "top_k": $VAL_TOP_K,
    "custom_reward_function_path": "$CUSTOM_REWARD_FUNCTION_PATH",
    "custom_reward_function_name": "$CUSTOM_REWARD_FUNCTION_NAME",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

resolve_model() {
    local label=$1
    local backend=$2
    local source_path=$3
    local model_root=$4
    local resolved_path

    case "$backend" in
        hf)
            resolved_path=$source_path
            ;;
        fsdp|megatron)
            if [ ! -d "$source_path" ]; then
                echo "Checkpoint directory not found: $source_path" >&2
                return 1
            fi
            resolved_path="$(dirname "$source_path")/actor_huggingface"
            if [ ! -s "$resolved_path/config.json" ]; then
                mkdir -p "$resolved_path"
                python3 -m verl.model_merger merge \
                    --backend "$backend" \
                    --local_dir "$source_path" \
                    --target_dir "$resolved_path" \
                    2>&1 | tee "$model_root/merge.log" >&2
                local merge_status=${PIPESTATUS[0]}
                if [ "$merge_status" -ne 0 ]; then
                    return "$merge_status"
                fi
            fi
            ;;
        *)
            echo "Unsupported manifest backend '$backend' for model '$label'" >&2
            return 2
            ;;
    esac

    printf '%s\n' "$resolved_path"
}

overall_status=0
model_count=0
while IFS=$'\t' read -r label backend source_path extra; do
    if [ -z "${label// }" ] || [[ "$label" == \#* ]]; then
        continue
    fi
    if [ -n "${extra:-}" ] || [ -z "$backend" ] || [ -z "$source_path" ]; then
        echo "Invalid manifest row for '$label'; expected: label<TAB>backend<TAB>path" >&2
        overall_status=1
        continue
    fi
    if [[ ! "$label" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid model label: $label" >&2
        overall_status=1
        continue
    fi

    model_count=$((model_count + 1))
    MODEL_ROOT="$EVAL_ROOT/models/$label"
    mkdir -p "$MODEL_ROOT/validation"

    resolved_path=$(resolve_model "$label" "$backend" "$source_path" "$MODEL_ROOT")
    resolve_status=$?
    if [ "$resolve_status" -ne 0 ]; then
        echo "$resolve_status" > "$MODEL_ROOT/exit_status.txt"
        printf '%s\t%s\t%s\t\n' "$label" "$backend" "$source_path" >> "$RESOLVED_MODELS"
        overall_status=1
        continue
    fi
    printf '%s\t%s\t%s\t%s\n' "$label" "$backend" "$source_path" "$resolved_path" >> "$RESOLVED_MODELS"

    ray stop --force > "$MODEL_ROOT/ray_cleanup_before.log" 2>&1 || true

    EXTRA_DATA_ARGS=()
    if [ "$ENABLE_THINKING" != "unset" ]; then
        EXTRA_DATA_ARGS+=("+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING")
    fi
    EXTRA_REWARD_ARGS=()
    if [ -n "$CUSTOM_REWARD_FUNCTION_PATH" ]; then
        EXTRA_REWARD_ARGS+=(
            "reward.custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH"
            "reward.custom_reward_function.name=$CUSTOM_REWARD_FUNCTION_NAME"
        )
    fi

    set +e
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="$BENCHMARK_FILE" \
        data.val_files="$BENCHMARK_FILE" \
        data.train_batch_size=1 \
        data.val_batch_size=null \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.filter_overlong_prompts=False \
        data.validation_shuffle=False \
        data.dataloader_num_workers=0 \
        data.truncation=error \
        actor_rollout_ref.model.path="$resolved_path" \
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
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_INTERNAL_DP" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
        actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
        actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.ignore_eos=False \
        actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
        actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
        actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
        actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
        trainer.logger='["console"]' \
        trainer.project_name="benchmark_$ALGORITHM_ID" \
        trainer.experiment_name="${MODEL_ID}_${BENCHMARK_ID}_${label}_${EVAL_ID}" \
        trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.resume_mode=disable \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.validation_data_dir="$MODEL_ROOT/validation" \
        trainer.default_local_dir="$MODEL_ROOT/trainer_state" \
        +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
        "${EXTRA_DATA_ARGS[@]}" \
        "${EXTRA_REWARD_ARGS[@]}" \
        2>&1 | tee "$MODEL_ROOT/eval.log"
    eval_status=${PIPESTATUS[0]}
    echo "$eval_status" > "$MODEL_ROOT/exit_status.txt"
    if [ "$eval_status" -ne 0 ]; then
        overall_status=1
    fi

    ray stop --force > "$MODEL_ROOT/ray_cleanup_after.log" 2>&1 || true
done < "$MODEL_MANIFEST"

if [ "$model_count" -eq 0 ]; then
    echo "No model rows found in $MODEL_MANIFEST" >&2
    exit 2
fi

python3 "$SCRIPT_DIR/summarize_checkpoint_benchmark.py" \
    --algorithm-id "$ALGORITHM_ID" \
    --model-id "$MODEL_ID" \
    --training-run-id "$TRAINING_RUN_ID" \
    --benchmark-id "$BENCHMARK_ID" \
    --eval-id "$EVAL_ID" \
    --eval-root "$EVAL_ROOT" \
    --registry-path "$REGISTRY_PATH" \
    --resolved-models "$RESOLVED_MODELS" \
    --config-snapshot "$EVAL_ROOT/config_snapshot.json" \
    --expected-prompts "$EXPECTED_PROMPTS" \
    --samples-per-prompt "$VAL_N" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --tokenizer-path "$TOKENIZER_PATH"
summary_status=$?
if [ "$summary_status" -ne 0 ]; then
    overall_status=1
fi

echo "EVAL_ROOT=$EVAL_ROOT"
echo "SUMMARY_JSON=$EVAL_ROOT/summary.json"
echo "REGISTRY_JSON=$REGISTRY_PATH"
echo "OVERALL_EXIT_STATUS=$overall_status"
exit "$overall_status"
