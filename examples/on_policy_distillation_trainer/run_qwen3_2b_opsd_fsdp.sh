#!/usr/bin/env bash
# OPSD | text | vLLM rollout | FSDP training | NVIDIA GPUs
#
# This is an experimental entrypoint for a local OPSD branch. It follows the
# existing on_policy_distillation_trainer script style, but routes OPSD-specific
# settings through +opsd.* instead of the original distillation teacher-server path.

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}
OPSD_TEACHER_MODEL=${OPSD_TEACHER_MODEL:-$STUDENT_MODEL}
opsd_teacher_privileged_input_mode=${OPSD_TEACHER_PRIVILEGED_INPUT_MODE:-answer}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
# Strict OPSD consumes each rollout batch in one actor update epoch.
ppo_epochs=${PPO_EPOCHS:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}

opsd_loss_mode=${OPSD_LOSS_MODE:-full_vocab_kl}
opsd_kl_mode=${OPSD_KL_MODE:-reverse_kl}
opsd_rl_coupling=${OPSD_RL_COUPLING:-none}
opsd_topk_strategy=${OPSD_TOPK_STRATEGY:-full}
opsd_student_topk=${OPSD_STUDENT_TOPK:-8}
opsd_teacher_topk=${OPSD_TEACHER_TOPK:-8}
if [[ -n "${OPSD_USE_TAIL:-}" ]]; then
    opsd_use_tail=${OPSD_USE_TAIL}
elif [[ "${opsd_topk_strategy}" == "full" ]]; then
    opsd_use_tail=False
else
    opsd_use_tail=True
fi
opsd_loss_coef=${OPSD_LOSS_COEF:-1.0}
opsd_temperature=${OPSD_TEMPERATURE:-1.0}

opsd_teacher_update_mode=${OPSD_TEACHER_UPDATE_MODE:-none}
opsd_teacher_update_interval=${OPSD_TEACHER_UPDATE_INTERVAL:-1}
opsd_teacher_update_ema_decay=${OPSD_TEACHER_UPDATE_EMA_DECAY:-0.999}

# Teacher forward uses dynamic token-budgeted micro-batches by default.
opsd_teacher_use_dynamic_bsz=${OPSD_TEACHER_USE_DYNAMIC_BSZ:-True}
# Teacher forward token budget per GPU; default matches actor log-prob forward.
opsd_teacher_max_token_len_per_gpu=${OPSD_TEACHER_MAX_TOKEN_LEN_PER_GPU:-$ppo_max_token_len_per_gpu}
# Teacher fixed micro-batch size per GPU; only used when dynamic batching is disabled.
opsd_teacher_micro_batch_size_per_gpu=${OPSD_TEACHER_MICRO_BATCH_SIZE_PER_GPU:-null}
# Teacher removes padding during forward to match the actor model input path.
opsd_teacher_use_remove_padding=${OPSD_TEACHER_USE_REMOVE_PADDING:-True}
# Keep fused kernels off for the first OPSD logits/topK/tail implementation.
opsd_teacher_use_fused_kernels=${OPSD_TEACHER_USE_FUSED_KERNELS:-False}
# Keep teacher parameters on GPU by default for the first OPSD implementation.
opsd_teacher_param_offload=${OPSD_TEACHER_PARAM_OFFLOAD:-False}
# Teacher is forward-only, so optimizer offload is kept disabled explicitly.
opsd_teacher_optimizer_offload=${OPSD_TEACHER_OPTIMIZER_OFFLOAD:-False}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

logger=${LOGGER:-'["console"]'}
project_name=${PROJECT_NAME:-verl_opsd}
experiment_name=${EXPERIMENT_NAME:-qwen3_2b_opsd_vllm_fsdp}
# ---- end user-adjustable ----

train_files=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
val_files=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# OPSD topK compression strategy:
#   full    - compare over the full vocabulary; first debug path, no tail bucket needed
#   union   - compare over topK(student) union topK(teacher), plus optional tail
#   teacher - compare over teacher topK only, plus optional tail
#   student - compare over student topK only, plus optional tail
case "${opsd_topk_strategy}" in
    full|union|teacher|student)
        ;;
    *)
        echo "Invalid OPSD_TOPK_STRATEGY=${opsd_topk_strategy}. Expected one of: full, union, teacher, student." >&2
        exit 1
        ;;
esac

# OPSD token-level KL direction:
#   forward_kl    - KL(teacher || student), first implementation target
#   reverse_kl    - KL(student || teacher), reserved for comparison
#   dynamic_merge - reserved for dynamically mixing forward/reverse KL variants
case "${opsd_kl_mode}" in
    forward_kl|reverse_kl|dynamic_merge)
        ;;
    *)
        echo "Invalid OPSD_KL_MODE=${opsd_kl_mode}. Expected one of: forward_kl, reverse_kl, dynamic_merge." >&2
        exit 1
        ;;
esac

# OPSD/RL objective coupling mode:
#   none - pure OPSD loss, no PPO/GRPO policy-gradient loss
#   grpo - reserved for future GRPO + OPSD coupled objective
case "${opsd_rl_coupling}" in
    none|grpo)
        ;;
    *)
        echo "Invalid OPSD_RL_COUPLING=${opsd_rl_coupling}. Expected one of: none, grpo." >&2
        exit 1
        ;;
esac

# OPSD teacher parameter update strategy:
#   none - keep teacher fixed after initialization
#   copy - reserved for hard sync: teacher <- actor
#   ema  - reserved for EMA sync: teacher <- decay * teacher + (1 - decay) * actor
case "${opsd_teacher_update_mode}" in
    none|copy|ema)
        ;;
    *)
        echo "Invalid OPSD_TEACHER_UPDATE_MODE=${opsd_teacher_update_mode}. Expected one of: none, copy, ema." >&2
        exit 1
        ;;
esac

if (( opsd_teacher_update_interval <= 0 )); then
    echo "OPSD_TEACHER_UPDATE_INTERVAL must be greater than 0." >&2
    exit 1
fi

# OPSD teacher privileged input source:
#   answer        - use the current sample's final answer/ground truth; first implemented path
#   answer_reason - reserved for answer plus reasoning/rationale
#   cot_examples  - reserved for shared or retrieved CoT demonstrations
case "${opsd_teacher_privileged_input_mode}" in
    answer|answer_reason|cot_examples)
        ;;
    *)
        echo "Invalid OPSD_TEACHER_PRIVILEGED_INPUT_MODE=${opsd_teacher_privileged_input_mode}. Expected one of: answer, answer_reason, cot_examples." >&2
        exit 1
        ;;
esac
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    # Strict OPSD does not reuse one rollout batch for multiple actor epochs.
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="$logger"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

OPSD=(
    # Disable the original OPD teacher-server path. The OPSD teacher is created
    # inside ActorRolloutRefWorker when +opsd.enabled=True.
    distillation.enabled=False
    +opsd.enabled=True

    # Teacher model initialization. The first OPSD implementation keeps the
    # teacher in the actor worker group and builds it as a forward-only FSDP model.
    +opsd.teacher.model_path="$OPSD_TEACHER_MODEL"
    +opsd.teacher.share_actor_worker=True
    # Controls how teacher-only privileged information is added before teacher forward.
    # Only answer is implemented first; answer_reason/cot_examples are reserved hooks.
    +opsd.teacher.privileged_input.mode=${opsd_teacher_privileged_input_mode}
    # Use token-budgeted dynamic micro-batching for teacher forward.
    +opsd.teacher.use_dynamic_bsz=${opsd_teacher_use_dynamic_bsz}
    # Maximum teacher forward tokens per GPU when dynamic batching is enabled.
    +opsd.teacher.max_token_len_per_gpu=${opsd_teacher_max_token_len_per_gpu}
    # Teacher fixed micro-batch size per GPU when dynamic batching is disabled.
    +opsd.teacher.micro_batch_size_per_gpu=${opsd_teacher_micro_batch_size_per_gpu}
    # Match actor-style remove-padding behavior for teacher forward.
    +opsd.teacher.use_remove_padding=${opsd_teacher_use_remove_padding}
    # Keep teacher forward on the non-fused path while OPSD logits/topK/tail is developed.
    +opsd.teacher.use_fused_kernels=${opsd_teacher_use_fused_kernels}
    # Teacher FSDP parameter offload policy; default keeps teacher parameters on GPU.
    +opsd.teacher.fsdp_config.param_offload=${opsd_teacher_param_offload}
    # Teacher is forward-only, so no optimizer state should be offloaded.
    +opsd.teacher.fsdp_config.optimizer_offload=${opsd_teacher_optimizer_offload}

    # Teacher parameter update policy. The default is no update; copy/ema are
    # reserved modes for later actor-to-teacher synchronization implementations.
    +opsd.teacher.update.mode=${opsd_teacher_update_mode}
    +opsd.teacher.update.interval=${opsd_teacher_update_interval}
    +opsd.teacher.update.ema_decay=${opsd_teacher_update_ema_decay}

    # OPSD loss and vocabulary comparison strategy. Start with full_vocab_kl/full
    # to verify the teacher-forward and loss path before enabling compressed topK modes.
    +opsd.loss.loss_mode=${opsd_loss_mode}
    # Token-level KL direction. forward_kl is the first implementation target;
    # reverse_kl and dynamic_merge are reserved comparison modes.
    +opsd.loss.kl_mode=${opsd_kl_mode}
    # Whether OPSD is standalone or coupled with an RL objective. Only none is
    # implemented now; grpo is an explicit future-extension hook.
    +opsd.loss.rl_coupling=${opsd_rl_coupling}
    +opsd.loss.topk_strategy=${opsd_topk_strategy}
    +opsd.loss.student_topk=${opsd_student_topk}
    +opsd.loss.teacher_topk=${opsd_teacher_topk}
    +opsd.loss.use_tail=${opsd_use_tail}
    +opsd.loss.loss_coef=${opsd_loss_coef}
    +opsd.loss.temperature=${opsd_temperature}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${OPSD[@]}" \
    "$@"
