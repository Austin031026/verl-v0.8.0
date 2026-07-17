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
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
enable_thinking=${ENABLE_THINKING:-True}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}

opsd_kl_mode=${OPSD_KL_MODE:-reverse_kl}
opsd_rl_coupling=${OPSD_RL_COUPLING:-none}
# The first cluster diagnostic defaults to full-softmax then student top-K truncation.
opsd_vocab_strategy=${OPSD_VOCAB_STRATEGY:-student_truncated}
opsd_student_topk=${OPSD_STUDENT_TOPK:-8}
opsd_chunked_topk_chunk_size=${OPSD_CHUNKED_TOPK_CHUNK_SIZE:-4096}
opsd_loss_coef=${OPSD_LOSS_COEF:-1.0}
opsd_temperature=${OPSD_TEMPERATURE:-1.0}

opsd_teacher_update_mode=${OPSD_TEACHER_UPDATE_MODE:-none}

# Teacher forward uses dynamic token-budgeted micro-batches by default.
opsd_teacher_use_dynamic_bsz=${OPSD_TEACHER_USE_DYNAMIC_BSZ:-True}
# Maximum tokenized question + privileged information + chat-template prefix.
opsd_teacher_max_prompt_length=${OPSD_TEACHER_MAX_PROMPT_LENGTH:-12288}
# Total teacher input limits include the student rollout response.
opsd_teacher_max_context_no_think=${OPSD_TEACHER_MAX_CONTEXT_NO_THINK:-16000}
opsd_teacher_max_context_thinking=${OPSD_TEACHER_MAX_CONTEXT_THINKING:-32000}
# Teacher forward token budget per GPU; it must fit at least one teacher sequence.
opsd_teacher_max_token_len_per_gpu=${OPSD_TEACHER_MAX_TOKEN_LEN_PER_GPU:-16000}
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

# Real-runtime OPSD test mode. The report is populated by the same logits and
# token losses consumed by the actor backward pass.
opsd_test_enabled=${OPSD_TEST_ENABLED:-False}
opsd_test_steps=${OPSD_TEST_STEPS:-'[1,2]'}
opsd_test_output_path=${OPSD_TEST_OUTPUT_PATH:-$PWD/opsd_test_result.json}
opsd_test_topk=${OPSD_TEST_TOPK:-5}
opsd_test_max_samples_per_step=${OPSD_TEST_MAX_SAMPLES_PER_STEP:-2}
opsd_test_max_samples_per_worker_micro_batch=${OPSD_TEST_MAX_SAMPLES_PER_WORKER_MICRO_BATCH:-2}
opsd_test_max_response_tokens_per_sample=${OPSD_TEST_MAX_RESPONSE_TOKENS_PER_SAMPLE:-32}
opsd_test_max_loss_vocab_tokens=${OPSD_TEST_MAX_LOSS_VOCAB_TOKENS:-32}

# Smoke-test default: two fresh rollout -> OPSD update cycles. Each rollout is consumed once.
total_training_steps=${TOTAL_TRAINING_STEPS:-2}
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

logger=${LOGGER:-'["console"]'}
project_name=${PROJECT_NAME:-verl_opsd}
experiment_name=${EXPERIMENT_NAME:-qwen3_2b_opsd_vllm_fsdp}
# ---- end user-adjustable ----

train_files=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
val_files=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# OPSD vocabulary strategy:
#   full              - compare over the full vocabulary
#   student_renorm    - select student topK logits and renormalize both distributions within topK
#   student_truncated - select probabilities from the full-vocab distributions without renormalizing
#   teacher_* / union_* variants are reserved for future implementations
case "${opsd_vocab_strategy}" in
    full|student_renorm|student_truncated)
        ;;
    teacher_renorm|teacher_truncated|union_renorm|union_truncated)
        echo "OPSD_VOCAB_STRATEGY=${opsd_vocab_strategy} is reserved but not implemented yet." >&2
        exit 1
        ;;
    *)
        echo "Invalid OPSD_VOCAB_STRATEGY=${opsd_vocab_strategy}." >&2
        echo "Expected full, student_*, teacher_*, or union_* with an explicit probability mode." >&2
        exit 1
        ;;
esac

# OPSD token-level KL direction:
#   reverse_kl    - KL(student || teacher), implemented path
#   forward_kl    - KL(teacher || student), reserved for future implementation
#   dynamic_merge - reserved for dynamically mixing forward/reverse KL variants
case "${opsd_kl_mode}" in
    reverse_kl)
        ;;
    forward_kl|dynamic_merge)
        echo "OPSD_KL_MODE=${opsd_kl_mode} is reserved but not implemented yet." >&2
        exit 1
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
    none)
        ;;
    grpo)
        echo "OPSD_RL_COUPLING=grpo is reserved but not implemented yet." >&2
        exit 1
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
    none)
        ;;
    copy|ema)
        echo "OPSD_TEACHER_UPDATE_MODE=${opsd_teacher_update_mode} is reserved but not implemented yet." >&2
        exit 1
        ;;
    *)
        echo "Invalid OPSD_TEACHER_UPDATE_MODE=${opsd_teacher_update_mode}. Expected one of: none, copy, ema." >&2
        exit 1
        ;;
esac

# OPSD teacher privileged input source:
#   answer       - use the current sample's final answer/ground truth
#   reason       - use the current sample's teacher reasoning/rationale
#   cot_examples - reserved for shared or retrieved CoT demonstrations
case "${opsd_teacher_privileged_input_mode}" in
    answer|reason)
        ;;
    cot_examples)
        echo "OPSD_TEACHER_PRIVILEGED_INPUT_MODE=${opsd_teacher_privileged_input_mode}" \
            "is reserved but not implemented yet." >&2
        exit 1
        ;;
    *)
        echo "Invalid OPSD_TEACHER_PRIVILEGED_INPUT_MODE=${opsd_teacher_privileged_input_mode}." \
            "Expected one of: answer, reason, cot_examples." >&2
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
    +data.apply_chat_template_kwargs.enable_thinking=${enable_thinking}
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
    # One optimizer update per rollout batch; the next update must use a freshly generated rollout.
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # The current OPSD path computes each complete sequence without Ulysses token sharding.
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
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
    trainer.total_training_steps=${total_training_steps}
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
    # answer and reason are implemented; cot_examples remains a reserved hook.
    +opsd.teacher.privileged_input.mode=${opsd_teacher_privileged_input_mode}
    # Limit the exact tokenized teacher prefix after question, privileged input,
    # and the chat template have been combined.
    +opsd.teacher.max_prompt_length=${opsd_teacher_max_prompt_length}
    # Limit the complete teacher sequence independently for no-think/thinking templates.
    +opsd.teacher.max_context_length.no_think=${opsd_teacher_max_context_no_think}
    +opsd.teacher.max_context_length.thinking=${opsd_teacher_max_context_thinking}
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

    # Teacher parameter update policy. Only a fixed teacher is implemented.
    +opsd.teacher.update.mode=${opsd_teacher_update_mode}

    # Vocabulary support used by the common OPSD KL objective.
    +opsd.loss.vocab_strategy=${opsd_vocab_strategy}
    # Token-level KL direction. reverse_kl is implemented; the other modes are reserved.
    +opsd.loss.kl_mode=${opsd_kl_mode}
    # Whether OPSD is standalone or coupled with an RL objective. Only none is
    # implemented now; grpo is an explicit future-extension hook.
    +opsd.loss.rl_coupling=${opsd_rl_coupling}
    +opsd.loss.student_topk=${opsd_student_topk}
    +opsd.loss.chunked_topk_chunk_size=${opsd_chunked_topk_chunk_size}
    +opsd.loss.loss_coef=${opsd_loss_coef}
    +opsd.loss.temperature=${opsd_temperature}

    # Optional bounded evidence capture from the real verl OPSD forward/backward path.
    +opsd.test.enabled=${opsd_test_enabled}
    +opsd.test.steps=${opsd_test_steps}
    +opsd.test.output_path=${opsd_test_output_path}
    +opsd.test.topk=${opsd_test_topk}
    +opsd.test.max_samples_per_step=${opsd_test_max_samples_per_step}
    +opsd.test.max_samples_per_worker_micro_batch=${opsd_test_max_samples_per_worker_micro_batch}
    +opsd.test.max_response_tokens_per_sample=${opsd_test_max_response_tokens_per_sample}
    +opsd.test.max_loss_vocab_tokens=${opsd_test_max_loss_vocab_tokens}
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
