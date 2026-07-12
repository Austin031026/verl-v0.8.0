#!/usr/bin/env bash
# Two real rollout -> fixed-teacher OPSD update cycles with bounded JSON evidence.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}

export OPSD_TEST_ENABLED=${OPSD_TEST_ENABLED:-True}
export OPSD_TEST_STEPS=${OPSD_TEST_STEPS:-'[1,2]'}
export OPSD_TEST_OUTPUT_PATH=${OPSD_TEST_OUTPUT_PATH:-$PWD/opsd_test_result.json}
export OPSD_TEST_TOPK=${OPSD_TEST_TOPK:-5}
export OPSD_TEST_MAX_SAMPLES_PER_STEP=${OPSD_TEST_MAX_SAMPLES_PER_STEP:-2}
export OPSD_TEST_MAX_SAMPLES_PER_WORKER_MICRO_BATCH=${OPSD_TEST_MAX_SAMPLES_PER_WORKER_MICRO_BATCH:-2}
export OPSD_TEST_MAX_RESPONSE_TOKENS_PER_SAMPLE=${OPSD_TEST_MAX_RESPONSE_TOKENS_PER_SAMPLE:-32}
# The default OPSD student_topk is 8, so all loss terms fit in the report.
export OPSD_TEST_MAX_LOSS_VOCAB_TOKENS=${OPSD_TEST_MAX_LOSS_VOCAB_TOKENS:-32}

exec bash "$SCRIPT_DIR/run_qwen3_2b_opsd_fsdp.sh" "$@"
