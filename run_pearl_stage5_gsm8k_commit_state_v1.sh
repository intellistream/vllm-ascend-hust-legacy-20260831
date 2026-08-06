#!/usr/bin/env bash
# GSM8K batch=2 test for nano-PEARL batch accepted_len/valid_len commit.
# Run this inside the already activated vllm-hust-dev environment.

set +e

cd /root/data/vllm-ascend-hust || exit 2

LOG=/tmp/pearl_stage5_gsm8k_commit_state_b2.log
TERMINAL_LOG=/tmp/pearl_stage5_gsm8k_commit_state_b2_terminal.log
: > "$LOG"
: > "$TERMINAL_LOG"

export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=6,7

# True nano-PEARL batch commit path.
export PEARL_STAGE5_NANOPEARL_COMMIT_STATE=1
export PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK=1
export PEARL_STAGE5_NANOPEARL_TRACE=1
export PEARL_STAGE5_SAMPLE_OUTPUT_TRACE=1
export PEARL_ACCEPTANCE_DEBUG=1

python -u pearl_stage5_gsm8k_ac_benchmark_v1.py \
  --num-samples 2 \
  --max-num-seqs 2 \
  --batch-size 2 \
  --max-tokens 128 \
  --gamma 2 \
  --draft-device 6 \
  --target-device 7 \
  --max-model-len 2048 \
  --log-file "$LOG" \
  2>&1 | tee "$TERMINAL_LOG"

rc=${PIPESTATUS[0]}
echo "benchmark_exit=$rc" | tee -a "$TERMINAL_LOG"

echo
echo "===== COMMIT STATE SUMMARY ====="
grep -E \
  'commit_batch_start|commit_batch_done|COMMIT_STATE_V1|PEARL AC SUMMARY|draft tokens|accepted tokens|average AC rate|child exit code|Traceback|ERROR' \
  "$LOG" | tail -120

echo
echo "full_log=$LOG"
echo "terminal_log=$TERMINAL_LOG"

exit "$rc"
