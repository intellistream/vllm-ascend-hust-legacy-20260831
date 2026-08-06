#!/usr/bin/env bash
# GSM8K smoke runner for nano-PEARL gamma=2 by default.
# gamma=4 is intentionally excluded until the container setup is fixed.

set +e

cd /root/data/vllm-ascend-hust || exit 2

PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_PATH=${DATA_PATH:-/data/datasets/gsm8k/test.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-2}
MAX_TOKENS=${MAX_TOKENS:-128}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
DRAFT_DEVICE=${DRAFT_DEVICE:-6}
TARGET_DEVICE=${TARGET_DEVICE:-7}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-600}
LOG_DIR=${LOG_DIR:-/tmp}
GAMMAS=${GAMMAS:-"2"}

case " $GAMMAS " in
  *" 4 "*)
    echo "gamma=4 is disabled for this runner; use only GAMMAS=\"1 2\"."
    exit 2
    ;;
esac

for gamma in $GAMMAS; do
  case "$gamma" in
    1|2)
      ;;
    *)
      echo "unsupported gamma=$gamma; this runner only allows gamma=1 or gamma=2."
      exit 2
      ;;
  esac
done

export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${DRAFT_DEVICE},${TARGET_DEVICE}}

# True nano-PEARL commit path. Defaults can be overridden by the caller.
export PEARL_STAGE5_NANOPEARL_COMMIT_STATE=${PEARL_STAGE5_NANOPEARL_COMMIT_STATE:-1}
export PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK=${PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK:-1}
export PEARL_STAGE5_NANOPEARL_LENGTH_ONLY=${PEARL_STAGE5_NANOPEARL_LENGTH_ONLY:-1}
export PEARL_STAGE5_NANOPEARL_STRICT=${PEARL_STAGE5_NANOPEARL_STRICT:-0}
export PEARL_STAGE5_NANOPEARL_TRACE=${PEARL_STAGE5_NANOPEARL_TRACE:-1}
export PEARL_STAGE5_SAMPLE_OUTPUT_TRACE=${PEARL_STAGE5_SAMPLE_OUTPUT_TRACE:-1}
export PEARL_ACCEPTANCE_DEBUG=${PEARL_ACCEPTANCE_DEBUG:-1}

mkdir -p "$LOG_DIR"

overall_rc=0
summary_file="$LOG_DIR/pearl_stage5_gsm8k_gamma12_summary.log"
: > "$summary_file"

echo "===== GSM8K nano-PEARL gamma runner =====" | tee -a "$summary_file"
echo "data_path=$DATA_PATH" | tee -a "$summary_file"
echo "num_samples=$NUM_SAMPLES max_tokens=$MAX_TOKENS" | tee -a "$summary_file"
echo "max_num_seqs=$MAX_NUM_SEQS batch_size=$BATCH_SIZE" | tee -a "$summary_file"
echo "draft_device=$DRAFT_DEVICE target_device=$TARGET_DEVICE" | tee -a "$summary_file"
echo "gammas=$GAMMAS" | tee -a "$summary_file"

for gamma in $GAMMAS; do
  log_file="$LOG_DIR/pearl_stage5_gsm8k_gamma${gamma}.log"
  terminal_log="$LOG_DIR/pearl_stage5_gsm8k_gamma${gamma}_terminal.log"
  : > "$log_file"
  : > "$terminal_log"

  echo | tee -a "$summary_file"
  echo "===== RUN gamma=$gamma =====" | tee -a "$summary_file"

  "$PYTHON_BIN" -u pearl_stage5_gsm8k_ac_benchmark_v1.py \
    --data-path "$DATA_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --gamma "$gamma" \
    --draft-device "$DRAFT_DEVICE" \
    --target-device "$TARGET_DEVICE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --startup-timeout "$STARTUP_TIMEOUT" \
    --log-file "$log_file" \
    2>&1 | tee "$terminal_log"

  rc=${PIPESTATUS[0]}
  echo "benchmark_exit=$rc" | tee -a "$terminal_log" "$summary_file"

  echo "----- gamma=$gamma AC summary -----" | tee -a "$summary_file"
  grep -E \
    'PEARL AC SUMMARY|draft tokens|accepted tokens|average AC rate|child exit code' \
    "$terminal_log" | tail -20 | tee -a "$summary_file"

  echo "----- gamma=$gamma key markers -----" | tee -a "$summary_file"
  grep -E \
    'commit_batch_start|commit_batch_done|LENGTH_ONLY|COMMIT_STATE_V1|PEARL AC SUMMARY|draft tokens|accepted tokens|average AC rate|child exit code|Traceback|ERROR' \
    "$log_file" | tail -160 | tee -a "$summary_file"

  echo "full_log=$log_file" | tee -a "$summary_file"
  echo "terminal_log=$terminal_log" | tee -a "$summary_file"

  if [ "$rc" -ne 0 ]; then
    overall_rc=1
  fi
done

echo | tee -a "$summary_file"
echo "===== FINAL gamma summary =====" | tee -a "$summary_file"
echo "overall_exit=$overall_rc" | tee -a "$summary_file"
echo "summary_log=$summary_file"

exit "$overall_rc"
