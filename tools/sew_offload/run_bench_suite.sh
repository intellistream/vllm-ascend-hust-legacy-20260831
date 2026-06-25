#!/usr/bin/env bash
# Benchmark suite: 4 configs x {conc=1, conc=10}, STRICT ISOLATION.
# Each config runs in its OWN server process; after its two client runs we kill
# it BY PID (never pgrep python3 — that would kill the user's :8016 server) and
# wait until npu-smi shows the device free before starting the next config.
#
# Shared server args = the validated reference command (max-num-seqs=1 etc.),
# port 8020 (NEVER 8016). Client = vllm bench serve over 100 ShareGPT prompts.
set -u
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
VLLM=/root/miniconda3/envs/vllm-hust-dev/bin/vllm
MODEL=/data/shared-models/Qwen3-30B-A3B
SERVED=qwen3-30b-a3b
PORT="${PORT:-8020}"
DEV="${DEV:?set DEV to a FREE NPU id (avoid the :8016 card)}"
DATASET=benchmarks/results/moe_offload_real_sharegpt_qwen3_30b_a3b/ShareGPT_prompt_le256_for_mlen512.json
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTDIR=benchmarks/results/bench_suite_4cfg
LOGDIR=.planning/sew_offload/logs
mkdir -p "$OUTDIR" "$LOGDIR"

SERVER_PID=""
HBM_POLL_PID=""
PEAK_FILE=""

wait_for_health() {
  local waited=0 timeout=1800
  while (( waited < timeout )); do
    if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then return 0; fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[suite] server pid=$SERVER_PID died during startup"; return 1
    fi
    sleep 3; ((waited+=3))
  done
  echo "[suite] timeout waiting for /health"; return 1
}

start_hbm_poll() {  # $1=label  -> background npu-smi peak sampler
  PEAK_FILE="$LOGDIR/HBM_${1}.peak"
  : > "$PEAK_FILE"
  ( while true; do
      m=$(npu-smi info -t proc-mem -i "$DEV" 2>/dev/null | grep -oiE "[0-9]+ *MB|Process memory.*" | grep -oE "[0-9]+" | sort -rn | head -1)
      [[ -n "$m" ]] && echo "$m" >> "$PEAK_FILE"
      sleep 2
    done ) &
  HBM_POLL_PID=$!
}

stop_hbm_poll() {
  [[ -n "$HBM_POLL_PID" ]] && kill "$HBM_POLL_PID" 2>/dev/null || true
  HBM_POLL_PID=""
}

kill_server_by_pid() {
  [[ -z "$SERVER_PID" ]] && return 0
  echo "[suite] killing server pid=$SERVER_PID (+ its process group)"
  kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  local waited=0
  while (( waited < 90 )); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2; ((waited+=2))
  done
  kill -KILL "-${SERVER_PID}" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  # Confirm the device is actually released before the next config.
  local w=0
  while (( w < 90 )); do
    if npu-smi info -t proc-mem -i "$DEV" 2>/dev/null | grep -qi "No process"; then
      echo "[suite] DEV=$DEV released"; return 0
    fi
    sleep 3; ((w+=3))
  done
  echo "[suite] WARN: DEV=$DEV still shows a process after kill"; return 0
}

run_client() {  # $1=cfg label  $2=concurrency
  local cfg="$1" conc="$2"
  local rf="bench_${cfg}_conc${conc}.json"
  echo "[suite] client cfg=$cfg conc=$conc num_prompts=$NUM_PROMPTS -> $OUTDIR/$rf"
  "$VLLM" bench serve \
    --backend openai --endpoint /v1/completions \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --served-model-name "$SERVED" \
    --dataset-name sharegpt --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$conc" --request-rate inf \
    --save-result --result-dir "$OUTDIR" --result-filename "$rf" \
    > "$LOGDIR/client_${cfg}_conc${conc}.log" 2>&1
  if [[ -f "$OUTDIR/$rf" ]]; then
    "$PY" - "$OUTDIR/$rf" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"   TTFT mean/med={d.get('mean_ttft_ms'):.1f}/{d.get('median_ttft_ms'):.1f}ms "
      f"TPOT mean/med={d.get('mean_tpot_ms'):.2f}/{d.get('median_tpot_ms'):.2f}ms "
      f"out_tok/s={d.get('output_throughput'):.1f} req/s={d.get('request_throughput'):.3f}")
PYEOF
  else
    echo "   [suite] no result json — see client log"; tail -5 "$LOGDIR/client_${cfg}_conc${conc}.log"
  fi
}

run_config() {  # $1=label  $2=enforce_eager(0/1)  $3=extra env string (space-sep KEY=VAL)
  local cfg="$1" eager="$2" extra="$3"
  local slog="$LOGDIR/server_${cfg}.log"
  echo "=================================================================="
  echo "[suite] CONFIG $cfg  eager=$eager  extra='$extra'  DEV=$DEV PORT=$PORT"
  echo "=================================================================="
  local eager_flag=""; [[ "$eager" == "1" ]] && eager_flag="--enforce-eager"
  # setsid -> own process group so we can kill the whole tree by -PID.
  env $extra \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    setsid "$VLLM" serve "$MODEL" \
      --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
      --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 \
      --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 \
      --kv-cache-memory-bytes 536870912 $eager_flag \
      > "$slog" 2>&1 &
  SERVER_PID=$!
  echo "[suite] server pid=$SERVER_PID -> $slog"
  if ! wait_for_health; then
    echo "[suite] $cfg server failed health"; grep -E "Error|error|Traceback|CUDA|HBM|OOM|exceeds" "$slog" | head -8
    kill_server_by_pid; return 1
  fi
  echo "[suite] $cfg HEALTHY. weights: $(grep 'Loading model weights took' "$slog" | tail -1 | sed 's/.*INFO//')"
  start_hbm_poll "$cfg"
  run_client "$cfg" 1
  run_client "$cfg" 10
  stop_hbm_poll
  local peak="n/a"
  [[ -s "$PEAK_FILE" ]] && peak="$(sort -rn "$PEAK_FILE" | head -1) MB"
  echo "[suite] $cfg PEAK HBM (npu-smi proc-mem) = $peak"
  echo "$cfg weights=$(grep 'Loading model weights took' "$slog" | tail -1 | grep -oE '[0-9.]+ GB') peak_hbm=$peak" >> "$LOGDIR/HBM_summary.txt"
  kill_server_by_pid
}

trap 'stop_hbm_poll; kill_server_by_pid' EXIT INT TERM

: > "$LOGDIR/HBM_summary.txt"
# Config 1: full residency + ACLGraph (capture on)
run_config cfg1_fullresid_aclgraph 0 "VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0"
# Config 2: full residency + eager single-op
run_config cfg2_fullresid_eager 1 "VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0"
# Config 3: GB=14 eager single-op slot=8 (reference command as-is: PrefetchOffloader)
run_config cfg3_gb14_eager_slot8 1 "VLLM_ASCEND_MOE_OFFLOAD_GB=14"
# Config 4: GB=14 B2 + seam (graph capture) slot=8
run_config cfg4_gb14_b2seam_slot8 0 "VLLM_ASCEND_MOE_OFFLOAD_GB=14 VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1"
echo "[suite] ALL DONE. results in $OUTDIR ; HBM summary $LOGDIR/HBM_summary.txt"
