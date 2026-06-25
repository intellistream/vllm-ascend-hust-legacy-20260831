#!/usr/bin/env bash
# P0: latency measurement of the B1 offload path vs full-residency baseline.
#
# Answers the question the V-D correctness run could not: how much does offload
# cost, and where does the time go? Two macro points (base = offload OFF, captured;
# b1 = num_slots=96 < n=128 per-step staging) measured with the differential
# TTFT/TPOT method (--latency), plus per-seam STAGE_MS (SEW_SEAM_PROBE) on the b1
# run so decode-step staging cost is attributable.
#
# Greppable result markers:
#   LATENCY case=... TTFT_MS=... TPOT_MS=... DECODE_TPS=...   (both runs)
#   SEW_SEAM branch=EAGER_STAGED ... STAGE_MS=...             (b1 run only)
#
# This run does NOT modify any main path: offload stays env-gated + default-off;
# the STAGE_MS sync and the latency repeats are diagnostic-only.
set -u

PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV to a FREE NPU id, e.g. DEV=4}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-32}"          # need enough decode steps for a stable TPOT
REPS="${REPS:-5}"
NONRES="${NONRES:-2,3,4,5}"

mkdir -p "$LOGDIR"

resident_csv() {
  "$PY" -c "import sys; nr=set(int(x) for x in sys.argv[1].split(',') if x.strip()); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))" "$1"
}

run_base() {
  local log="$LOGDIR/P0_base_latency.log"
  echo "=================================================================="
  echo "[p0] START base (offload OFF, captured) maxtok=$MAXTOK reps=$REPS -> $log"
  echo "=================================================================="
  env \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
    VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
    "$PY" "$PROBE" --case-name "p0_base" --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
      --latency --latency-repeats "$REPS" > "$log" 2>&1
  grep -E "LATENCY case=" "$log" || { echo "[p0] base FAILED (see $log)"; grep -E "CASE_FAILED|Error" "$log" | head -5; }
}

run_b1() {
  local slots="${1:-96}"
  local log="$LOGDIR/P0_b1_slots${slots}_latency.log"
  local resident; resident="$(resident_csv "$NONRES")"
  echo "=================================================================="
  echo "[p0] START b1 num_slots=$slots nonres={$NONRES} maxtok=$MAXTOK reps=$REPS -> $log"
  echo "=================================================================="
  env \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
    SEW_SEAM_PROBE=1 \
    VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
    VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS="$slots" \
    VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD="$slots" \
    VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1 \
    VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1 \
    VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline \
    VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0 \
    VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$resident" \
    VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=1 \
    VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=1 \
    "$PY" "$PROBE" --case-name "p0_b1_slots${slots}" --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
      --latency --latency-repeats "$REPS" > "$log" 2>&1
  if grep -qiE "out of memory|HBM out of memory" "$log"; then
    echo "[p0] b1 -> OOM (see $log)"
  elif grep -qE "LATENCY case=" "$log"; then
    grep -E "LATENCY case=" "$log"
    # Decode-step staging cost: STAGE_MS distribution over decode-fanout calls
    # (n_active small, ~8). Reported as count + mean for a quick read; the log has
    # every line for a full histogram.
    "$PY" - "$log" <<'PYEOF'
import re, sys, statistics
ms = [float(m.group(1)) for line in open(sys.argv[1])
      for m in [re.search(r"STAGE_MS=([0-9.]+)", line)] if m]
small = [float(re.search(r"STAGE_MS=([0-9.]+)", l).group(1)) for l in open(sys.argv[1])
         if "branch=EAGER_STAGED" in l and re.search(r"n_active=([0-9]+)", l) and int(re.search(r"n_active=([0-9]+)", l).group(1)) <= 8]
if ms:
    print(f"[p0] STAGE_MS all: n={len(ms)} mean={statistics.mean(ms):.3f} "
          f"median={statistics.median(ms):.3f} max={max(ms):.3f}")
if small:
    print(f"[p0] STAGE_MS decode-only (n_active<=8): n={len(small)} "
          f"mean={statistics.mean(small):.3f} median={statistics.median(small):.3f}")
PYEOF
  else
    echo "[p0] b1 FAILED (see $log)"; grep -E "CASE_FAILED|Error|assert" "$log" | head -8
  fi
}

run_base
run_b1 96
echo "[p0] ALL DONE — compare TTFT_MS / TPOT_MS / DECODE_TPS between base and b1;"
echo "[p0] STAGE_MS decode-only mean x (#offload layers=4) approximates per-step staging tax."
