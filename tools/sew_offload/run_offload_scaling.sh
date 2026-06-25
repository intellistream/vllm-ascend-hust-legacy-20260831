#!/usr/bin/env bash
# Offload-layer-count scaling experiment (decisive test: mis-routing vs masking).
#
# Hypothesis A (mis-routing defect): captured ACLGraph reads the persistent -1
#   log2phy buffer, so as the number of non-resident (offloaded) layers N grows,
#   more layers route to garbage -> divergence from BASE grows monotonically and
#   output degrades from ~=BASE to garbage.
# Hypothesis B (would-be correct): divergence stays ~flat near 0 regardless of N.
# Control: eager-SEW at the same N must stay ~=BASE (correct routing every step),
#   proving the defect is specific to the CAPTURED path, not SEW per se.
#
# Runs sequentially on a single dedicated NPU (default 5) to hold the card and
# avoid collisions. Each config = a fresh engine (config is set at init).
set -u

PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:-5}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-8}"
LOGPROBS="${LOGPROBS:-20}"

resident_csv() {  # $1 = nonresident csv (may be empty)
  "$PY" -c "import sys; nr=set(int(x) for x in sys.argv[1].split(',') if x.strip()); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))" "$1"
}

run_one() {  # $1=label  $2=enforce_flag(--enforce-eager|--no-enforce-eager)  $3=sew(1|0)  $4=nonresident_csv  $5=graphcompat(1|0)
  local label="$1" eager="$2" sew="$3" nonres="$4" gcompat="$5"
  local log="$LOGDIR/SCALE_${label}.log"
  local resident; resident="$(resident_csv "$nonres")"
  echo "=================================================================="
  echo "[scale] START label=$label eager=$eager sew=$sew nonres={$nonres} gcompat=$gcompat -> $log"
  echo "=================================================================="
  if [ "$sew" = "1" ]; then
    env \
      ASCEND_RT_VISIBLE_DEVICES="$DEV" \
      VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
      VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=128 \
      VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD=128 \
      VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1 \
      VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1 \
      VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline \
      VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0 \
      VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$resident" \
      VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE="$gcompat" \
      "$PY" "$PROBE" --case-name "scale_$label" "$eager" \
        --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
        --logprobs "$LOGPROBS" > "$log" 2>&1
  else
    # BASE: SEW disabled entirely
    env \
      ASCEND_RT_VISIBLE_DEVICES="$DEV" \
      VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
      "$PY" "$PROBE" --case-name "scale_$label" "$eager" \
        --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
        --logprobs "$LOGPROBS" > "$log" 2>&1
  fi
  local rc=$?
  if grep -qiE "out of memory|HBM out of memory|Free memory on device" "$log"; then
    echo "[scale] label=$label -> OOM (ceiling marker)"
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[scale] label=$label -> OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
  else
    echo "[scale] label=$label -> FAILED rc=$rc (see $log)"
  fi
}

# Order: BASE first (reference), then ascending captured N, then eager control.
run_one base_N0          --no-enforce-eager 0 ""              0
run_one cap_N1           --no-enforce-eager 1 "2"             1
run_one cap_N2           --no-enforce-eager 1 "2,3"           1
run_one cap_N4           --no-enforce-eager 1 "2,3,4,5"       1
run_one cap_N6           --no-enforce-eager 1 "2,3,4,5,6,7"   1
run_one eager_N4_control --enforce-eager    1 "2,3,4,5"       1

echo "[scale] ALL DONE"
