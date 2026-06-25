#!/usr/bin/env bash
# Regime B path ① seam validation (R3: FX-splitter partition + token correctness).
#
# The seam = vllm::moe_offload_stage, a custom splitting op inserted between the
# router (select_experts) and the grouped MLP. Registered into splitting_ops
# (platform.py, gated on VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM), so the FX splitter
# cuts the captured region there and the op runs EAGER between two captured
# pieces -> legal D2H + H2D + log2phy write per step.
#
# R3-a (NUM_SLOTS=128): working-set guard never fires (all 128 experts fit), so
#   the ONLY thing under test is the partition mechanism: does capture succeed,
#   does the op run eager between pieces, are tokens == BASE? Isolates R3 from R4.
# R3-b (NUM_SLOTS=16): true HBM saving (16 < 128). Decode per-step working set is
#   <= top_k=8 so it should fit; prefill union may exceed 16 -> exposes whether
#   the minimal seam needs an eager-prefill fallback (R4). Run only if R3-a green.
#
# BASE reference tokens (N4, greedy): [3555, 525, 279, 22146, 323, 63625, 315, 1667]
set -u

PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV to a FREE NPU id, e.g. DEV=1}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-8}"
LOGPROBS="${LOGPROBS:-20}"

resident_csv() {
  "$PY" -c "import sys; nr=set(int(x) for x in sys.argv[1].split(',') if x.strip()); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))" "$1"
}

run_seam() {  # $1=label $2=num_slots $3=nonresident_csv
  local label="$1" slots="$2" nonres="$3"
  local log="$LOGDIR/SEAM_${label}.log"
  local resident; resident="$(resident_csv "$nonres")"
  echo "=================================================================="
  echo "[seam] START label=$label num_slots=$slots nonres={$nonres} -> $log"
  echo "=================================================================="
  env \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
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
    "$PY" "$PROBE" --case-name "seam_$label" --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
      --logprobs "$LOGPROBS" > "$log" 2>&1
  local rc=$?
  if grep -qiE "out of memory|HBM out of memory|Free memory on device" "$log"; then
    echo "[seam] label=$label -> OOM"
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[seam] label=$label -> OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
  else
    echo "[seam] label=$label -> FAILED rc=$rc (see $log)"
    grep -E "CASE_FAILED|Error|error|assert" "$log" | head -5
  fi
}

# R3-a: partition mechanism only (all experts fit; guard never fires).
run_seam r3a_slots128_N4 128 "2,3,4,5"
# R3-b: true HBM saving (16 < 128). Run regardless to see prefill working-set behavior.
run_seam r3b_slots16_N4  16  "2,3,4,5"

echo "[seam] ALL DONE"
