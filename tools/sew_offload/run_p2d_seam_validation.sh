#!/usr/bin/env bash
# P2d: NPU validation of the P2c three-way seam (moe_router_indirect | moe_offload_stage
# [splitting/eager] | moe_mlp) wired via AscendMoERunner._select_forward.
#
# This supersedes the R3 single-op-in-apply() seam (which was NEGATIVE: the seam
# was buried inside the opaque vllm::moe_forward node, invisible to split_graph,
# so decode ran ZERO Python in the MoE body). The P2c path makes the three ops
# TOP-LEVEL FX nodes; moe_offload_stage is in compilation_config.splitting_ops so
# the FX splitter cuts the captured region there and it runs EAGER every step.
#
# Decisive markers (single seam run, SEW_SEAM_PROBE=1):
#   V-C  numerical equivalence: OUTPUT_TOKENS + per-pos LOGPROBS == BASE (to 1e-5).
#   V-D  per-step eager staging: "SEW_SEAM branch=EAGER_STAGED" lines appear during
#        DECODE (not just prefill) -> count >> R3's 0. In R3 the op never ran in
#        decode; here it must run once per offloaded layer per decode step.
#   V-E  (corollary of V-D): EAGER_STAGED at replay proves moe_offload_stage is a
#        real top-level splitting boundary between two captured pieces.
set -u

PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV to a FREE NPU id, e.g. DEV=4}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-8}"
LOGPROBS="${LOGPROBS:-20}"
NONRES="${NONRES:-2,3,4,5}"

resident_csv() {
  "$PY" -c "import sys; nr=set(int(x) for x in sys.argv[1].split(',') if x.strip()); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))" "$1"
}

run_base() {
  local log="$LOGDIR/P2D_base_N0.log"
  echo "=================================================================="
  echo "[p2d] START base (offload OFF, captured) -> $log"
  echo "=================================================================="
  env \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
    VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
    "$PY" "$PROBE" --case-name "p2d_base" --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
      --logprobs "$LOGPROBS" > "$log" 2>&1
  if grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[p2d] base -> OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
  else
    echo "[p2d] base -> FAILED (see $log)"; grep -E "CASE_FAILED|Error|assert" "$log" | head -5
  fi
}

run_seam() {  # $1=label $2=num_slots
  local label="$1" slots="$2"
  local log="$LOGDIR/P2D_${label}.log"
  local resident; resident="$(resident_csv "$NONRES")"
  echo "=================================================================="
  echo "[p2d] START $label num_slots=$slots nonres={$NONRES} seam=1 -> $log"
  echo "=================================================================="
  env \
    ASCEND_RT_VISIBLE_DEVICES="$DEV" \
    SEW_SEAM_PROBE=1 \
    SEW_OFFLOAD_LEDGER=1 \
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
    "$PY" "$PROBE" --case-name "p2d_$label" --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" \
      --logprobs "$LOGPROBS" > "$log" 2>&1
  local rc=$?
  if grep -qiE "out of memory|HBM out of memory|Free memory on device" "$log"; then
    echo "[p2d] $label -> OOM"
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[p2d] $label -> OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
    echo "[p2d] $label EAGER_STAGED markers: $(grep -c 'branch=EAGER_STAGED' "$log")  CAPTURING: $(grep -c 'branch=CAPTURING' "$log")  PASSTHROUGH(regime_a): $(grep -c 'branch=EAGER_PASSTHROUGH reason=regime_a' "$log")  PASSTHROUGH(full_weight_path): $(grep -c 'branch=EAGER_PASSTHROUGH reason=full_weight_path' "$log")"
    echo "[p2d] $label load-time log2phy fill (first/last layer):"
    grep 'SEW_LEDGER' "$log" | head -1
    grep 'SEW_LEDGER' "$log" | tail -1
  else
    echo "[p2d] $label -> FAILED rc=$rc (see $log)"
    grep -E "CASE_FAILED|Error|error|assert" "$log" | head -8
  fi
}

run_base
# V-C + V-D: slots=128 (all experts fit; isolates the partition mechanism from R4
# eviction). Offloaded layers {NONRES} stage eager every step via the seam.
run_seam r3a_slots128 128
# V-D (true Regime B "B1"): slots=96 < n(128). An offloaded layer has NO NPU
# full-weight copy (its w13/w2 were staged to CPU at load), so EVERY call -- the
# eager prefill AND each captured decode step -- must run through the slot bank,
# computing at most num_slots experts. With num_slots >= the prompt's per-layer
# active union (~51) the prefill call fits, and decode (<=8) stages per step. This
# is genuine num_slots < n per-step staging (32 experts never own a slot). A call
# whose distinct active set exceeds num_slots fail-closes with "exceeds num_slots"
# (that working set needs wave-streamed prefill, "B2", a separate feature).
run_seam b1_slots96 96
echo "[p2d] ALL DONE"
