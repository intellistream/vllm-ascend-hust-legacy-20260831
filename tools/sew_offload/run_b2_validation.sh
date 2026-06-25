#!/usr/bin/env bash
# B2 wave-streamed prefill NPU validation. Eager non-seam path so the B2 early
# branch in fused_experts intercepts before _maybe_apply_moe_offload_plan would
# fail-closed. num_slots=8 << prefill union (~51) => prefill runs B2 waves
# (ceil(51/8)=7 waves), decode (fanout 8 <= 8) keeps the normal single-pass path.
#
# Decisive check: OUTPUT_TOKENS + per-pos LOGPROBS == BASE (full residency) to
# 1e-5. B2 only changes WHERE expert weights live + wave accumulation order; the
# router/topk/gate/combine math is unchanged, so it must be numerically equal.
#
# Markers: SEW_B2 branch=WAVE_RUN (n_active/num_slots/n_waves) during prefill.
set -u
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV to a FREE NPU id}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-8}"
LOGPROBS="${LOGPROBS:-20}"
NONRES="${NONRES:-2,3,4,5}"
SLOTS="${SLOTS:-8}"
mkdir -p "$LOGDIR"
resident=$("$PY" -c "nr=set(int(x) for x in '$NONRES'.split(',')); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))")

run_base() {
  local log="$LOGDIR/B2_base.log"
  echo "[b2] BASE (offload OFF, eager) -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
    "$PY" "$PROBE" --case-name b2_base --enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  grep -E "OUTPUT_TOKENS|CASE_FAILED" "$log" | head -2
}

run_b2() {
  local log="$LOGDIR/B2_slots${SLOTS}.log"
  echo "[b2] B2 waves num_slots=$SLOTS nonres={$NONRES} eager -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" SEW_B2_PROBE=1 SEW_OFFLOAD_LEDGER=1 \
    VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
    VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS="$SLOTS" \
    VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD="$SLOTS" \
    VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1 \
    VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1 \
    VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline \
    VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0 \
    VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$resident" \
    VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=0 \
    VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=0 \
    VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL=1 \
    "$PY" "$PROBE" --case-name b2_slots${SLOTS} --enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  if grep -qiE "out of memory" "$log"; then echo "[b2] OOM";
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[b2] OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
    echo "[b2] WAVE_RUN markers: $(grep -c 'branch=WAVE_RUN' "$log")"
    grep 'branch=WAVE_RUN' "$log" | head -4
  else echo "[b2] FAILED"; grep -E "CASE_FAILED|Error|assert" "$log" | head -8; fi
}

run_base_captured() {
  local log="$LOGDIR/B2_base_captured.log"
  echo "[b2] BASE captured (offload OFF, ACLGraph) -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
    "$PY" "$PROBE" --case-name b2_base_captured --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  grep -E "OUTPUT_TOKENS|CASE_FAILED" "$log" | head -2
}

run_b2_seam() {
  local log="$LOGDIR/B2_seam_slots${SLOTS}.log"
  echo "[b2] B2+seam captured num_slots=$SLOTS nonres={$NONRES} -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" SEW_B2_PROBE=1 SEW_SEAM_PROBE=1 SEW_OFFLOAD_LEDGER=1 \
    VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
    VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS="$SLOTS" \
    VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD="$SLOTS" \
    VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1 \
    VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1 \
    VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline \
    VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0 \
    VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$resident" \
    VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=1 \
    VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=1 \
    VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL=1 \
    "$PY" "$PROBE" --case-name b2_seam_slots${SLOTS} --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  if grep -qiE "out of memory" "$log"; then echo "[b2] seam OOM";
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[b2] seam OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
    echo "[b2] seam WAVE_RUN: $(grep -c 'branch=WAVE_RUN' "$log")  b2_defer: $(grep -c 'reason=b2_wave_defer' "$log")  CAPTURING: $(grep -c 'branch=CAPTURING' "$log")"
    grep 'branch=WAVE_RUN' "$log" | head -4
  else echo "[b2] seam FAILED"; grep -E "CASE_FAILED|Error|assert" "$log" | head -8; fi
}

run_base
run_b2
run_base_captured
run_b2_seam
echo "[b2] DONE — eager: B2_base vs B2_slots${SLOTS}; captured: B2_base_captured vs B2_seam_slots${SLOTS}"
