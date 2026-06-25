#!/usr/bin/env bash
# End-to-end SEW data-plane via the SERVICE path: only --ascend-moe-offload-gb
# (VLLM_ASCEND_MOE_OFFLOAD_GB) + VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1 are set.
# Autoconfig derives everything else (num_slots=8, ~12 offloaded layers for gb=14,
# and arms GRAPH_COMPATIBLE+STAGE_SEAM+B2_WAVE_PREFILL; does NOT wire the
# PrefetchOffloader). Captured (no --enforce-eager) => decode runs the captured
# graph, prefill runs B2 waves. This is the real "service command on SEW" config.
#
# Compares tokens against captured full-residency BASE.
set -u
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV to a FREE NPU id}"
LOGDIR=.planning/sew_offload/logs
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXTOK="${MAXTOK:-8}"
LOGPROBS="${LOGPROBS:-20}"
GB="${GB:-14}"
mkdir -p "$LOGDIR"

run_base() {
  local log="$LOGDIR/E2E_base_captured.log"
  echo "[e2e] BASE captured (offload OFF) -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" VLLM_ASCEND_MOE_OFFLOAD_ENABLED=0 \
    "$PY" "$PROBE" --case-name e2e_base --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  grep -E "OUTPUT_TOKENS|CASE_FAILED|Loading model weights took" "$log" | head -3
}

run_sew_service() {
  local log="$LOGDIR/E2E_sew_gb${GB}.log"
  echo "[e2e] SEW service path GB=$GB SEW_DATAPLANE=1 (autoconfig derives rest) -> $log"
  env ASCEND_RT_VISIBLE_DEVICES="$DEV" SEW_B2_PROBE=1 SEW_SEAM_PROBE=1 SEW_OFFLOAD_LEDGER=1 \
    VLLM_ASCEND_MOE_OFFLOAD_GB="$GB" \
    VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1 \
    "$PY" "$PROBE" --case-name e2e_sew_gb${GB} --no-enforce-eager \
      --gpu-memory-utilization "$GPU_UTIL" --max-tokens "$MAXTOK" --logprobs "$LOGPROBS" > "$log" 2>&1
  if grep -qiE "out of memory" "$log"; then echo "[e2e] OOM";
  elif grep -q "OUTPUT_TOKENS" "$log"; then
    echo "[e2e] OK: $(grep OUTPUT_TOKENS "$log" | tail -1)"
    echo "[e2e] weights: $(grep 'Loading model weights took' "$log" | tail -1 | sed 's/.*INFO//')"
    echo "[e2e] WAVE_RUN: $(grep -c 'branch=WAVE_RUN' "$log")  b2_defer: $(grep -c 'reason=b2_wave_defer' "$log")  CAPTURING: $(grep -c 'branch=CAPTURING' "$log")"
    echo "[e2e] distinct offloaded layers in WAVE_RUN: $(grep 'branch=WAVE_RUN' "$log" | grep -oE 'layer=[0-9]+' | sort -u | tr '\n' ' ')"
  else echo "[e2e] FAILED"; grep -E "CASE_FAILED|Error|assert|exceeds" "$log" | head -8; fi
}

run_base
run_sew_service
echo "[e2e] DONE — base=$LOGDIR/E2E_base_captured.log sew=$LOGDIR/E2E_sew_gb${GB}.log"
