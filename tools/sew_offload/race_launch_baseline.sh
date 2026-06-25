#!/bin/bash
# Baseline ground-truth launcher: NO MoE offload, ACLGraph ENABLED.
# Scans ALL 8 cards and grabs the first one that frees up (<10% HBM), so we do
# not depend on any single contended card. Produces the ground-truth OUTPUT_TOKENS
# for the same prompt/seed/max_tokens as the SEW-only run, for token-id parity.
#
# It sets NO VLLM_ASCEND_MOE_OFFLOAD_* and NO GB env -> offload fully disabled
# (NoopOffloader, no fixed slots) -> the model runs every layer fully resident.
# This is the correctness reference the SEW-only capture run must match.
#
# Usage: race_launch_baseline.sh <flag_unused> <case_name> <logfile> <max_tries>
set -u
CASE="${1:?case}"; LOG="${2:?log}"; MAX="${3:-40}"
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=/root/vllm-ascend-hust/tools/sew_offload/run_graph_compat_capture_probe.py

free_pct() { npu-smi info -t usages -i "$1" -c 0 2>/dev/null | grep "HBM Usage Rate" | grep -oE "[0-9]+$"; }

for try in $(seq 1 "$MAX"); do
  PICK=""
  for w in $(seq 1 200); do
    for d in 0 1 2 3 4 5 6 7; do
      u=$(free_pct "$d"); u=${u:-100}
      if [ "$u" -lt 10 ]; then PICK="$d"; break; fi
    done
    [ -n "$PICK" ] && break
    sleep 5
  done
  [ -z "$PICK" ] && { echo "[base try=$try] no free card after wait" | tee -a "$LOG.race"; continue; }
  echo "[base try=$try] NPU$PICK free — launching baseline (no offload)" | tee -a "$LOG.race"
  ASCEND_RT_VISIBLE_DEVICES="$PICK" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$PY" "$PROBE" --case-name "$CASE" --no-enforce-eager --gpu-memory-utilization 0.90 ${EXTRA_PROBE_ARGS:-} \
    > "$LOG" 2>&1
  if grep -q "OUTPUT_TOKENS" "$LOG"; then
    echo "[base try=$try] WON on NPU$PICK — tokens captured" | tee -a "$LOG.race"; exit 0
  fi
  if grep -qE "Free memory on device|NPU out of memory|HBM out of memory" "$LOG"; then
    echo "[base try=$try] lost race (OOM) on NPU$PICK — retrying" | tee -a "$LOG.race"; continue
  fi
  echo "[base try=$try] non-OOM exit on NPU$PICK — stopping for inspection" | tee -a "$LOG.race"; exit 2
done
echo "[base] exhausted $MAX tries" | tee -a "$LOG.race"; exit 3
