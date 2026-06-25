#!/bin/bash
# Race-and-retry launcher for the SEW-only EAGER control (token-correctness).
#
# DIFFERENCE vs race_launch.sh: we do NOT set VLLM_ASCEND_MOE_OFFLOAD_GB, so
# autoconfig's apply_moe_offload_defaults() returns early (target_offload_gb<=0)
# and the vLLM PrefetchOffloader is NEVER wired -> get_offloader()=NoopOffloader.
# This removes the data-plane copy_stream entirely (no 107024/107025). We set the
# SEW fixed-slot env vars DIRECTLY (autoconfig is inert here). With
# GRAPH_COMPATIBLE=1 the SEW capture-safe path avoids torch.unique(...).cpu()
# (no 107027/107030). Net: this isolates whether the SEW control-plane primitives
# alone are ACLGraph-capturable.
#
# Token-correctness at replay is a SEPARATE milestone (needs a staging hook or a
# full-residency identity load); this harness validates capture-pass (LOAD_OK).
#
# FANOUT_THRESHOLD=128 (>= top_k=8) forces the pre-capture eager profile/warmup
# run onto SLOT_CACHE_PATH (reads NPU slot tensors), avoiding FULL_WEIGHT_PATH
# reading the CPU-staged w13/w2 (which would crash before capture is reached).
#
# Usage: race_launch_sew_only.sh <device> <flag_val> <case_name> <logfile> <max_tries> <resident_csv>
set -u
DEV="${1:?device}"; FLAG="${2:?graph_compat flag}"; CASE="${3:?case}"; LOG="${4:?log}"; MAX="${5:-12}"
RESIDENT="${6:?resident_layer_ids csv}"
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=/root/vllm-ascend-hust/tools/sew_offload/run_graph_compat_capture_probe.py

free_pct() { npu-smi info -t usages -i "$DEV" -c 0 2>/dev/null | grep "HBM Usage Rate" | grep -oE "[0-9]+$"; }

# HBM usage threshold below which a fresh full-residency engine can actually fit.
# Qwen3-30B-A3B full load is ~56.9 GB (~87% of 65536 MB), so a new process needs
# the card almost empty. Require <15% before launching.
FREE_THRESHOLD="${FREE_THRESHOLD:-15}"

for try in $(seq 1 "$MAX"); do
  # Wait for a genuine free window. NEVER fall through to a doomed high-util
  # launch: if the window does not open within the budget, skip to the next try
  # instead of launching into an OOM (which only churns the log with stacks).
  got_window=0
  for w in $(seq 1 120); do
    u=$(free_pct); u=${u:-100}
    if [ "$u" -lt "$FREE_THRESHOLD" ]; then got_window=1; break; fi
    sleep 3
  done
  if [ "$got_window" -eq 0 ]; then
    u=$(free_pct); u=${u:-100}
    echo "[race-sew-eager try=$try] no free window (NPU$DEV at ${u}%, need <${FREE_THRESHOLD}%) — skipping launch, retrying" | tee -a "$LOG.race"
    continue
  fi
  echo "[race-sew-eager try=$try] NPU$DEV at ${u}% (<${FREE_THRESHOLD}%) — launching (SEW-only, NoopOffloader)" | tee -a "$LOG.race"
  ASCEND_RT_VISIBLE_DEVICES="$DEV" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
  VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=128 \
  VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD=128 \
  VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1 \
  VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1 \
  VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline \
  VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0 \
  VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$RESIDENT" \
  VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE="$FLAG" \
  "$PY" "$PROBE" --case-name "$CASE" --enforce-eager --gpu-memory-utilization 0.90 ${EXTRA_PROBE_ARGS:-} \
    > "$LOG" 2>&1
  if grep -q "OUTPUT_TOKENS\|GENERATE_OK\|LOAD_OK" "$LOG"; then
    echo "[race-sew-eager try=$try] WON — engine loaded" | tee -a "$LOG.race"; exit 0
  fi
  if grep -qE "Free memory on device|NPU out of memory|HBM out of memory" "$LOG"; then
    echo "[race-sew-eager try=$try] lost race (OOM) — retrying" | tee -a "$LOG.race"; continue
  fi
  echo "[race-sew-eager try=$try] non-OOM exit — stopping for inspection" | tee -a "$LOG.race"; exit 2
done
echo "[race-sew] exhausted $MAX tries" | tee -a "$LOG.race"; exit 3
