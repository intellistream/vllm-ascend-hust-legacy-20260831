#!/bin/bash
# Race-and-retry launcher for a contended NPU.
# Polls the target card; the instant it is free (<10% HBM), launches the probe.
# If the run OOMs within the load window, retries — until it wins the race.
#
# Usage: race_launch.sh <device> <flag_val> <case_name> <logfile> <max_tries>
set -u
DEV="${1:?device}"; FLAG="${2:?graph_compat flag}"; CASE="${3:?case}"; LOG="${4:?log}"; MAX="${5:-12}"
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=/root/vllm-ascend-hust/tools/sew_offload/run_graph_compat_capture_probe.py

free_pct() { npu-smi info -t usages -i "$DEV" -c 0 2>/dev/null | grep "HBM Usage Rate" | grep -oE "[0-9]+$"; }

for try in $(seq 1 "$MAX"); do
  # spin until the card looks free
  for w in $(seq 1 120); do
    u=$(free_pct); u=${u:-100}
    if [ "$u" -lt 10 ]; then break; fi
    sleep 3
  done
  u=$(free_pct); u=${u:-100}
  echo "[race try=$try] NPU$DEV at ${u}% — launching" | tee -a "$LOG.race"
  # Faithful to the validated --ascend-moe-offload-gb command: set ONLY the
  # budget (GB) + the flag under test. autoconfig (now armed via the probe's
  # up-front `import vllm_ascend.patch.platform`) owns ENABLED/NUM_SLOTS/
  # LAYERED_RUNTIME/FANOUT/MAX_PHASES via os.environ.setdefault AND wires the
  # vLLM PrefetchOffloader (device residency). Setting them manually here was
  # what previously activated SEW WITHOUT PrefetchOffloader -> false-negative.
  ASCEND_RT_VISIBLE_DEVICES="$DEV" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ASCEND_MOE_OFFLOAD_GB=14 \
  VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE="$FLAG" \
  "$PY" "$PROBE" --case-name "$CASE" --no-enforce-eager --gpu-memory-utilization 0.90 \
    > "$LOG" 2>&1
  # classify outcome
  if grep -q "OUTPUT_TOKENS\|GENERATE_OK\|LOAD_OK" "$LOG"; then
    echo "[race try=$try] WON — engine loaded" | tee -a "$LOG.race"; exit 0
  fi
  if grep -q "Free memory on device" "$LOG"; then
    echo "[race try=$try] lost race (OOM) — retrying" | tee -a "$LOG.race"; continue
  fi
  # some other failure (e.g. capture crash 107027/107030): that's a REAL result, stop
  echo "[race try=$try] non-OOM exit — stopping for inspection" | tee -a "$LOG.race"; exit 2
done
echo "[race] exhausted $MAX tries" | tee -a "$LOG.race"; exit 3
