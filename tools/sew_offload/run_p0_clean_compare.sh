#!/usr/bin/env bash
# P0-clean: B1 (captured offload) vs eager single-op offload, with NO diagnostic
# probes. The earlier P0 B1 run carried SEW_SEAM_PROBE=1, which triggers a
# torch.npu.synchronize() bracket inside the seam on every staging call -> it
# inflated B1's latency (esp. prefill) and was NOT present in the eager run.
# Here neither run sets any probe, so the synchronize confound is gone and the
# only variable left is graph capture (enforce_eager). Identical offload
# footprint: NUM_SLOTS=96, nonres={2,3,4,5}, ASYNC_LOAD=0.
set -u
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
PROBE=tools/sew_offload/run_graph_compat_capture_probe.py
DEV="${DEV:?set DEV}"
NLAYERS="${NLAYERS:-48}"
LOGDIR=.planning/sew_offload/logs
MAXTOK="${MAXTOK:-32}"
REPS="${REPS:-5}"
NONRES="${NONRES:-2,3,4,5}"
resident=$("$PY" -c "nr=set(int(x) for x in '$NONRES'.split(',')); print(','.join(str(i) for i in range($NLAYERS) if i not in nr))")

common_offload_env() {  # shared between both runs (identical footprint)
  export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
  export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=96
  export VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD=96
  export VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1
  export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
  export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
  export VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0
  export VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS="$resident"
}

run_b1_clean() {
  local log="$LOGDIR/P0clean_b1_slots96.log"
  echo "[p0-clean] B1 captured offload (no probe) -> $log"
  ( common_offload_env
    export VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=1
    export VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=1
    ASCEND_RT_VISIBLE_DEVICES="$DEV" "$PY" "$PROBE" --case-name p0clean_b1 \
      --no-enforce-eager --gpu-memory-utilization 0.90 --max-tokens "$MAXTOK" \
      --latency --latency-repeats "$REPS" ) > "$log" 2>&1
  grep -E "LATENCY case=|Loading model weights took|CASE_FAILED|out of memory" "$log" | head
}

run_eager_clean() {
  local log="$LOGDIR/P0clean_eager_slots96.log"
  echo "[p0-clean] eager single-op offload (no probe) -> $log"
  ( common_offload_env
    export VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=0
    export VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=0
    ASCEND_RT_VISIBLE_DEVICES="$DEV" "$PY" "$PROBE" --case-name p0clean_eager \
      --enforce-eager --gpu-memory-utilization 0.90 --max-tokens "$MAXTOK" \
      --latency --latency-repeats "$REPS" ) > "$log" 2>&1
  grep -E "LATENCY case=|Loading model weights took|CASE_FAILED|out of memory" "$log" | head
}

run_b1_clean
run_eager_clean
echo "[p0-clean] DONE"
