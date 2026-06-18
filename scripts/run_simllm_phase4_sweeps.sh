#!/usr/bin/env bash
set -euo pipefail

# Run Phase 4 Sim-LLM tuning sweeps without modifying benchmark spec JSON.
#
# Usage:
#   bash scripts/run_simllm_phase4_sweeps.sh <same-spec-json>
#
# Optional environment:
#   VLLM_HUST_BENCHMARK_REPO=/path/to/vllm-hust-benchmark
#   VLLM_HUST_REPO=/path/to/vllm-hust
#   RESULT_ROOT=/path/to/output-root
#   SIMLLM_SWEEP_THRESHOLDS="0.5 0.6 0.7 0.8 0.9"
#   SIMLLM_SWEEP_CACHE_SIZES="128 256 512 1024 2048 4096"
#   SIMLLM_SWEEP_RUN_BASELINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ASCEND_REPO_ROOT}/.." && pwd)"

BENCHMARK_REPO="${VLLM_HUST_BENCHMARK_REPO:-${WORKSPACE_ROOT}/vllm-hust-benchmark}"
VLLM_HUST_REPO="${VLLM_HUST_REPO:-${WORKSPACE_ROOT}/vllm-hust}"
RESULT_ROOT="${RESULT_ROOT:-${BENCHMARK_REPO}/.benchmarks/simllm-phase4-sweeps}"
SPEC_INPUT="${1:-${SPEC:-}}"

if [[ -z "${SPEC_INPUT}" ]]; then
  echo "Usage: bash scripts/run_simllm_phase4_sweeps.sh <same-spec-json>" >&2
  exit 2
fi

if [[ ! -d "${BENCHMARK_REPO}" ]]; then
  echo "Benchmark repo not found: ${BENCHMARK_REPO}" >&2
  exit 2
fi

if [[ -f "${SPEC_INPUT}" ]]; then
  SPEC_FILE="$(cd "$(dirname "${SPEC_INPUT}")" && pwd)/$(basename "${SPEC_INPUT}")"
elif [[ -f "${BENCHMARK_REPO}/${SPEC_INPUT}" ]]; then
  SPEC_FILE="${BENCHMARK_REPO}/${SPEC_INPUT}"
else
  echo "Spec file not found: ${SPEC_INPUT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"

PLUGIN_COMMIT="$(git -C "${ASCEND_REPO_ROOT}" rev-parse HEAD)"
CORE_COMMIT="$(git -C "${VLLM_HUST_REPO}" rev-parse HEAD 2>/dev/null || true)"

cat > "${RESULT_ROOT}/README.md" <<EOF
# Sim-LLM Phase 4 Sweeps

- Plugin repo: ${ASCEND_REPO_ROOT}
- Plugin commit: ${PLUGIN_COMMIT}
- Core repo: ${VLLM_HUST_REPO}
- Core commit: ${CORE_COMMIT:-unknown}
- Benchmark repo: ${BENCHMARK_REPO}
- Spec: ${SPEC_FILE}

Each child directory is produced by \`scripts/run-current-ascend-same-spec.sh\`.
EOF

run_case() {
  local run_id="$1"
  shift

  local result_dir="${RESULT_ROOT}/${run_id}"
  local -a env_args=(
    "CURRENT_VLLM_ASCEND_HUST_REPO=${ASCEND_REPO_ROOT}"
    "CURRENT_VLLM_HUST_REPO=${VLLM_HUST_REPO}"
    "CURRENT_PLUGIN_GIT_COMMIT=${PLUGIN_COMMIT}"
    "RESULT_DIR=${result_dir}"
    "RUN_ID=${run_id}"
  )
  env_args+=("$@")

  echo "[INFO] Running ${run_id}"
  (
    cd "${BENCHMARK_REPO}"
    env "${env_args[@]}" \
      bash scripts/run-current-ascend-same-spec.sh "${SPEC_FILE}"
  )
}

if [[ "${SIMLLM_SWEEP_RUN_BASELINE:-1}" == "1" ]]; then
  run_case "baseline-disabled" \
    "VLLM_ASCEND_SIMLLM_ENABLED=0"
fi

read -r -a THRESHOLDS <<< "${SIMLLM_SWEEP_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9}"
for threshold in "${THRESHOLDS[@]}"; do
  label="${threshold//./p}"
  run_case "threshold-${label}" \
    "VLLM_ASCEND_SIMLLM_ENABLED=1" \
    "VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD=${threshold}"
done

read -r -a CACHE_SIZES <<< "${SIMLLM_SWEEP_CACHE_SIZES:-128 256 512 1024 2048 4096}"
for cache_size in "${CACHE_SIZES[@]}"; do
  run_case "cache-size-${cache_size}" \
    "VLLM_ASCEND_SIMLLM_ENABLED=1" \
    "VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE=${cache_size}"
done

echo "[OK] Phase 4 sweeps completed under ${RESULT_ROOT}"
