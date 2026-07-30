#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

mode=${1:?Usage: $0 <current|stale>}
if [[ "$mode" != "current" && "$mode" != "stale" ]]; then
  echo "Unsupported cleanup mode: $mode" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cleanup_script=$SCRIPT_DIR/cleanup_ascend_benchmark_processes.py
root_helper=${ASCEND_BENCHMARK_ROOT_HELPER:-/usr/local/bin/run_ascend_benchmark_root_helper.sh}
use_sudo=${ASCEND_BENCHMARK_USE_SUDO:-auto}
python_bin=${PYTHON_BIN:-$(command -v python3)}

ownership_vars=(
  GITHUB_REPOSITORY
  GITHUB_WORKFLOW
  GITHUB_JOB
  GITHUB_RUN_ID
  GITHUB_RUN_ATTEMPT
  RUNNER_NAME
  RUNNER_WORKSPACE
  VLLM_ASCEND_HUST_REPO
)

for var_name in "${ownership_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required Ascend benchmark cleanup context: $var_name" >&2
    exit 2
  fi
  export "$var_name"
done

if [[ "$use_sudo" == "auto" ]]; then
  if [[ "$(id -u)" == "0" ]]; then
    use_sudo=0
  elif command -v sudo >/dev/null 2>&1 && [[ -x "$root_helper" ]]; then
    use_sudo=1
  else
    use_sudo=0
  fi
fi

cleanup_args=(
  --mode "$mode"
  --term-timeout-seconds "${ASCEND_BENCHMARK_CLEANUP_TERM_TIMEOUT_SECONDS:-10}"
  --kill-timeout-seconds "${ASCEND_BENCHMARK_CLEANUP_KILL_TIMEOUT_SECONDS:-5}"
)

if [[ "$use_sudo" == "1" ]]; then
  preserve_list=$(IFS=,; printf '%s' "${ownership_vars[*]}")
  if ! sudo --preserve-env="$preserve_list" -E -n "$root_helper" \
    cleanup-processes "$cleanup_script" "${cleanup_args[@]}"; then
    echo "Ascend benchmark root cleanup failed. The runner helper may be stale." >&2
    echo "Reinstall it with: sudo RUNNER_USER=${USER:-grunner} bash $VLLM_ASCEND_HUST_REPO/scripts/install_ascend_benchmark_root_helper.sh" >&2
    exit 1
  fi
else
  "$python_bin" "$cleanup_script" "${cleanup_args[@]}"
fi
