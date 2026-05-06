#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SPEC_FILE=${1:-"$BENCHMARK_ROOT/tests/vllm-hust-goal-baseline.json"}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-"$BENCHMARK_ROOT/tests/vllm-hust-goal-constraints.stub.json"}
WORKSPACE_ROOT=${VLLM_HUST_WORKSPACE_ROOT:-$(cd "$BENCHMARK_ROOT/../../.." && pwd)}
BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-"$WORKSPACE_ROOT/vllm-hust-benchmark"}
GOAL_BASELINE_ENV_PREFIX=${GOAL_BASELINE_ENV_PREFIX:-}
RESULT_DIR=${RESULT_DIR:-"$BENCHMARK_ROOT/results/vllm-hust-goal-baseline"}
RUN_ID=${RUN_ID:-"official-ascend-jan-2026-$(date -u +%Y%m%dT%H%M%SZ)"}
PYTHON_RUN_CWD=${GOAL_BASELINE_RUN_CWD:-"/tmp"}

if [[ -z "$GOAL_BASELINE_ENV_PREFIX" ]]; then
  echo "GOAL_BASELINE_ENV_PREFIX is required" >&2
  exit 2
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Spec file not found: $SPEC_FILE" >&2
  exit 2
fi

if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "Constraints stub not found: $CONSTRAINTS_FILE" >&2
  exit 2
fi

if [[ ! -d "$BENCHMARK_REPO/src" ]]; then
  echo "Benchmark repo not found: $BENCHMARK_REPO" >&2
  exit 2
fi

if [[ ! -d "$PYTHON_RUN_CWD" ]]; then
  echo "Runner working directory not found: $PYTHON_RUN_CWD" >&2
  exit 2
fi

json2args() {
  local json_string=$1
  echo "$json_string" | jq -r '
    to_entries |
    map(if (.value | tostring) == "" then "--" + (.key | gsub("_"; "-")) else "--" + (.key | gsub("_"; "-")) + " " + (.value | tostring) end) |
    join(" ")
  '
}

wait_for_server() {
  local host=$1
  local port=$2
  local waited=0
  local timeout_sec=${READY_TIMEOUT_SECONDS:-300}

  while (( waited < timeout_sec )); do
    if curl -s "http://${host}:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 1
    ((waited++))
  done

  echo "Timed out waiting for official baseline server at ${host}:${port}" >&2
  return 1
}

kill_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
}

trap kill_server EXIT

mkdir -p "$RESULT_DIR"

SCENARIO=$(jq -r '.scenario' "$SPEC_FILE")
MODEL=$(jq -r '.model' "$SPEC_FILE")
MODEL_PARAMETERS=$(jq -r '.model_parameters' "$SPEC_FILE")
MODEL_PRECISION=$(jq -r '.model_precision' "$SPEC_FILE")
HARDWARE_VENDOR=$(jq -r '.hardware_vendor' "$SPEC_FILE")
HARDWARE_CHIP_MODEL=$(jq -r '.hardware_chip_model' "$SPEC_FILE")
CHIP_COUNT=$(jq -r '.chip_count' "$SPEC_FILE")
NODE_COUNT=$(jq -r '.node_count' "$SPEC_FILE")
ENGINE=$(jq -r '.export.engine' "$SPEC_FILE")
ENGINE_VERSION=$(jq -r '.export.engine_version' "$SPEC_FILE")
SUBMITTER=$(jq -r '.export.submitter' "$SPEC_FILE")
BASELINE_ENGINE=$(jq -r '.export.baseline_engine' "$SPEC_FILE")
GITHUB_REPOSITORY=$(jq -r '.export.github_repository' "$SPEC_FILE")
GITHUB_REF=$(jq -r '.export.github_ref' "$SPEC_FILE")
GIT_COMMIT=$(jq -r '.export.git_commit' "$SPEC_FILE")
DATA_SOURCE=$(jq -r '.export.data_source' "$SPEC_FILE")
SERVER_HOST=$(jq -r '.server_parameters.host' "$SPEC_FILE")
SERVER_PORT=$(jq -r '.server_parameters.port' "$SPEC_FILE")
CLIENT_HOST=$(jq -r '.client_parameters.host' "$SPEC_FILE")
CLIENT_PORT=$(jq -r '.client_parameters.port' "$SPEC_FILE")
INPUT_LEN=$(jq -r '.client_parameters.input_len // .client_parameters.random_input_len' "$SPEC_FILE")
OUTPUT_LEN=$(jq -r '.client_parameters.output_len // .client_parameters.random_output_len' "$SPEC_FILE")

SERVER_ARGS=$(json2args "$(jq -c --arg model "$MODEL" '.server_parameters + {model: $model}' "$SPEC_FILE")")
CLIENT_ARGS=$(json2args "$(jq -c --arg model "$MODEL" '
  (.client_parameters + {model: $model})
  | if .dataset_name == "random" then
      . + {
        random_input_len: (.random_input_len // .input_len),
        random_output_len: (.random_output_len // .output_len)
      }
      | del(.input_len, .output_len)
    else
      .
    end
' "$SPEC_FILE")")

RAW_RESULT_FILE="$RESULT_DIR/raw_benchmark_result.json"
ARTIFACT_DIR="$RESULT_DIR/submission"

SERVER_COMMAND="cd $PYTHON_RUN_CWD && conda run -p $GOAL_BASELINE_ENV_PREFIX python -m vllm.entrypoints.openai.api_server $SERVER_ARGS"
CLIENT_COMMAND="cd $PYTHON_RUN_CWD && conda run -p $GOAL_BASELINE_ENV_PREFIX vllm bench serve --save-result --result-dir $RESULT_DIR --result-filename $(basename "$RAW_RESULT_FILE") $CLIENT_ARGS"

echo "[goal-baseline] server command: $SERVER_COMMAND"
bash -lc "$SERVER_COMMAND" &
SERVER_PID=$!

wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"

echo "[goal-baseline] client command: $CLIENT_COMMAND"
bash -lc "$CLIENT_COMMAND"

PYTHONPATH="$BENCHMARK_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
conda run -p "$GOAL_BASELINE_ENV_PREFIX" \
bash -lc "cd '$PYTHON_RUN_CWD' && python -m vllm_hust_benchmark.cli export-leaderboard-artifact \
  "$SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$CONSTRAINTS_FILE" \
  --output-dir "$ARTIFACT_DIR" \
  --run-id "$RUN_ID" \
  --engine "$ENGINE" \
  --engine-version "$ENGINE_VERSION" \
  --model-name "$MODEL" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "$SUBMITTER" \
  --baseline-engine "$BASELINE_ENGINE" \
  --data-source "$DATA_SOURCE" \
  --input-length "$INPUT_LEN" \
  --output-length "$OUTPUT_LEN" \
  --git-commit "$GIT_COMMIT" \
  --github-repository "$GITHUB_REPOSITORY" \
  --github-ref "$GITHUB_REF""

echo "[goal-baseline] exported leaderboard artifact to $ARTIFACT_DIR"