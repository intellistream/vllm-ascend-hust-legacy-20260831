#!/bin/bash
set -euo pipefail

COMMIT=${1:-${FORK_POINT:-${GITHUB_SHA:-}}}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
OUTPUT_DIR=${PERFGATE_BASELINE_OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/perfgate-baselines}
ALLOW_BASELINE_FALLBACK=${PERFGATE_ALLOW_BASELINE_FALLBACK:-0}
MODE=${PERFGATE_MODE:-report}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
BENCHMARK_REPO_DIR=${VLLM_HUST_BENCHMARK_REPO:-${GITHUB_WORKSPACE:-$PWD}/vllm-hust-benchmark}
TARGET_REPOSITORY=${PERFGATE_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-vLLM-HUST/vllm-ascend-hust}}
SPEC_FILE=${SAME_SPEC_SPEC_FILE:-${PERFGATE_SPEC_FILE:-}}
FETCH_MAX_ATTEMPTS=${PERFGATE_BASELINE_FETCH_MAX_ATTEMPTS:-4}
FETCH_RETRY_SECONDS=${PERFGATE_BASELINE_FETCH_RETRY_SECONDS:-15}

write_env() {
  local name=$1
  local value=$2
  local delimiter="EOF_${name}_$$_${RANDOM}"
  {
    echo "${name}<<${delimiter}"
    printf '%s\n' "$value"
    echo "$delimiter"
  } >> "$GITHUB_ENV"
}

baseline_unavailable() {
  local reason=$1
  echo "$reason" >&2
  write_env PERFGATE_BASELINE_AVAILABLE 0
  write_env PERFGATE_BASELINE_COMMIT "$COMMIT"
  write_env PERFGATE_BASELINE_SOURCE unavailable
  write_env PERFGATE_BASELINE_UNAVAILABLE_REASON "$reason"
  if [[ "$MODE" == "report" ]]; then
    echo "Perfgate baseline unavailable in report mode; continuing without baseline."
    exit 0
  fi
  echo "Perfgate baseline unavailable in enforce mode; failing."
  exit 2
}

fetch_baseline_branch_with_retry() {
  local attempt=1
  local delay_seconds=$FETCH_RETRY_SECONDS

  while (( attempt <= FETCH_MAX_ATTEMPTS )); do
    echo "Fetching perfgate baseline branch (attempt ${attempt}/${FETCH_MAX_ATTEMPTS}): $BASELINE_BRANCH"
    if git -C "$BENCHMARK_REPO_DIR" fetch --quiet --depth=1 origin \
      "+$BASELINE_BRANCH:refs/remotes/origin/$BASELINE_BRANCH"; then
      return 0
    fi
    if (( attempt < FETCH_MAX_ATTEMPTS )); then
      echo "Perfgate baseline fetch failed; retrying in ${delay_seconds}s." >&2
      sleep "$delay_seconds"
      delay_seconds=$((delay_seconds * 2))
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

if [[ -z "$COMMIT" ]]; then
  echo "Usage: $0 <commit-sha> or set FORK_POINT/GITHUB_SHA" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -d "$BENCHMARK_REPO_DIR/.git" ]]; then
  baseline_unavailable "Benchmark repository checkout is unavailable: $BENCHMARK_REPO_DIR"
fi
if [[ ! "$FETCH_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  baseline_unavailable "PERFGATE_BASELINE_FETCH_MAX_ATTEMPTS must be a positive integer: $FETCH_MAX_ATTEMPTS"
fi
if [[ ! "$FETCH_RETRY_SECONDS" =~ ^[0-9]+$ ]]; then
  baseline_unavailable "PERFGATE_BASELINE_FETCH_RETRY_SECONDS must be a non-negative integer: $FETCH_RETRY_SECONDS"
fi
if ! fetch_baseline_branch_with_retry; then
  baseline_unavailable "Perfgate baseline branch cannot be fetched from benchmark repository: $BASELINE_BRANCH"
fi
baseline_ref="refs/remotes/origin/$BASELINE_BRANCH"
if ! git -C "$BENCHMARK_REPO_DIR" rev-parse --verify "$baseline_ref" >/dev/null 2>&1; then
  baseline_unavailable "Perfgate baseline branch not found in benchmark repository: $BASELINE_BRANCH"
fi
if [[ -z "$SPEC_FILE" || ! -f "$SPEC_FILE" ]]; then
  baseline_unavailable "Perfgate spec file is unavailable: ${SPEC_FILE:-unset}"
fi
spec_id=$(jq -er '.id // empty' "$SPEC_FILE") || baseline_unavailable "Perfgate spec id is missing: $SPEC_FILE"
scenario=${BENCH_SCENARIO:-}
if [[ -z "$scenario" ]]; then
  scenario=$(jq -er '.scenario // empty' "$SPEC_FILE") || baseline_unavailable "Perfgate scenario is missing: $SPEC_FILE"
fi

target_root="baselines/${TARGET_REPOSITORY}/${COMMIT}/${scenario}/${spec_id}"
baseline_file=""
baseline_metadata_file=""
baseline_metadata_local_file=""
baseline_identity_mismatch=""
while IFS= read -r metadata_path; do
  [[ "$metadata_path" == */baseline-metadata.json ]] || continue
  metadata_json=$(git -C "$BENCHMARK_REPO_DIR" show "$baseline_ref:$metadata_path") || continue
  if ! jq -e --arg repo "$TARGET_REPOSITORY" --arg sha "$COMMIT" \
    --arg expected_scenario "$scenario" --arg expected_spec_id "$spec_id" \
    '.identity.target_repository == $repo and .identity.target_sha == $sha and
     .identity.scenario == $expected_scenario and .identity.spec_id == $expected_spec_id and
     (.identity.spec_hash | type == "string" and test("^[0-9a-f]{64}$")) and
     (.artifact.sha256 | type == "string" and test("^[0-9a-f]{64}$"))' \
    <<<"$metadata_json" >/dev/null; then
    continue
  fi
  candidate_dir=${metadata_path%/baseline-metadata.json}
  candidate_artifact_path="$candidate_dir/run_leaderboard.json"
  candidate_artifact="$OUTPUT_DIR/candidate-run-leaderboard.json"
  if ! git -C "$BENCHMARK_REPO_DIR" show "$baseline_ref:$candidate_artifact_path" >"$candidate_artifact"; then
    continue
  fi
  expected_sha=$(jq -er '.artifact.sha256' <<<"$metadata_json") || continue
  actual_sha=$(sha256sum "$candidate_artifact" | awk '{print $1}')
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    baseline_unavailable "Perfgate baseline artifact checksum mismatch: $candidate_artifact_path"
  fi
  if ! jq -e --arg expected_scenario "$scenario" --arg expected_spec_id "$spec_id" \
    --arg expected_spec_hash "$(jq -er '.identity.spec_hash' <<<"$metadata_json")" \
    '.same_spec.scenario == $expected_scenario and
     .same_spec.spec_id == $expected_spec_id and
     .same_spec.resolved_spec_hash == $expected_spec_hash' \
    "$candidate_artifact" >/dev/null; then
    baseline_identity_mismatch="$candidate_artifact_path"
    continue
  fi
  baseline_metadata_local_file="$OUTPUT_DIR/baseline-metadata-${COMMIT:0:8}.json"
  printf '%s\n' "$metadata_json" >"$baseline_metadata_local_file"
  baseline_file="$candidate_artifact"
  baseline_metadata_file="$metadata_path"
  break
done < <(git -C "$BENCHMARK_REPO_DIR" ls-tree -r --name-only "$baseline_ref" -- "$target_root")

baseline_commit="$COMMIT"
baseline_source="exact"
if [[ -z "$baseline_file" ]]; then
  if [[ "$ALLOW_BASELINE_FALLBACK" != "1" ]]; then
    if [[ -n "$baseline_identity_mismatch" ]]; then
      baseline_unavailable "No exact perfgate baseline matched same-spec identity for $TARGET_REPOSITORY@$COMMIT ($scenario/$spec_id); rejected $baseline_identity_mismatch"
    fi
    baseline_unavailable "No exact perfgate baseline found for $TARGET_REPOSITORY@$COMMIT ($scenario/$spec_id)"
  fi
  pointer_path="pointers/${TARGET_REPOSITORY}/${scenario}/${spec_id}/latest-main.json"
  pointer_file="$OUTPUT_DIR/latest-main-pointer.json"
  if ! git -C "$BENCHMARK_REPO_DIR" show "$baseline_ref:$pointer_path" >"$pointer_file"; then
    baseline_unavailable "No exact perfgate baseline found and latest-main pointer is missing: $pointer_path"
  fi
  pointer_repo=$(jq -er '.identity.target_repository // empty' "$pointer_file") || baseline_unavailable "latest-main pointer identity is invalid: $pointer_path"
  pointer_sha=$(jq -er '.identity.target_sha // empty' "$pointer_file") || baseline_unavailable "latest-main pointer identity is invalid: $pointer_path"
  pointer_scenario=$(jq -er '.identity.scenario // empty' "$pointer_file") || baseline_unavailable "latest-main pointer identity is invalid: $pointer_path"
  pointer_spec_id=$(jq -er '.identity.spec_id // empty' "$pointer_file") || baseline_unavailable "latest-main pointer identity is invalid: $pointer_path"
  pointer_spec_hash=$(jq -er '.identity.spec_hash // empty' "$pointer_file") || baseline_unavailable "latest-main pointer identity is invalid: $pointer_path"
  pointer_artifact_sha=$(jq -er '.artifact_sha256 // empty' "$pointer_file") || baseline_unavailable "latest-main pointer checksum is invalid: $pointer_path"
  pointer_artifact_path=$(jq -er '.path // empty' "$pointer_file") || baseline_unavailable "latest-main pointer path is invalid: $pointer_path"
  if [[ "$pointer_repo" != "$TARGET_REPOSITORY" || "$pointer_scenario" != "$scenario" || "$pointer_spec_id" != "$spec_id" || ! "$pointer_sha" =~ ^[0-9a-f]{40}$ || ! "$pointer_spec_hash" =~ ^[0-9a-f]{64}$ || ! "$pointer_artifact_sha" =~ ^[0-9a-f]{64}$ ]]; then
    baseline_unavailable "latest-main pointer identity does not match requested target: $pointer_path"
  fi
  expected_pointer_prefix="baselines/${TARGET_REPOSITORY}/"
  expected_pointer_suffix="/${scenario}/${spec_id}/${pointer_spec_hash}/run_leaderboard.json"
  if [[ "$pointer_artifact_path" != "$expected_pointer_prefix"*"$expected_pointer_suffix" || "$pointer_artifact_path" == /* || "$pointer_artifact_path" == ../* || "$pointer_artifact_path" == */../* || "$pointer_artifact_path" == */.. ]]; then
    baseline_unavailable "latest-main pointer path is outside the expected target root: $pointer_artifact_path"
  fi
  baseline_file="$OUTPUT_DIR/latest-main.json"
  if ! git -C "$BENCHMARK_REPO_DIR" show "$baseline_ref:$pointer_artifact_path" >"$baseline_file"; then
    baseline_unavailable "latest-main pointer artifact is unavailable: $pointer_artifact_path"
  fi
  actual_pointer_sha=$(sha256sum "$baseline_file" | awk '{print $1}')
  if [[ "$actual_pointer_sha" != "$pointer_artifact_sha" ]]; then
    baseline_unavailable "latest-main pointer artifact checksum mismatch: $pointer_artifact_path"
  fi
  pointer_metadata_path="${pointer_artifact_path%/run_leaderboard.json}/baseline-metadata.json"
  pointer_metadata=$(git -C "$BENCHMARK_REPO_DIR" show "$baseline_ref:$pointer_metadata_path") || baseline_unavailable "latest-main pointer metadata is unavailable: $pointer_metadata_path"
  if ! jq -e --arg repo "$TARGET_REPOSITORY" --arg sha "$pointer_sha" \
    --arg expected_scenario "$scenario" --arg expected_spec_id "$spec_id" \
    --arg spec_hash "$pointer_spec_hash" --arg artifact_sha "$pointer_artifact_sha" \
    '.identity.target_repository == $repo and .identity.target_sha == $sha and
     .identity.scenario == $expected_scenario and .identity.spec_id == $expected_spec_id and
     .identity.spec_hash == $spec_hash and .artifact.sha256 == $artifact_sha' \
    <<<"$pointer_metadata" >/dev/null; then
    baseline_unavailable "latest-main pointer metadata does not match the referenced artifact: $pointer_metadata_path"
  fi
  if ! jq -e --arg expected_scenario "$scenario" --arg expected_spec_id "$spec_id" \
    --arg expected_spec_hash "$pointer_spec_hash" \
    '.same_spec.scenario == $expected_scenario and
     .same_spec.spec_id == $expected_spec_id and
     .same_spec.resolved_spec_hash == $expected_spec_hash' \
    "$baseline_file" >/dev/null; then
    baseline_unavailable "latest-main pointer artifact does not match same-spec identity: $pointer_artifact_path"
  fi
  baseline_metadata_local_file="$OUTPUT_DIR/baseline-metadata-latest-main.json"
  printf '%s\n' "$pointer_metadata" >"$baseline_metadata_local_file"
  baseline_metadata_file="$pointer_metadata_path"
  baseline_commit="latest-main"
  baseline_source="latest-main-fallback"
fi

if [[ ! -f "$baseline_file" ]]; then
  baseline_unavailable "No perfgate baseline found for $TARGET_REPOSITORY@$COMMIT ($scenario/$spec_id)"
fi

resolved_file="$OUTPUT_DIR/baseline-${COMMIT:0:8}.json"
cp "$baseline_file" "$resolved_file"
write_env PERFGATE_BASELINE_FILE "$resolved_file"
write_env PERFGATE_BASELINE_AVAILABLE 1
write_env PERFGATE_BASELINE_COMMIT "$baseline_commit"
write_env PERFGATE_BASELINE_SOURCE "$baseline_source"
write_env PERFGATE_BASELINE_REPOSITORY_COMMIT "$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$baseline_ref")"
if [[ -n "$baseline_metadata_file" ]]; then
  write_env PERFGATE_BASELINE_METADATA_PATH "$baseline_metadata_file"
fi
if [[ -n "$baseline_metadata_local_file" ]]; then
  write_env PERFGATE_BASELINE_METADATA_FILE "$baseline_metadata_local_file"
fi

echo "Fetched perfgate baseline: $baseline_commit ($baseline_source) -> $resolved_file"
