# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
STAGE2_SCRIPT = SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh"
FETCH_BASELINE_SCRIPT = SCRIPT_DIR / "perfgate_fetch_baseline.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_commands(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(parents=True)
    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
set -euo pipefail
case "${1:-}" in
  rev-parse)
    if [[ "${2:-}" == "--verify" ]]; then
      exit 1
    fi
    if [[ "${2:-}" == "HEAD" ]]; then
      echo "original-ref"
    else
      echo "${FAKE_M2_COMMIT}"
    fi
    ;;
  ls-remote)
    exit "${FAKE_LS_REMOTE_RC:-0}"
    ;;
  rebase)
    exit "${FAKE_REBASE_RC:-0}"
    ;;
  diff)
    echo "conflicting-file.py"
    ;;
  *)
    exit 0
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "bash",
        """#!/bin/bash
set -euo pipefail
target=${1:-}
printf '%s\n' "$target" >> "${FAKE_BASH_LOG}"
case "$target" in
  *install_ascend_benchmark_with_dev_hub.sh)
    exit 0
    ;;
  *run_ascend_benchmark_ci.sh)
    mkdir -p "${RESULT_ROOT}/submissions/${RUN_ID}"
    printf '{}\n' > "${RESULT_ROOT}/submissions/${RUN_ID}/run_leaderboard.json"
    exit 0
    ;;
  *perfgate_fetch_baseline.sh)
    if [[ "${FAKE_FETCH_AVAILABLE}" == "1" ]]; then
      printf '{}\n' > "${FAKE_BASELINE_FILE}"
      {
        echo "PERFGATE_BASELINE_AVAILABLE=1"
        echo "PERFGATE_BASELINE_FILE=${FAKE_BASELINE_FILE}"
        echo "PERFGATE_BASELINE_COMMIT=${FAKE_M2_COMMIT}"
        echo "PERFGATE_BASELINE_SOURCE=exact"
      } > "${GITHUB_ENV}"
    else
      {
        echo "PERFGATE_BASELINE_AVAILABLE=0"
        echo "PERFGATE_BASELINE_COMMIT=${FAKE_M2_COMMIT}"
        echo "PERFGATE_BASELINE_SOURCE=unavailable"
        echo "PERFGATE_BASELINE_UNAVAILABLE_REASON=No exact M2 baseline"
      } > "${GITHUB_ENV}"
    fi
    exit "${FAKE_FETCH_RC:-0}"
    ;;
  *)
    echo "Unexpected bash target: $target" >&2
    exit 99
    ;;
esac
""",
    )
    return fake_bin


def _stage2_env(
    tmp_path: Path,
    *,
    mode: str = "enforce",
    fork_point: str = "m2-commit",
    rebase_rc: str = "0",
    fetch_available: str = "1",
    fetch_rc: str = "0",
) -> dict[str, str]:
    fake_bin = _prepare_fake_commands(tmp_path)
    result_root = tmp_path / "stage2-result"
    baseline_file = tmp_path / "m2-baseline.json"
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PERFGATE_MODE": mode,
        "FORK_POINT": fork_point,
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUNNER_TEMP": str(tmp_path),
        "PERFGATE_STAGE2_RESULT_ROOT": str(result_root),
        "PERFGATE_STAGE2_RUN_ID": "test-stage2",
        "FAKE_M2_COMMIT": "m2-commit",
        "FAKE_REBASE_RC": rebase_rc,
        "FAKE_FETCH_AVAILABLE": fetch_available,
        "FAKE_FETCH_RC": fetch_rc,
        "FAKE_BASELINE_FILE": str(baseline_file),
        "FAKE_BASH_LOG": str(tmp_path / "bash.log"),
        "PYTHON_BIN": "",
    }


def _run_stage2(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(STAGE2_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_stage2_revalidates_latest_main_in_enforce_mode(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path)

    result = _run_stage2(env)

    assert result.returncode == 0
    assert "required revalidation" in result.stdout
    assert "run_ascend_benchmark_ci.sh" in Path(env["FAKE_BASH_LOG"]).read_text(
        encoding="utf-8"
    )
    github_env = Path(env["GITHUB_ENV"]).read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_EXECUTED" in github_env
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env


def test_stage2_rebase_conflict_fails_only_in_enforce_mode(tmp_path: Path) -> None:
    enforce_env = _stage2_env(
        tmp_path / "enforce",
        fork_point="fork-point",
        rebase_rc="1",
    )
    report_env = _stage2_env(
        tmp_path / "report",
        mode="report",
        fork_point="fork-point",
        rebase_rc="1",
    )

    enforce_result = _run_stage2(enforce_env)
    report_result = _run_stage2(report_env)

    assert enforce_result.returncode == 2
    assert report_result.returncode == 0
    assert "rebase conflict recorded" in enforce_result.stdout


def test_stage2_missing_m2_baseline_preserves_reason_and_fails(
    tmp_path: Path,
) -> None:
    env = _stage2_env(
        tmp_path,
        fetch_available="0",
        fetch_rc="2",
    )

    result = _run_stage2(env)

    assert result.returncode == 2
    github_env = Path(env["GITHUB_ENV"]).read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env
    assert "No exact M2 baseline" in github_env


def test_fetch_baseline_preserves_reason_in_enforce_mode(tmp_path: Path) -> None:
    fake_bin = _prepare_fake_commands(tmp_path)
    github_env = tmp_path / "fetch-github-env"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PERFGATE_MODE": "enforce",
        "FORK_POINT": "missing-commit",
        "GITHUB_ENV": str(github_env),
        "FAKE_LS_REMOTE_RC": "1",
        "FAKE_M2_COMMIT": "m2-commit",
        "FAKE_FETCH_AVAILABLE": "0",
        "FAKE_BASH_LOG": str(tmp_path / "bash.log"),
    }

    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 2
    written_env = github_env.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_AVAILABLE" in written_env
    assert "Benchmark repository checkout is unavailable" in written_env


def _prepare_central_baseline_repo(tmp_path: Path, *, target_sha: str) -> tuple[Path, Path]:
    remote = tmp_path / "central.git"
    worktree = tmp_path / "central-worktree"
    benchmark_repo = tmp_path / "benchmark"
    spec_file = tmp_path / "spec.json"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(worktree))
    _git(worktree, "config", "user.email", "test@example.com")
    _git(worktree, "config", "user.name", "Test")
    _git(worktree, "switch", "-c", "benchmark-baselines")

    spec_id = "perfgate-ascend-qwen25-3b-910b2"
    scenario = "random-online"
    spec_hash = "a" * 64
    target_repo = "vLLM-HUST/vllm-ascend-hust"
    artifact = (
        f'{{"same_spec":{{"scenario":"random-online","spec_id":"{spec_id}",'
        f'"resolved_spec_hash":"{spec_hash}"}}}}\n'
    )
    artifact_path = (
        worktree
        / "baselines"
        / target_repo
        / target_sha
        / scenario
        / spec_id
        / spec_hash
        / "run_leaderboard.json"
    )
    metadata_path = artifact_path.with_name("baseline-metadata.json")
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(artifact, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "identity": {
                    "target_repository": target_repo,
                    "target_sha": target_sha,
                    "scenario": scenario,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                },
                "artifact": {"sha256": hashlib.sha256(artifact.encode()).hexdigest()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(worktree, "add", ".")
    _git(worktree, "commit", "-m", "test baseline")
    _git(worktree, "remote", "add", "origin", str(remote))
    _git(worktree, "push", "origin", "benchmark-baselines")
    _git(tmp_path, "clone", str(remote), str(benchmark_repo))
    spec_file.write_text(
        json.dumps({"id": spec_id, "scenario": scenario}) + "\n", encoding="utf-8"
    )
    return benchmark_repo, spec_file


def _install_transient_fetch_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "transient-fetch-bin"
    fake_bin.mkdir()
    counter = tmp_path / "fetch-attempts"
    real_git = shutil.which("git")
    assert real_git is not None
    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
set -euo pipefail
is_fetch=0
for arg in "$@"; do
  if [[ "$arg" == "fetch" ]]; then
    is_fetch=1
    break
  fi
done
if [[ "$is_fetch" == "1" ]]; then
  attempt=0
  if [[ -f "$FETCH_ATTEMPT_COUNTER" ]]; then
    attempt=$(<"$FETCH_ATTEMPT_COUNTER")
  fi
  attempt=$((attempt + 1))
  printf '%s\n' "$attempt" > "$FETCH_ATTEMPT_COUNTER"
  if (( attempt < FETCH_SUCCEED_ON_ATTEMPT )); then
    echo "fatal: unable to access remote: simulated transient timeout" >&2
    exit 128
  fi
fi
exec "$REAL_GIT" "$@"
""",
    )
    return fake_bin, counter


def test_fetch_baseline_reads_central_nested_exact_artifact(tmp_path: Path) -> None:
    target_sha = "b" * 40
    benchmark_repo, spec_file = _prepare_central_baseline_repo(tmp_path, target_sha=target_sha)
    github_env = tmp_path / "github-env"
    output_dir = tmp_path / "output"
    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT), target_sha],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_ENV": str(github_env),
            "PERFGATE_MODE": "enforce",
            "VLLM_HUST_BENCHMARK_REPO": str(benchmark_repo),
            "SAME_SPEC_SPEC_FILE": str(spec_file),
            "BENCH_SCENARIO": "random-online",
            "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
            "PERFGATE_BASELINE_OUTPUT_DIR": str(output_dir),
        },
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / f"baseline-{target_sha[:8]}.json").is_file()
    assert "PERFGATE_BASELINE_AVAILABLE" in github_env.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_METADATA_PATH" in github_env.read_text(encoding="utf-8")


def test_fetch_baseline_retries_transient_fetch_failure(tmp_path: Path) -> None:
    target_sha = "1" * 40
    benchmark_repo, spec_file = _prepare_central_baseline_repo(tmp_path, target_sha=target_sha)
    fake_bin, counter = _install_transient_fetch_wrapper(tmp_path)
    github_env = tmp_path / "github-env"
    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT), target_sha],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "FETCH_ATTEMPT_COUNTER": str(counter),
            "FETCH_SUCCEED_ON_ATTEMPT": "3",
            "GITHUB_ENV": str(github_env),
            "PERFGATE_MODE": "enforce",
            "PERFGATE_BASELINE_FETCH_MAX_ATTEMPTS": "4",
            "PERFGATE_BASELINE_FETCH_RETRY_SECONDS": "0",
            "VLLM_HUST_BENCHMARK_REPO": str(benchmark_repo),
            "SAME_SPEC_SPEC_FILE": str(spec_file),
            "BENCH_SCENARIO": "random-online",
            "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
            "PERFGATE_BASELINE_OUTPUT_DIR": str(tmp_path / "output"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert counter.read_text(encoding="utf-8").strip() == "3"
    assert "PERFGATE_BASELINE_AVAILABLE" in github_env.read_text(encoding="utf-8")


def test_fetch_baseline_fails_closed_after_fetch_retries(tmp_path: Path) -> None:
    target_sha = "2" * 40
    benchmark_repo, spec_file = _prepare_central_baseline_repo(tmp_path, target_sha=target_sha)
    fake_bin, counter = _install_transient_fetch_wrapper(tmp_path)
    github_env = tmp_path / "github-env"
    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT), target_sha],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "FETCH_ATTEMPT_COUNTER": str(counter),
            "FETCH_SUCCEED_ON_ATTEMPT": "99",
            "GITHUB_ENV": str(github_env),
            "PERFGATE_MODE": "enforce",
            "PERFGATE_BASELINE_FETCH_MAX_ATTEMPTS": "3",
            "PERFGATE_BASELINE_FETCH_RETRY_SECONDS": "0",
            "VLLM_HUST_BENCHMARK_REPO": str(benchmark_repo),
            "SAME_SPEC_SPEC_FILE": str(spec_file),
            "BENCH_SCENARIO": "random-online",
            "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
            "PERFGATE_BASELINE_OUTPUT_DIR": str(tmp_path / "output"),
        },
    )

    assert result.returncode == 2
    assert counter.read_text(encoding="utf-8").strip() == "3"
    env_text = github_env.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_AVAILABLE" in env_text
    assert "cannot be fetched" in env_text


def test_fetch_baseline_rejects_nested_metadata_identity_mismatch(tmp_path: Path) -> None:
    target_sha = "c" * 40
    benchmark_repo, spec_file = _prepare_central_baseline_repo(tmp_path, target_sha="d" * 40)
    github_env = tmp_path / "github-env"
    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT), target_sha],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_ENV": str(github_env),
            "PERFGATE_MODE": "enforce",
            "VLLM_HUST_BENCHMARK_REPO": str(benchmark_repo),
            "SAME_SPEC_SPEC_FILE": str(spec_file),
            "BENCH_SCENARIO": "random-online",
            "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
            "PERFGATE_BASELINE_OUTPUT_DIR": str(tmp_path / "output"),
        },
    )
    assert result.returncode == 2
    assert "No exact perfgate baseline found" in github_env.read_text(encoding="utf-8")


def test_fetch_baseline_explicit_fallback_accepts_different_main_sha(tmp_path: Path) -> None:
    main_sha = "e" * 40
    fork_sha = "f" * 40
    benchmark_repo, spec_file = _prepare_central_baseline_repo(tmp_path, target_sha=main_sha)
    worktree = tmp_path / "pointer-worktree"
    _git(tmp_path, "clone", str(tmp_path / "central.git"), str(worktree))
    _git(worktree, "switch", "benchmark-baselines")
    spec_id = "perfgate-ascend-qwen25-3b-910b2"
    scenario = "random-online"
    spec_hash = "a" * 64
    target_repo = "vLLM-HUST/vllm-ascend-hust"
    artifact_path = (
        f"baselines/{target_repo}/{main_sha}/{scenario}/{spec_id}/{spec_hash}/run_leaderboard.json"
    )
    artifact = worktree / artifact_path
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pointer = worktree / f"pointers/{target_repo}/{scenario}/{spec_id}/latest-main.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "perfgate-baseline/v1",
                "identity": {
                    "target_repository": target_repo,
                    "target_sha": main_sha,
                    "scenario": scenario,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                },
                "path": artifact_path,
                "artifact_sha256": artifact_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(worktree, "add", ".")
    _git(worktree, "commit", "-m", "test latest-main pointer")
    _git(worktree, "push", "origin", "benchmark-baselines")

    github_env = tmp_path / "github-env"
    output_dir = tmp_path / "output"
    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT), fork_sha],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_ENV": str(github_env),
            "PERFGATE_MODE": "enforce",
            "PERFGATE_ALLOW_BASELINE_FALLBACK": "1",
            "VLLM_HUST_BENCHMARK_REPO": str(benchmark_repo),
            "SAME_SPEC_SPEC_FILE": str(spec_file),
            "BENCH_SCENARIO": scenario,
            "PERFGATE_TARGET_REPOSITORY": target_repo,
            "PERFGATE_BASELINE_OUTPUT_DIR": str(output_dir),
        },
    )
    assert result.returncode == 0, result.stderr
    env_text = github_env.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_SOURCE" in env_text
    assert "latest-main-fallback" in env_text


def _run_compare_with_fake_python(
    tmp_path: Path,
    report: str,
    python_rc: str,
    extra_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        f"""#!/bin/bash
set -euo pipefail
report_file=""
while (( $# > 0 )); do
  if [[ "$1" == "--report-file" ]]; then
    report_file=$2
    break
  fi
  shift
done
printf '%s\\n' '{report}' > "$report_file"
exit {python_rc}
""",
    )
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{}\n", encoding="utf-8")
    current.write_text("{}\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "PERFGATE_MODE": "enforce",
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline),
        "PERFGATE_STAGE1_CURRENT_FILE": str(current),
        "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        **extra_env,
    }
    return subprocess.run(
        ["/bin/bash", str(SCRIPT_DIR / "perfgate_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_final_compare_reports_rebase_conflict_and_fails(tmp_path: Path) -> None:
    conflict_file = tmp_path / "rebase-conflict.txt"
    conflict_file.write_text("CONFLICT (content): workflow.yml\n", encoding="utf-8")

    result = _run_compare_with_fake_python(
        tmp_path,
        "**Overall: FAIL**\n**Stage 2: FAIL**",
        "1",
        {
            "PERFGATE_STAGE2_REBASE_CONFLICT": "1",
            "PERFGATE_STAGE2_REBASE_CONFLICT_FILE": str(conflict_file),
        },
    )

    assert result.returncode == 1
    github_env = Path(tmp_path / "github-env").read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_COMPLETED" in github_env
    assert "PERFGATE_STAGE2_RESULT" in github_env


def test_final_compare_reports_stage2_not_run_and_fails(tmp_path: Path) -> None:
    result = _run_compare_with_fake_python(
        tmp_path,
        "**Overall: FAIL**\n**Stage 2: NOT RUN**",
        "1",
        {"PERFGATE_STAGE2_NOT_RUN_REASON": "Stage 1 did not pass"},
    )

    assert result.returncode == 1
    github_env = Path(tmp_path / "github-env").read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_COMPLETED" in github_env
    assert "PERFGATE_STAGE2_RESULT" in github_env
