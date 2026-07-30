#
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

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEANUP_SCRIPT = REPO_ROOT / ".github/workflows/scripts/cleanup_ascend_benchmark_processes.py"
SPEC = importlib.util.spec_from_file_location("cleanup_ascend_benchmark_processes", CLEANUP_SCRIPT)
assert SPEC and SPEC.loader
cleanup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


@pytest.fixture
def context() -> dict[str, str]:
    return {
        "GITHUB_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
        "GITHUB_WORKFLOW": "Ascend Benchmark Leaderboard",
        "GITHUB_JOB": "ascend-benchmark",
        "GITHUB_RUN_ID": "1780",
        "GITHUB_RUN_ATTEMPT": "2",
        "RUNNER_NAME": "ascend-runner-01",
        "RUNNER_WORKSPACE": "/opt/actions-runner/_work",
    }


def create_process(proc_root: Path, pid: int, environment: dict[str, str], *, engine_core: bool = True) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True, exist_ok=True)
    name = "VLLM::EngineCor" if engine_core else "python3"
    (process_dir / "status").write_text(f"Name:\t{name}\n", encoding="utf-8")
    cmdline = b"VLLM::EngineCore\0worker\0" if engine_core else b"python3\0worker.py\0"
    (process_dir / "cmdline").write_bytes(cmdline)
    environ = b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0"
    (process_dir / "environ").write_bytes(environ)


def matching_pids(proc_root: Path, context: dict[str, str], mode: str) -> list[int]:
    return [process.pid for process in cleanup.find_matching_processes(proc_root, context, mode)]


def test_current_mode_matches_only_exact_run(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 101, context)
    create_process(tmp_path, 102, {**context, "GITHUB_RUN_ATTEMPT": "1"})
    create_process(tmp_path, 103, context, engine_core=False)

    assert matching_pids(tmp_path, context, "current") == [101]


def test_stale_mode_matches_previous_run_or_attempt(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 201, context)
    create_process(tmp_path, 202, {**context, "GITHUB_RUN_ID": "1779", "GITHUB_RUN_ATTEMPT": "1"})
    create_process(tmp_path, 203, {**context, "GITHUB_RUN_ATTEMPT": "1"})

    assert matching_pids(tmp_path, context, "stale") == [202, 203]


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_REPOSITORY", "another/repository"),
        ("GITHUB_WORKFLOW", "Another workflow"),
        ("GITHUB_JOB", "another-job"),
        ("RUNNER_NAME", "ascend-runner-02"),
        ("RUNNER_WORKSPACE", "/another/workspace"),
    ],
)
def test_owner_boundary_mismatch_is_not_selected(tmp_path: Path, context: dict[str, str], key: str, value: str) -> None:
    create_process(tmp_path, 301, {**context, key: value})

    assert matching_pids(tmp_path, context, "current") == []
    assert matching_pids(tmp_path, context, "stale") == []


def test_cleanup_escalates_from_term_to_kill(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 401, context)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            shutil.rmtree(tmp_path / str(pid))

    matched, remaining = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        kill=fake_kill,
        sleep=lambda _: None,
    )

    assert matched == [401]
    assert remaining == []
    assert signals == [(401, signal.SIGTERM), (401, signal.SIGKILL)]


def test_cleanup_rescans_ownership_before_kill(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 501, context)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGTERM:
            create_process(tmp_path, pid, {**context, "RUNNER_NAME": "another-runner"})

    matched, remaining = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        kill=fake_kill,
        sleep=lambda _: None,
    )

    assert matched == [501]
    assert remaining == []
    assert signals == [(501, signal.SIGTERM)]


def test_missing_context_fails_closed() -> None:
    result = subprocess.run(
        [sys.executable, str(CLEANUP_SCRIPT), "--mode", "current"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )

    assert result.returncode == 2
    assert "missing required ownership context" in result.stderr
