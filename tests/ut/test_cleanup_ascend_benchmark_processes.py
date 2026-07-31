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
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEANUP_SCRIPT = REPO_ROOT / ".github/workflows/scripts/cleanup_ascend_benchmark_processes.py"
CLEANUP_WRAPPER = REPO_ROOT / ".github/workflows/scripts/cleanup_ascend_benchmark_processes.sh"
ROOT_HELPER = REPO_ROOT / ".github/workflows/scripts/run_ascend_benchmark_root_helper.sh"
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


def create_process(
    proc_root: Path,
    pid: int,
    environment: dict[str, str],
    *,
    engine_core: bool = True,
    start_time: int | None = None,
    process_group: int | None = None,
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True, exist_ok=True)
    name = "VLLM::EngineCor" if engine_core else "python3"
    (process_dir / "status").write_text(f"Name:\t{name}\n", encoding="utf-8")
    cmdline = b"VLLM::EngineCore\0worker\0" if engine_core else b"python3\0worker.py\0"
    (process_dir / "cmdline").write_bytes(cmdline)
    environ = b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0"
    (process_dir / "environ").write_bytes(environ)
    stat_suffix = [
        "S",
        "0",
        str(process_group if process_group is not None else pid),
        *("0" for _ in range(16)),
        str(start_time if start_time is not None else pid * 100),
    ]
    (process_dir / "stat").write_text(f"{pid} ({name}) {' '.join(stat_suffix)}\n", encoding="utf-8")


def matching_pids(proc_root: Path, context: dict[str, str], mode: str) -> list[int]:
    return [process.pid for process in cleanup.find_matching_processes(proc_root, context, mode)]


def wrapper_environment(context: dict[str, str], tmp_path: Path) -> dict[str, str]:
    proc_root = tmp_path / "proc"
    proc_root.mkdir(exist_ok=True)
    return {
        **os.environ,
        **context,
        "PYTHON_BIN": sys.executable,
        "VLLM_ASCEND_HUST_REPO": str(REPO_ROOT),
        "ASCEND_BENCHMARK_CLEANUP_PROC_ROOT": str(proc_root),
        "ASCEND_BENCHMARK_CLEANUP_TERM_TIMEOUT_SECONDS": "0",
        "ASCEND_BENCHMARK_CLEANUP_KILL_TIMEOUT_SECONDS": "0",
    }


def install_fake_command(directory: Path, name: str, content: str) -> Path:
    directory.mkdir(exist_ok=True)
    command = directory / name
    command.write_text(f"#!/bin/bash\nset -euo pipefail\n{content}\n", encoding="utf-8")
    command.chmod(0o755)
    return command


@pytest.fixture(autouse=True)
def fake_pidfds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cleanup, "open_process_handle", lambda pid: pid)
    monkeypatch.setattr(cleanup, "close_process_handle", lambda _pidfd: None)


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


def test_stale_mode_fails_closed_without_runner_or_run_metadata(tmp_path: Path, context: dict[str, str]) -> None:
    legacy_environment = {key: context[key] for key in cleanup.LEGACY_OWNER_KEYS}
    create_process(tmp_path, 204, legacy_environment)

    matches, ambiguities = cleanup.find_matching_processes_with_ambiguities(tmp_path, context, "stale")

    assert matches == []
    assert len(ambiguities) == 1
    assert "lacks stable runner metadata" in ambiguities[0]


def test_stale_mode_rejects_explicit_runner_mismatch_in_legacy_process(tmp_path: Path, context: dict[str, str]) -> None:
    legacy_environment = {
        **{key: context[key] for key in cleanup.LEGACY_OWNER_KEYS},
        "RUNNER_NAME": "another-runner",
    }
    create_process(tmp_path, 205, legacy_environment)

    assert matching_pids(tmp_path, context, "stale") == []


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


def test_cleanup_escalates_from_term_to_kill(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 401, context)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            shutil.rmtree(tmp_path / str(pid))

    monkeypatch.setattr(cleanup, "send_process_signal", fake_kill)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [401]
    assert signaled == [401]
    assert remaining == []
    assert ambiguities == []
    assert signals == [(401, signal.SIGTERM), (401, signal.SIGKILL)]


def test_cleanup_rescans_ownership_before_kill(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 501, context)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGTERM:
            create_process(tmp_path, pid, {**context, "RUNNER_NAME": "another-runner"})

    monkeypatch.setattr(cleanup, "send_process_signal", fake_kill)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [501]
    assert signaled == [501]
    assert remaining == []
    assert ambiguities == []
    assert signals == [(501, signal.SIGTERM)]


def test_cleanup_rescans_start_time_before_term(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 502, context, start_time=1000)
    signals: list[tuple[int, int]] = []
    original_find_matching_processes = cleanup.find_matching_processes_with_ambiguities
    first_scan = True

    def replace_process_after_scan(proc_root: Path, owner: dict[str, str], mode: str) -> tuple[list[object], list[str]]:
        nonlocal first_scan
        matches = original_find_matching_processes(proc_root, owner, mode)
        if first_scan:
            first_scan = False
            create_process(tmp_path, 502, {**context, "GITHUB_JOB": "another-job"}, start_time=2000)
        return matches

    monkeypatch.setattr(cleanup, "find_matching_processes_with_ambiguities", replace_process_after_scan)

    monkeypatch.setattr(cleanup, "send_process_signal", lambda pid, signum: signals.append((pid, signum)))
    matched, signaled, remaining, ambiguities = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [502]
    assert signaled == []
    assert remaining == []
    assert ambiguities == []
    assert signals == []


def test_signal_uses_verified_pidfd(tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    create_process(tmp_path, 503, context)
    opened: list[int] = []
    closed: list[int] = []
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(cleanup, "open_process_handle", lambda pid: opened.append(pid) or 73)
    monkeypatch.setattr(cleanup, "close_process_handle", closed.append)
    monkeypatch.setattr(cleanup, "send_process_signal", lambda pidfd, signum: signals.append((pidfd, signum)))

    signaled, ambiguities = cleanup.signal_processes(
        cleanup.find_matching_processes(tmp_path, context, "current"),
        signal.SIGTERM,
        tmp_path,
        context,
        "current",
    )

    assert signaled == [503]
    assert ambiguities == []
    assert opened == [503]
    assert signals == [(73, signal.SIGTERM)]
    assert closed == [73]


def test_marker_cleanup_signals_owned_process_group_members_before_reporting_ambiguity(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_file = tmp_path / "server.marker"
    create_process(tmp_path, 701, context, engine_core=False, start_time=7001, process_group=701)
    create_process(tmp_path, 702, context, engine_core=False, process_group=701)
    create_process(
        tmp_path,
        703,
        {"GITHUB_REPOSITORY": context["GITHUB_REPOSITORY"]},
        engine_core=False,
        process_group=701,
    )
    cleanup.record_process_marker(marker_file, 701, tmp_path, context)
    signals: list[tuple[int, int]] = []

    def fake_signal(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        shutil.rmtree(tmp_path / str(pid))

    monkeypatch.setattr(cleanup, "send_process_signal", fake_signal)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_marker_group(
        marker_file,
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [701, 702]
    assert signaled == [701, 702]
    assert remaining == []
    assert signals == [(701, signal.SIGTERM), (702, signal.SIGTERM)]
    assert len(ambiguities) == 1
    assert "PID 703 lacks ownership metadata" in ambiguities[0]
    assert marker_file.exists()


def test_marker_cleanup_handles_children_after_launcher_exits(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_file = tmp_path / "server.marker"
    create_process(tmp_path, 711, context, engine_core=False, start_time=7101, process_group=711)
    create_process(tmp_path, 712, context, engine_core=False, process_group=711)
    cleanup.record_process_marker(marker_file, 711, tmp_path, context)
    shutil.rmtree(tmp_path / "711")
    signals: list[tuple[int, int]] = []

    def fake_signal(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        shutil.rmtree(tmp_path / str(pid))

    monkeypatch.setattr(cleanup, "send_process_signal", fake_signal)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_marker_group(
        marker_file,
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [712]
    assert signaled == [712]
    assert remaining == []
    assert ambiguities == []
    assert signals == [(712, signal.SIGTERM)]
    assert not marker_file.exists()


def test_stale_marker_cleanup_uses_marker_run_context(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_file = tmp_path / "server.marker"
    old_context = {**context, "GITHUB_RUN_ID": "1779", "GITHUB_RUN_ATTEMPT": "1"}
    create_process(tmp_path, 721, old_context, engine_core=False, start_time=7201, process_group=721)
    cleanup.record_process_marker(marker_file, 721, tmp_path, old_context)

    def fake_signal(pid: int, signum: int) -> None:
        assert signum == signal.SIGTERM
        shutil.rmtree(tmp_path / str(pid))

    monkeypatch.setattr(cleanup, "send_process_signal", fake_signal)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_marker_group(
        marker_file,
        tmp_path,
        context,
        "stale",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [721]
    assert signaled == [721]
    assert remaining == []
    assert ambiguities == []
    assert not marker_file.exists()


def test_unreadable_engine_core_environment_fails_closed(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 601, context)
    original_read_bytes = Path.read_bytes

    def deny_environment(path: Path) -> bytes:
        if path.name == "environ":
            raise PermissionError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_environment)

    matches, ambiguities = cleanup.find_matching_processes_with_ambiguities(tmp_path, context, "current")

    assert matches == []
    assert len(ambiguities) == 1
    assert "permission denied reading" in ambiguities[0]


def test_engine_core_without_stable_ownership_metadata_fails_closed(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 602, {"GITHUB_REPOSITORY": context["GITHUB_REPOSITORY"]})

    matches, ambiguities = cleanup.find_matching_processes_with_ambiguities(tmp_path, context, "stale")

    assert matches == []
    assert len(ambiguities) == 1
    assert "lacks ownership metadata" in ambiguities[0]


def test_cleanup_signals_proven_processes_before_reporting_ambiguous_process(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 610, context)
    create_process(tmp_path, 611, {"GITHUB_REPOSITORY": context["GITHUB_REPOSITORY"]})
    signals: list[tuple[int, int]] = []

    def fake_signal(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        shutil.rmtree(tmp_path / str(pid))

    monkeypatch.setattr(cleanup, "send_process_signal", fake_signal)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )

    assert matched == [610]
    assert signaled == [610]
    assert remaining == []
    assert signals == [(610, signal.SIGTERM)]
    assert len(ambiguities) == 1
    assert "PID 611 lacks ownership metadata" in ambiguities[0]


def test_main_fails_when_only_ambiguous_processes_are_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cleanup,
        "parse_args",
        lambda: SimpleNamespace(
            mode="current",
            proc_root=Path("/proc"),
            term_timeout_seconds=0,
            kill_timeout_seconds=0,
            marker_file=None,
            record_marker=None,
            pid=None,
            target_job=None,
        ),
    )
    monkeypatch.setattr(cleanup, "validate_context", lambda _environment: {})
    monkeypatch.setattr(
        cleanup,
        "cleanup_processes",
        lambda **_kwargs: ([], [], [], ["PID 811 lacks stable runner metadata"]),
    )

    result = cleanup.main()

    assert result == 2
    assert "PID 811 lacks stable runner metadata" in capsys.readouterr().err


def test_target_job_overrides_cleanup_job_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        mode="current",
        proc_root=Path("/proc"),
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        marker_file=None,
        record_marker=None,
        pid=None,
        target_job="ascend-benchmark",
        target_run_id="1780",
        target_run_attempt="2",
    )
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(cleanup, "parse_args", lambda: args)
    monkeypatch.setattr(cleanup, "validate_context", lambda _environment: {"GITHUB_JOB": "cleanup-ascend-benchmark"})

    def fake_cleanup(**kwargs: object) -> tuple[list[int], list[int], list[int], list[str]]:
        captured.append(kwargs["context"])
        return [], [], [], []

    monkeypatch.setattr(cleanup, "cleanup_processes", fake_cleanup)

    assert cleanup.main() == 0
    assert captured == [
        {
            "GITHUB_JOB": "ascend-benchmark",
            "GITHUB_RUN_ID": "1780",
            "GITHUB_RUN_ATTEMPT": "2",
        }
    ]


def test_incomplete_metadata_with_known_repository_mismatch_is_ignored(tmp_path: Path, context: dict[str, str]) -> None:
    create_process(tmp_path, 604, {"GITHUB_REPOSITORY": "another/repository"})

    assert matching_pids(tmp_path, context, "stale") == []


def test_signal_permission_failure_is_reported(
    tmp_path: Path, context: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_process(tmp_path, 603, context)

    def deny_signal(_pid: int, _signum: int) -> None:
        raise PermissionError

    monkeypatch.setattr(cleanup, "send_process_signal", deny_signal)
    matched, signaled, remaining, ambiguities = cleanup.cleanup_processes(
        tmp_path,
        context,
        "current",
        term_timeout_seconds=0,
        kill_timeout_seconds=0,
        sleep=lambda _: None,
    )
    assert matched == [603]
    assert signaled == []
    assert remaining == [603]
    assert any("permission denied signaling" in ambiguity for ambiguity in ambiguities)


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


def test_auto_mode_fails_when_non_root_helper_is_missing(tmp_path: Path, context: dict[str, str]) -> None:
    fake_bin = tmp_path / "bin"
    install_fake_command(fake_bin, "id", '[[ "${1:-}" == "-u" ]] && echo 1000')
    install_fake_command(fake_bin, "sudo", "exit 99")
    env = wrapper_environment(context, tmp_path)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ASCEND_BENCHMARK_USE_SUDO": "auto",
            "ASCEND_BENCHMARK_ROOT_HELPER": str(tmp_path / "missing-helper"),
        }
    )

    result = subprocess.run(
        ["bash", str(CLEANUP_WRAPPER), "stale"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "requires an executable root helper" in result.stderr
    assert "No stale" not in result.stdout


def test_old_root_helper_failure_is_not_silently_ignored(tmp_path: Path, context: dict[str, str]) -> None:
    fake_bin = tmp_path / "bin"
    install_fake_command(fake_bin, "id", '[[ "${1:-}" == "-u" ]] && echo 1000')
    install_fake_command(
        fake_bin,
        "sudo",
        'while [[ "$#" -gt 0 && "$1" == -* ]]; do shift; done\nexec "$@"',
    )
    old_helper = install_fake_command(tmp_path, "old-root-helper", 'echo "Unsupported subcommand" >&2\nexit 2')
    env = wrapper_environment(context, tmp_path)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ASCEND_BENCHMARK_USE_SUDO": "auto",
            "ASCEND_BENCHMARK_ROOT_HELPER": str(old_helper),
        }
    )

    result = subprocess.run(
        ["bash", str(CLEANUP_WRAPPER), "stale"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "root cleanup failed" in result.stderr
    assert "Unsupported subcommand" in result.stderr


def test_wrapper_invokes_current_root_helper_end_to_end(tmp_path: Path, context: dict[str, str]) -> None:
    fake_bin = tmp_path / "bin"
    install_fake_command(
        fake_bin,
        "sudo",
        'while [[ "$#" -gt 0 && "$1" == -* ]]; do shift; done\nexec "$@"',
    )
    env = wrapper_environment(context, tmp_path)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ASCEND_BENCHMARK_USE_SUDO": "1",
            "ASCEND_BENCHMARK_ROOT_HELPER": str(ROOT_HELPER),
        }
    )

    result = subprocess.run(
        ["bash", str(CLEANUP_WRAPPER), "current"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "No current Ascend benchmark EngineCore processes found." in result.stdout
