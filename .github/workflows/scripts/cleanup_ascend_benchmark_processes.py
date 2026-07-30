#!/usr/bin/env python3
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
"""Clean up vLLM EngineCore processes owned by an Ascend benchmark job."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

LEGACY_OWNER_KEYS = (
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
)
RUNNER_KEYS = (
    "RUNNER_NAME",
    "RUNNER_WORKSPACE",
)
RUN_KEYS = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
REQUIRED_CONTEXT_KEYS = LEGACY_OWNER_KEYS + RUNNER_KEYS + RUN_KEYS


class ProcessDisappeared(Exception):
    """The process exited while its procfs metadata was being inspected."""


class ProcessInspectionError(RuntimeError):
    """A process exists but cannot be inspected safely."""


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    start_time: int


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ProcessDisappeared from exc
    except PermissionError as exc:
        raise ProcessInspectionError(f"permission denied reading {path}") from exc
    except OSError as exc:
        raise ProcessInspectionError(f"cannot read {path}: {exc}") from exc


def read_environment(path: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    for item in read_bytes(path).split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def read_start_time(process_dir: Path) -> int:
    stat = read_bytes(process_dir / "stat")
    command_end = stat.rfind(b")")
    if command_end < 0:
        raise ProcessInspectionError(f"malformed process stat: {process_dir / 'stat'}")

    # Fields after the command name begin at field 3 (state). Linux starttime
    # is field 22, so it is index 19 in this suffix.
    fields = stat[command_end + 1 :].split()
    if len(fields) <= 19:
        raise ProcessInspectionError(f"process stat lacks start time: {process_dir / 'stat'}")
    try:
        return int(fields[19])
    except ValueError as exc:
        raise ProcessInspectionError(f"invalid process start time: {process_dir / 'stat'}") from exc


def open_process_handle(pid: int) -> int:
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None or not hasattr(signal, "pidfd_send_signal"):
        raise ProcessInspectionError("pidfd support is required for safe EngineCore cleanup")
    try:
        return pidfd_open(pid)
    except ProcessLookupError as exc:
        raise ProcessDisappeared from exc
    except PermissionError as exc:
        raise ProcessInspectionError(f"permission denied opening PID {pid}") from exc
    except OSError as exc:
        raise ProcessInspectionError(f"cannot open PID {pid}: {exc}") from exc


def send_process_signal(pidfd: int, signum: signal.Signals) -> None:
    try:
        signal.pidfd_send_signal(pidfd, signum)
    except ProcessLookupError as exc:
        raise ProcessDisappeared from exc
    except PermissionError as exc:
        raise ProcessInspectionError("permission denied signaling EngineCore process") from exc
    except OSError as exc:
        raise ProcessInspectionError(f"cannot signal EngineCore process: {exc}") from exc


def close_process_handle(pidfd: int) -> None:
    try:
        os.close(pidfd)
    except OSError:
        pass


def is_engine_core(process_dir: Path) -> bool:
    status = read_bytes(process_dir / "status").decode(errors="replace")
    for line in status.splitlines():
        if line.startswith("Name:") and line.split(":", 1)[1].strip() == "VLLM::EngineCor":
            return True

    cmdline = read_bytes(process_dir / "cmdline").replace(b"\0", b" ")
    return b"VLLM::EngineCore" in cmdline


def validate_context(environment: Mapping[str, str]) -> dict[str, str]:
    missing = [key for key in REQUIRED_CONTEXT_KEYS if not environment.get(key)]
    if missing:
        raise ValueError(f"missing required ownership context: {', '.join(missing)}")
    return {key: environment[key] for key in REQUIRED_CONTEXT_KEYS}


def belongs_to_context(process_environment: Mapping[str, str], context: Mapping[str, str], mode: str) -> bool:
    if any(process_environment.get(key) != context[key] for key in LEGACY_OWNER_KEYS):
        return False

    if mode == "current":
        return all(process_environment.get(key) == context[key] for key in RUNNER_KEYS + RUN_KEYS)
    if mode == "stale":
        return all(process_environment.get(key) == context[key] for key in RUNNER_KEYS) and any(
            process_environment.get(key) != context[key] for key in RUN_KEYS
        )
    raise ValueError(f"unsupported cleanup mode: {mode}")


def inspect_process(process_dir: Path, context: Mapping[str, str], mode: str) -> ProcessInfo | None:
    start_time = read_start_time(process_dir)
    if not is_engine_core(process_dir):
        return None
    environment = read_environment(process_dir / "environ")
    if read_start_time(process_dir) != start_time:
        return None

    # A known mismatch is enough to establish that this process belongs to
    # another workload. Otherwise incomplete stable metadata is ambiguous and
    # must fail closed instead of being reported as "no processes".
    if any(environment.get(key) and environment[key] != context[key] for key in LEGACY_OWNER_KEYS):
        return None
    missing_legacy_keys = [key for key in LEGACY_OWNER_KEYS if not environment.get(key)]
    if missing_legacy_keys:
        raise ProcessInspectionError(
            f"EngineCore PID {process_dir.name} lacks ownership metadata: {', '.join(missing_legacy_keys)}"
        )
    ownership_keys = RUNNER_KEYS + RUN_KEYS
    if any(environment.get(key) and environment[key] != context[key] for key in RUNNER_KEYS):
        return None
    missing_ownership_keys = [key for key in ownership_keys if not environment.get(key)]
    if missing_ownership_keys:
        raise ProcessInspectionError(
            f"EngineCore PID {process_dir.name} lacks stable runner metadata: {', '.join(missing_ownership_keys)}"
        )
    if not belongs_to_context(environment, context, mode):
        return None
    return ProcessInfo(pid=int(process_dir.name), start_time=start_time)


def find_matching_processes(proc_root: Path, context: Mapping[str, str], mode: str) -> list[ProcessInfo]:
    matches: list[ProcessInfo] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot scan process directory {proc_root}: {exc}") from exc

    for process_dir in entries:
        if not process_dir.name.isdigit():
            continue
        try:
            process = inspect_process(process_dir, context, mode)
        except ProcessDisappeared:
            continue
        if process is not None:
            matches.append(process)
    return sorted(matches, key=lambda process: process.pid)


def signal_processes(
    processes: Sequence[ProcessInfo],
    signum: signal.Signals,
    proc_root: Path,
    context: Mapping[str, str],
    mode: str,
) -> list[int]:
    signaled: list[int] = []
    for process in processes:
        try:
            pidfd = open_process_handle(process.pid)
        except ProcessDisappeared:
            continue
        try:
            current = inspect_process(proc_root / str(process.pid), context, mode)
            if current != process:
                continue
            send_process_signal(pidfd, signum)
            if process.pid not in signaled:
                signaled.append(process.pid)
        except ProcessDisappeared:
            continue
        except PermissionError as exc:
            raise ProcessInspectionError("permission denied signaling EngineCore process") from exc
        finally:
            close_process_handle(pidfd)
    return signaled


def wait_for_exit(
    proc_root: Path,
    context: Mapping[str, str],
    mode: str,
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
) -> list[ProcessInfo]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = find_matching_processes(proc_root, context, mode)
        if not remaining or time.monotonic() >= deadline:
            return remaining
        sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def cleanup_processes(
    proc_root: Path,
    context: Mapping[str, str],
    mode: str,
    term_timeout_seconds: float,
    kill_timeout_seconds: float,
    poll_seconds: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[int], list[int], list[int]]:
    initial = find_matching_processes(proc_root, context, mode)
    if not initial:
        return [], [], []

    signaled = signal_processes(initial, signal.SIGTERM, proc_root, context, mode)
    remaining = wait_for_exit(proc_root, context, mode, term_timeout_seconds, poll_seconds, sleep)
    if remaining:
        for pid in signal_processes(remaining, signal.SIGKILL, proc_root, context, mode):
            if pid not in signaled:
                signaled.append(pid)
        remaining = wait_for_exit(proc_root, context, mode, kill_timeout_seconds, poll_seconds, sleep)
    return [process.pid for process in initial], signaled, [process.pid for process in remaining]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "stale"), required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--term-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--kill-timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        context = validate_context(os.environ)
        matched, signaled, remaining = cleanup_processes(
            proc_root=args.proc_root,
            context=context,
            mode=args.mode,
            term_timeout_seconds=max(0.0, args.term_timeout_seconds),
            kill_timeout_seconds=max(0.0, args.kill_timeout_seconds),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Ascend benchmark process cleanup failed: {exc}", file=sys.stderr)
        return 2

    if not matched:
        print(f"No {args.mode} Ascend benchmark EngineCore processes found.")
        return 0

    if remaining:
        print(
            f"Ascend benchmark EngineCore process(es) survived SIGKILL: {remaining}",
            file=sys.stderr,
        )
        return 1
    skipped = [pid for pid in matched if pid not in signaled]
    if skipped:
        print(f"Skipped changed {args.mode} Ascend benchmark EngineCore process(es): {skipped}")
    print(f"Cleaned {args.mode} Ascend benchmark EngineCore process(es): {signaled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
