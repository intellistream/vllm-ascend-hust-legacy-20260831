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

OWNER_KEYS = (
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "RUNNER_NAME",
    "RUNNER_WORKSPACE",
)
RUN_KEYS = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
REQUIRED_KEYS = OWNER_KEYS + RUN_KEYS


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    environment: Mapping[str, str]


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def read_environment(path: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    for item in read_bytes(path).split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def is_engine_core(process_dir: Path) -> bool:
    status = read_bytes(process_dir / "status").decode(errors="replace")
    for line in status.splitlines():
        if line.startswith("Name:") and line.split(":", 1)[1].strip() == "VLLM::EngineCor":
            return True

    cmdline = read_bytes(process_dir / "cmdline").replace(b"\0", b" ")
    return b"VLLM::EngineCore" in cmdline


def validate_context(environment: Mapping[str, str]) -> dict[str, str]:
    missing = [key for key in REQUIRED_KEYS if not environment.get(key)]
    if missing:
        raise ValueError(f"missing required ownership context: {', '.join(missing)}")
    return {key: environment[key] for key in REQUIRED_KEYS}


def belongs_to_context(process_environment: Mapping[str, str], context: Mapping[str, str], mode: str) -> bool:
    if any(process_environment.get(key) != context[key] for key in OWNER_KEYS):
        return False

    process_run = tuple(process_environment.get(key) for key in RUN_KEYS)
    current_run = tuple(context[key] for key in RUN_KEYS)
    if mode == "current":
        return process_run == current_run
    if mode == "stale":
        return all(process_run) and process_run != current_run
    raise ValueError(f"unsupported cleanup mode: {mode}")


def find_matching_processes(proc_root: Path, context: Mapping[str, str], mode: str) -> list[ProcessInfo]:
    matches: list[ProcessInfo] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot scan process directory {proc_root}: {exc}") from exc

    for process_dir in entries:
        if not process_dir.name.isdigit() or not is_engine_core(process_dir):
            continue
        environment = read_environment(process_dir / "environ")
        if belongs_to_context(environment, context, mode):
            matches.append(ProcessInfo(pid=int(process_dir.name), environment=environment))
    return sorted(matches, key=lambda process: process.pid)


def signal_processes(
    processes: Sequence[ProcessInfo],
    signum: signal.Signals,
    kill: Callable[[int, int], None],
) -> None:
    for process in processes:
        try:
            kill(process.pid, signum)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"permission denied signaling PID {process.pid}") from exc


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
    kill: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[int], list[int]]:
    initial = find_matching_processes(proc_root, context, mode)
    if not initial:
        return [], []

    signal_processes(initial, signal.SIGTERM, kill)
    remaining = wait_for_exit(proc_root, context, mode, term_timeout_seconds, poll_seconds, sleep)
    if remaining:
        # Rescanning ownership before SIGKILL prevents signaling an unrelated
        # process if a PID was reused after the TERM phase.
        signal_processes(remaining, signal.SIGKILL, kill)
        remaining = wait_for_exit(proc_root, context, mode, kill_timeout_seconds, poll_seconds, sleep)
    return [process.pid for process in initial], [process.pid for process in remaining]


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
        matched, remaining = cleanup_processes(
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
    print(f"Cleaned {args.mode} Ascend benchmark EngineCore process(es): {matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
