#!/usr/bin/env python3
"""Add an opt-in HCCL transport to the Stage-5 coordinator.

The existing Stage-5 worker/proposer protocol is kept unchanged.  This patch
only changes the coordinator transport:

    Stage5 Target proposer -> target HCCL sidecar -> HCCL ->
    draft HCCL sidecar -> persistent Draft worker

The original Unix-RPC path remains the default.  Use
``--transport hccl`` only after the standalone HCCL probes pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent


TARGET = Path("pearl_stage5_coordinator.py")
MARKER = "# PEARL_STAGE5_HCCL_MAINFLOW_V1"


HCCL_HELPERS = dedent(
    '''
        # PEARL_STAGE5_HCCL_MAINFLOW_V1
        def _hccl_choose_port() -> int:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", 0))
                return int(sock.getsockname()[1])
            finally:
                sock.close()

        def _hccl_clean_env(device: str) -> dict[str, str]:
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["TORCHDYNAMO_DISABLE"] = "1"
            env.pop("VLLM_NPU_DEVICE", None)
            env.pop("VLLM_USE_V1", None)
            env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
                env.pop(name, None)
            script_dir = str(Path(__file__).resolve().parent)
            env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
            return env

        def _hccl_launch_bridge(
            bridge_script: Path,
            role: str,
            device: str,
            rank: int,
            master_addr: str,
            master_port: int,
            socket_path: Path,
            draft_socket: Path | None,
            max_model_len: int,
        ) -> subprocess.Popen:
            env = _hccl_clean_env(device)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            env["WORLD_SIZE"] = "2"
            env["MASTER_ADDR"] = str(master_addr)
            env["MASTER_PORT"] = str(master_port)
            env.setdefault("HCCL_CONNECT_TIMEOUT", "120")
            env.setdefault("HCCL_EXEC_TIMEOUT", "120")
            command = [
                sys.executable,
                str(bridge_script),
                "--role",
                role,
                "--socket",
                str(socket_path),
                "--max-model-len",
                str(max_model_len),
            ]
            if draft_socket is not None:
                command.extend(["--draft-socket", str(draft_socket)])
            print(
                f"[stage5] launch HCCL bridge {role}: rank={rank} "
                f"card={device} master={master_addr}:{master_port}",
                flush=True,
            )
            return subprocess.Popen(command, env=env)

        def _hccl_wait_socket(
            path: Path, process: subprocess.Popen, timeout: float
        ) -> None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"HCCL bridge exited with returncode={process.returncode}"
                    )
                if path.exists():
                    return
                time.sleep(0.1)
            raise TimeoutError(f"timeout waiting for HCCL socket {path}")

        def _hccl_stop_sidecar(process: subprocess.Popen) -> None:
            if process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        def _hccl_main(args: argparse.Namespace) -> None:
            bridge_script = Path(__file__).with_name(
                "pearl_stage6_hccl_bridge_v1.py"
            )
            if not bridge_script.is_file():
                raise FileNotFoundError(
                    f"HCCL bridge is required for --transport hccl: {bridge_script}"
                )

            master_port = args.hccl_master_port
            if master_port is None:
                master_port = int(
                    os.environ.get("PEARL_STAGE5_HCCL_MASTER_PORT", "0")
                )
            if master_port <= 0:
                master_port = _hccl_choose_port()

            script = Path(__file__).with_name("pearl_stage5_worker.py")
            processes: list[tuple[str, subprocess.Popen, JsonSocketClient | None]] = []
            sidecars: list[tuple[str, subprocess.Popen]] = []
            with tempfile.TemporaryDirectory(prefix="nano_pearl_stage5_hccl_") as temp_dir:
                temp = Path(temp_dir)
                draft_control = temp / "draft.control.sock"
                draft_proposal = temp / "draft.proposal.sock"
                target_control = temp / "target.control.sock"
                target_hccl = temp / "target.hccl.sock"
                try:
                    draft_process = launch(
                        script,
                        "draft",
                        args.draft_model,
                        args.draft_device,
                        draft_control,
                        args.max_model_len,
                        args.gamma,
                        proposal_socket=draft_proposal,
                    )
                    processes.append(("draft", draft_process, None))

                    draft_bridge = _hccl_launch_bridge(
                        bridge_script,
                        "draft",
                        args.draft_device,
                        0,
                        args.hccl_master_addr,
                        master_port,
                        temp / "draft.bridge.placeholder.sock",
                        draft_proposal,
                        args.max_model_len,
                    )
                    sidecars.append(("hccl-draft", draft_bridge))

                    target_bridge = _hccl_launch_bridge(
                        bridge_script,
                        "target",
                        args.target_device,
                        1,
                        args.hccl_master_addr,
                        master_port,
                        target_hccl,
                        None,
                        args.max_model_len,
                    )
                    sidecars.append(("hccl-target", target_bridge))
                    _hccl_wait_socket(target_hccl, target_bridge, args.startup_timeout)

                    target_process = launch(
                        script,
                        "target",
                        args.target_model,
                        args.target_device,
                        target_control,
                        args.max_model_len,
                        args.gamma,
                        draft_socket=target_hccl,
                    )
                    processes.append(("target", target_process, None))

                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [
                            pool.submit(
                                connect,
                                draft_control,
                                draft_process,
                                args.startup_timeout,
                            ),
                            pool.submit(
                                connect,
                                target_control,
                                target_process,
                                args.startup_timeout,
                            ),
                        ]
                        for index, future in enumerate(futures):
                            role, process, _ = processes[index]
                            processes[index] = (role, process, future.result())

                    draft_client = processes[0][2]
                    target_client = processes[1][2]
                    assert draft_client is not None and target_client is not None
                    print(draft_client.request({"cmd": "ping"}), flush=True)
                    print(target_client.request({"cmd": "ping"}), flush=True)

                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        args.target_model, trust_remote_code=True
                    )
                    prompts = args.prompts or ["The capital of France is"]
                    for prompt_index, prompt in enumerate(prompts, start=1):
                        prompt_ids = [
                            int(x)
                            for x in tokenizer.encode(prompt, add_special_tokens=True)
                        ]
                        response = target_client.request(
                            {
                                "cmd": "generate",
                                "prompt_token_ids": prompt_ids,
                                "max_tokens": args.max_tokens,
                            }
                        )
                        print(
                            f"[stage5-hccl] prompt {prompt_index}: {prompt!r}; "
                            f"output_ids={response.get('token_ids', [])}; "
                            f"text={response.get('text')!r}; "
                            f"elapsed_ms={response.get('elapsed_ms')}",
                            flush=True,
                        )

                    print(
                        "[stage5-hccl] Stage-5 HCCL Draft/Target path completed",
                        flush=True,
                    )
                finally:
                    # Stop vLLM workers before killing sidecars.  The sidecars
                    # may be blocked in HCCL recv and are not graceful workers.
                    for role, process, client in processes:
                        if client is not None:
                            try:
                                print(
                                    f"[{role}] {client.request({'cmd': 'stop'})}",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(f"[{role}] stop failed: {exc!r}", flush=True)
                            client.close()
                    for role, process, _client in processes:
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    for role, process in sidecars:
                        _hccl_stop_sidecar(process)
    '''
)


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("HCCL main-flow patch is already applied")

    source = _replace_once(
        source,
        '    parser.add_argument("--prompt", dest="prompts", action="append", default=None)\n',
        '    parser.add_argument("--prompt", dest="prompts", action="append", default=None)\n'
        '    parser.add_argument("--transport", choices=("rpc", "hccl"), default="rpc")\n'
        '    parser.add_argument("--hccl-master-addr", default="127.0.0.1")\n'
        '    parser.add_argument("--hccl-master-port", type=int, default=None)\n',
        "parser arguments",
    )
    source = _replace_once(
        source,
        '    env = os.environ.copy()\n    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)\n',
        '    env = os.environ.copy()\n'
        '    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):\n'
        '        env.pop(name, None)\n'
        '    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)\n',
        "worker environment",
    )
    source = _replace_once(
        source,
        'def connect(path: Path, process: subprocess.Popen, timeout: float) -> JsonSocketClient:\n',
        HCCL_HELPERS + '\n\ndef connect(path: Path, process: subprocess.Popen, timeout: float) -> JsonSocketClient:\n',
        "HCCL helper insertion",
    )
    source = _replace_once(
        source,
        '    if args.gamma <= 0 or args.max_tokens <= 0:\n        raise ValueError("--gamma and --max-tokens must be positive")\n\n',
        '    if args.gamma <= 0 or args.max_tokens <= 0:\n        raise ValueError("--gamma and --max-tokens must be positive")\n'
        '    if args.transport == "hccl":\n'
        '        _hccl_main(args)\n'
        '        return\n\n',
        "HCCL main-flow dispatch",
    )
    return source


def _backup_file(target: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / target.name)


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    digest = hashlib.sha256(original.encode()).hexdigest()[:12]
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: opt-in HCCL transport in Stage-5 coordinator")
    if MARKER in original:
        print("already patched: no files changed")
        return
    transformed = transform(original)
    if dry_run:
        print(f"dry-run: no files changed; source_sha256={digest}")
        return
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_hccl_mainflow_v1.{stamp}"
    _backup_file(target, backup_dir)
    target.write_text(transformed, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
