#!/usr/bin/env python3
"""Stage-6 coordinator: Stage-5 workers plus an opt-in HCCL sidecar path."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class JsonSocketClient:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.reader = sock.makefile("r", encoding="utf-8")

    def send(self, message: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))

    def receive(self) -> dict[str, Any]:
        line = self.reader.readline()
        if not line:
            raise RuntimeError("worker closed the Unix socket")
        message = json.loads(line)
        if message.get("status") == "error":
            raise RuntimeError(message.get("error", "worker error"))
        return message

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.send(message)
        return self.receive()

    def close(self) -> None:
        self.reader.close()
        self.sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-8B")
    parser.add_argument("--draft-device", default="6")
    parser.add_argument("--target-device", default="7")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--prompt", dest="prompts", action="append", default=None)
    parser.add_argument("--hccl-master-addr", default="127.0.0.1")
    parser.add_argument("--hccl-master-port", type=int, default=None)
    return parser


def choose_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def clean_worker_env(device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env.pop("VLLM_NPU_DEVICE", None)
    env.pop("VLLM_USE_V1", None)
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(name, None)
    return env


def launch_worker(
    script: Path,
    role: str,
    model: str,
    device: str,
    control_socket: Path,
    max_model_len: int,
    gamma: int,
    proposal_socket: Path | None = None,
    draft_socket: Path | None = None,
) -> subprocess.Popen:
    env = clean_worker_env(device)
    script_dir = str(script.parent)
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script),
        "--role",
        role,
        "--model",
        model,
        "--socket",
        str(control_socket),
        "--gamma",
        str(gamma),
        "--max-model-len",
        str(max_model_len),
    ]
    if proposal_socket is not None:
        command.extend(["--proposal-socket", str(proposal_socket)])
    if draft_socket is not None:
        command.extend(["--draft-socket", str(draft_socket)])
    print(
        f"[stage6] launch {role}: card={device}, model={model}",
        flush=True,
    )
    return subprocess.Popen(command, env=env)


def launch_bridge(
    script: Path,
    role: str,
    device: str,
    rank: int,
    master_addr: str,
    master_port: int,
    socket_path: Path,
    draft_socket: Path | None,
    max_model_len: int,
) -> subprocess.Popen:
    env = clean_worker_env(device)
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = str(rank)
    env["WORLD_SIZE"] = "2"
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env.setdefault("HCCL_CONNECT_TIMEOUT", "120")
    env.setdefault("HCCL_EXEC_TIMEOUT", "120")
    script_dir = str(script.parent)
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script),
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
        f"[stage6] launch HCCL bridge {role}: rank={rank}, card={device}, "
        f"master={master_addr}:{master_port}",
        flush=True,
    )
    return subprocess.Popen(command, env=env)


def connect(path: Path, process: subprocess.Popen, timeout: float) -> JsonSocketClient:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker exited with returncode={process.returncode}")
        if path.exists():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(path))
                client = JsonSocketClient(sock)
                while True:
                    message = client.receive()
                    print(f"[stage6] {message}", flush=True)
                    if message.get("status") == "ready":
                        return client
                    if message.get("status") != "loading":
                        raise RuntimeError(f"unexpected startup message: {message}")
            except Exception as exc:
                last_error = exc
                sock.close()
        time.sleep(0.2)
    raise TimeoutError(f"timeout waiting for {path}; last_error={last_error!r}")


def wait_for_bridge(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"HCCL bridge exited with returncode={process.returncode}"
            )
        if path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"timeout waiting for HCCL bridge socket {path}")


def stop_process(process: subprocess.Popen) -> None:
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    args = build_parser().parse_args()
    if args.gamma <= 0 or args.max_tokens <= 0:
        raise ValueError("--gamma and --max-tokens must be positive")

    worker_script = Path(__file__).with_name("pearl_stage5_worker.py")
    bridge_script = Path(__file__).with_name("pearl_stage6_hccl_bridge_v1.py")
    if not worker_script.is_file():
        raise FileNotFoundError(worker_script)
    if not bridge_script.is_file():
        raise FileNotFoundError(bridge_script)

    master_port = args.hccl_master_port
    if master_port is None:
        master_port = int(os.environ.get("PEARL_STAGE6_HCCL_MASTER_PORT", "0"))
    if master_port <= 0:
        master_port = choose_port()

    processes: list[tuple[str, subprocess.Popen, JsonSocketClient | None]] = []
    with tempfile.TemporaryDirectory(prefix="nano_pearl_stage6_hccl_") as temp_dir:
        temp = Path(temp_dir)
        draft_control = temp / "draft.control.sock"
        draft_proposal = temp / "draft.proposal.sock"
        target_control = temp / "target.control.sock"
        target_hccl = temp / "target.hccl.sock"
        try:
            draft_process = launch_worker(
                worker_script,
                "draft",
                args.draft_model,
                args.draft_device,
                draft_control,
                args.max_model_len,
                args.gamma,
                proposal_socket=draft_proposal,
            )
            processes.append(("draft", draft_process, None))

            draft_bridge = launch_bridge(
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
            processes.append(("hccl-draft", draft_bridge, None))

            target_bridge = launch_bridge(
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
            processes.append(("hccl-target", target_bridge, None))
            wait_for_bridge(target_hccl, target_bridge, args.startup_timeout)

            target_process = launch_worker(
                worker_script,
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
                    role, process, _ = processes[-1] if index else processes[0]
                    client = future.result()
                    if index == 0:
                        processes[0] = (role, process, client)
                    else:
                        processes[-1] = (role, process, client)

            draft_client = next(
                client for role, _, client in processes if role == "draft"
            )
            target_client = next(
                client for role, _, client in processes if role == "target"
            )
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
                    int(x) for x in tokenizer.encode(prompt, add_special_tokens=True)
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids": prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                output_ids = [int(x) for x in response.get("token_ids", [])]
                print(
                    f"[stage6] prompt {prompt_index}: {prompt!r}; "
                    f"output_ids={output_ids}; text={response.get('text')!r}; "
                    f"elapsed_ms={response.get('elapsed_ms')}",
                    flush=True,
                )

            print(
                "[stage6] HCCL Draft/Target path completed; inspect HCCL bridge, "
                "KV reuse, and acceptance logs",
                flush=True,
            )
        finally:
            # Stop the vLLM workers first.  The sidecars are intentionally
            # terminated afterwards because they may be blocked in HCCL recv.
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
            for role, process, client in processes:
                if client is None:
                    stop_process(process)


if __name__ == "__main__":
    main()
