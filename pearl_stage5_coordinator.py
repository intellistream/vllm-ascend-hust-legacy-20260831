# PEARL_STAGE5_HCCL_BATCH_MAINFLOW_V1
# PEARL_STAGE5_BATCH_GT1_V1
#!/usr/bin/env python3
"""Coordinator for the persistent-KV nano-PEARL Stage-5 integration."""

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
    parser.add_argument("--draft-device", default="4")
    parser.add_argument("--target-device", default="5")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--prompt", dest="prompts", action="append", default=None)
    parser.add_argument("--transport", choices=("rpc", "hccl"), default="rpc")
    parser.add_argument("--hccl-master-addr", default="127.0.0.1")
    parser.add_argument("--hccl-master-port", type=int, default=None)
    return parser


def launch(
    script: Path,
    role: str,
    model: str,
    device: str,
    control_socket: Path,
    max_model_len: int,
    gamma: int,
    max_num_seqs: int,
    proposal_socket: Path | None = None,
    draft_socket: Path | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(name, None)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env.pop("VLLM_NPU_DEVICE", None)
    env.pop("VLLM_USE_V1", None)
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
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
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-model-len",
        str(max_model_len),
    ]
    if proposal_socket is not None:
        command.extend(["--proposal-socket", str(proposal_socket)])
    if draft_socket is not None:
        command.extend(["--draft-socket", str(draft_socket)])
    print(f"[stage5] launch {role}: card={device}, model={model}", flush=True)
    return subprocess.Popen(command, env=env)



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
        "pearl_stage6_hccl_bridge_v3.py"  # PEARL_STAGE5_HCCL_MAINFLOW_V5  # PEARL_STAGE5_HCCL_MAINFLOW_V4
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
                getattr(args, "max_num_seqs", 1),
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
                getattr(args, "max_num_seqs", 1),
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
            if args.max_num_seqs < 1:
                raise ValueError("--max-num-seqs must be positive")
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            if args.batch_size > args.max_num_seqs:
                raise ValueError(
                    "--batch-size cannot exceed --max-num-seqs"
                )

            for batch_start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[
                    batch_start : batch_start + args.batch_size
                ]
                batch_prompt_ids = [
                    [
                        int(x)
                        for x in tokenizer.encode(
                            prompt, add_special_tokens=True
                        )
                    ]
                    for prompt in batch_prompts
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids_batch": batch_prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                results = response.get("results")
                if not isinstance(results, list):
                    raise RuntimeError(
                        "HCCL Target batch response is missing 'results'"
                    )
                if len(results) != len(batch_prompts):
                    raise RuntimeError(
                        "HCCL Target batch result count mismatch: "
                        f"expected={len(batch_prompts)} got={len(results)}"
                    )
                for offset, (prompt, result) in enumerate(
                    zip(batch_prompts, results)
                ):
                    output_ids = [
                        int(x) for x in result.get("token_ids", [])
                    ]
                    print(
                        f"[stage5-hccl] prompt "
                        f"{batch_start + offset + 1}: {prompt!r}; "
                        f"output_ids={output_ids}; "
                        f"text={result.get('text')!r}; "
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
                    print(f"[stage5] {message}", flush=True)
                    if message.get("status") == "ready":
                        return client
                    if message.get("status") != "loading":
                        raise RuntimeError(f"unexpected startup message: {message}")
            except Exception as exc:
                last_error = exc
                sock.close()
        time.sleep(0.2)
    raise TimeoutError(f"timeout waiting for {path}; last_error={last_error!r}")


def main() -> None:
    args = build_parser().parse_args()
    if args.transport == "hccl":
        _hccl_main(args)
        return
    if args.gamma <= 0 or args.max_tokens <= 0:
        raise ValueError("--gamma and --max-tokens must be positive")
    if args.max_num_seqs < 1 or args.batch_size < 1:
        raise ValueError("--max-num-seqs and --batch-size must be positive")
    if args.batch_size > args.max_num_seqs:
        raise ValueError("--batch-size cannot exceed --max-num-seqs")

    script = Path(__file__).with_name("pearl_stage5_worker.py")
    workers: list[tuple[str, subprocess.Popen, JsonSocketClient | None]] = []
    with tempfile.TemporaryDirectory(prefix="nano_pearl_stage5_") as temp_dir:
        temp = Path(temp_dir)
        draft_control = temp / "draft.control.sock"
        draft_proposal = temp / "draft.proposal.sock"
        target_control = temp / "target.control.sock"
        try:
            draft_process = launch(
                script,
                "draft",
                args.draft_model,
                args.draft_device,
                draft_control,
                args.max_model_len,
                args.gamma,
                args.max_num_seqs,
                proposal_socket=draft_proposal,
            )
            target_process = launch(
                script,
                "target",
                args.target_model,
                args.target_device,
                target_control,
                args.max_model_len,
                args.gamma,
                args.max_num_seqs,
                draft_socket=draft_proposal,
            )
            workers = [("draft", draft_process, None), ("target", target_process, None)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(connect, draft_control, draft_process, args.startup_timeout),
                    pool.submit(connect, target_control, target_process, args.startup_timeout),
                ]
                for index, future in enumerate(futures):
                    role, process, _ = workers[index]
                    workers[index] = (role, process, future.result())

            draft_client = workers[0][2]
            target_client = workers[1][2]
            assert draft_client is not None and target_client is not None
            print(draft_client.request({"cmd": "ping"}), flush=True)
            print(target_client.request({"cmd": "ping"}), flush=True)

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.target_model, trust_remote_code=True
            )
            prompts = args.prompts or ["The capital of France is"]
            if args.max_num_seqs < 1:
                raise ValueError("--max-num-seqs must be positive")
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            if args.batch_size > args.max_num_seqs:
                raise ValueError(
                    "--batch-size cannot exceed --max-num-seqs"
                )

            for batch_start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[batch_start : batch_start + args.batch_size]
                batch_prompt_ids = [
                    [
                        int(x)
                        for x in tokenizer.encode(
                            prompt, add_special_tokens=True
                        )
                    ]
                    for prompt in batch_prompts
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids_batch": batch_prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                results = response.get("results")
                if not isinstance(results, list):
                    results = [response]
                if len(results) != len(batch_prompts):
                    raise RuntimeError(
                        "Target returned an unexpected result count: "
                        f"expected={len(batch_prompts)} got={len(results)}"
                    )
                for offset, (prompt, result) in enumerate(
                    zip(batch_prompts, results)
                ):
                    output_ids = [
                        int(x) for x in result.get("token_ids", [])
                    ]
                    print(
                        f"[stage5] prompt {batch_start + offset + 1}: "
                        f"{prompt!r}; output_ids={output_ids}; "
                        f"text={result.get('text')!r}; "
                        f"elapsed_ms={response.get('elapsed_ms')}",
                        flush=True,
                    )

            print(
                "[stage5] persistent Draft/Target path completed; "
                "inspect logs for draft prefix sync and rejection statistics",
                flush=True,
            )
        finally:
            for role, process, client in workers:
                if client is not None:
                    try:
                        print(f"[{role}] {client.request({'cmd': 'stop'})}", flush=True)
                    except Exception as exc:
                        print(f"[{role}] stop failed: {exc!r}", flush=True)
                    client.close()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    main()
