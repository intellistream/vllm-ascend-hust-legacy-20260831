#!/usr/bin/env python3
"""Coordinator for the token-level two-card nano-PEARL smoke test.

It launches one worker on card 4 and one worker on card 5, waits for both
models to become ready, sends requests to both workers concurrently, and then
shuts them down cleanly.

The current protocol performs one explicit Draft -> Target token handoff:

1. tokenize a prefix on the CPU;
2. ask Draft for ``gamma`` token IDs;
3. send ``prefix + draft_token_ids`` to Target as token IDs;
4. ask Target to perform a greedy prompt-logprob verification;
5. append the accepted draft tokens and Target's replacement/bonus token to
   the committed prefix.

This is still a correctness prototype: every request recomputes the full
prefix and does not reuse or rollback KV cache.
"""

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
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(payload)

    def receive(self) -> dict[str, Any]:
        line = self.reader.readline()
        if not line:
            raise RuntimeError("worker closed the Unix socket")
        message = json.loads(line)
        if message.get("status") == "error":
            raise RuntimeError(f"worker error: {message.get('error')}")
        return message

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.send(message)
        return self.receive()

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model",
        default="/data/shared-models/Qwen3-0.6B",
    )
    parser.add_argument(
        "--target-model",
        default="/data/shared-models/Qwen3-8B",
    )
    parser.add_argument("--draft-device", default="4")
    parser.add_argument("--target-device", default="5")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--probe-max-tokens", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        default=None,
        help="Prompt to test; repeat the option to test multiple prompts.",
    )
    return parser


def launch_worker(
    worker_script: Path,
    role: str,
    model: str,
    device: str,
    socket_path: Path,
    max_model_len: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TORCHDYNAMO_DISABLE"] = "1"

    # These were reported as unknown by the user's current vLLM fork.  The
    # visible-device mapping above is sufficient: physical card N becomes
    # logical npu:0 inside this worker.
    env.pop("VLLM_NPU_DEVICE", None)
    env.pop("VLLM_USE_V1", None)

    command = [
        sys.executable,
        str(worker_script),
        "--role",
        role,
        "--model",
        model,
        "--socket",
        str(socket_path),
        "--max-model-len",
        str(max_model_len),
    ]

    print(
        f"[coordinator] launching {role}: card={device}, model={model}",
        flush=True,
    )
    return subprocess.Popen(command, env=env)


def connect_worker(
    socket_path: Path,
    process: subprocess.Popen,
    timeout: float,
) -> JsonSocketClient:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"worker exited before becoming ready, returncode={process.returncode}"
            )

        if socket_path.exists():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(socket_path))
                client = JsonSocketClient(sock)

                while True:
                    message = client.receive()
                    status = message.get("status")
                    print(f"[coordinator] {message}", flush=True)
                    if status == "ready":
                        return client
                    if status != "loading":
                        raise RuntimeError(
                            f"unexpected worker startup message: {message}"
                        )
            except Exception as exc:
                last_error = exc
                sock.close()

        time.sleep(0.2)

    raise TimeoutError(
        f"timed out waiting for worker socket {socket_path}; "
        f"last_error={last_error!r}"
    )


def run_token_round(
    draft_client: JsonSocketClient,
    target_client: JsonSocketClient,
    tokenizer,
    prefix_ids: list[int],
    gamma: int,
    probe_max_tokens: int,
    round_index: int,
) -> list[int]:

    print(
        f"[coordinator] round={round_index} prefix_len={len(prefix_ids)} "
        f"requesting gamma={gamma} draft tokens",
        flush=True,
    )

    draft_response = draft_client.request(
        {
            "cmd": "draft",
            "prompt_token_ids": prefix_ids,
            "max_tokens": gamma,
        }
    )
    print(f"[draft] {draft_response}", flush=True)

    if draft_response.get("status") != "result":
        raise RuntimeError(f"Draft did not return result: {draft_response}")

    draft_ids = [
        int(token_id) for token_id in draft_response.get("draft_token_ids", [])
    ]
    if not draft_ids:
        raise RuntimeError(f"Draft returned no token IDs: {draft_response}")

    candidate_ids = prefix_ids + draft_ids
    print(
        f"[coordinator] sending candidate prefix to Target: "
        f"prefix_len={len(prefix_ids)}, draft_len={len(draft_ids)}",
        flush=True,
    )

    target_response = target_client.request(
        {
            "cmd": "verify_greedy",
            "prompt_token_ids": candidate_ids,
            "draft_token_ids": draft_ids,
            "prefix_len": len(prefix_ids),
        }
    )
    print(f"[target] {target_response}", flush=True)

    if target_response.get("status") != "verify_result":
        raise RuntimeError(f"Target did not return verify_result: {target_response}")

    received_ids = target_response.get("input_token_ids")
    if received_ids != candidate_ids:
        raise RuntimeError(
            "Target did not receive the exact candidate token IDs: "
            f"expected_len={len(candidate_ids)}, "
            f"received_len={len(received_ids or [])}"
        )

    draft_text = tokenizer.decode(draft_ids, skip_special_tokens=False)
    target_ids = target_response.get("target_token_ids", [])
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    print(f"[coordinator] draft_text={draft_text!r}", flush=True)
    print(f"[coordinator] target_greedy_text={target_text!r}", flush=True)
    print(
        f"[coordinator] accepted_len={target_response['accepted_len']} "
        f"/ {target_response['draft_len']}, "
        f"bonus_token_id={target_response.get('bonus_token_id')}",
        flush=True,
    )

    committed_token_ids = target_response.get("committed_token_ids")
    if not isinstance(committed_token_ids, list) or not committed_token_ids:
        raise RuntimeError(
            f"Target returned no committed tokens: {target_response}"
        )

    next_prefix_ids = prefix_ids + [int(token_id) for token_id in committed_token_ids]
    print(
        f"[coordinator] committed_len={len(next_prefix_ids)} "
        f"(+{len(committed_token_ids)})",
        flush=True,
    )
    return next_prefix_ids


def stop_worker(name: str, client: JsonSocketClient) -> None:
    try:
        response = client.request({"cmd": "stop"})
        print(f"[{name}] {response}", flush=True)
    finally:
        client.close()


def main() -> None:
    args = build_parser().parse_args()
    worker_script = Path(__file__).with_name("pearl_worker.py")
    if not worker_script.exists():
        raise FileNotFoundError(worker_script)

    workers: list[tuple[str, subprocess.Popen, JsonSocketClient | None]] = []

    with tempfile.TemporaryDirectory(prefix="nano_pearl_ascend_") as temp_dir:
        temp_path = Path(temp_dir)
        definitions = [
            (
                "draft",
                args.draft_model,
                args.draft_device,
                temp_path / "draft.sock",
            ),
            (
                "target",
                args.target_model,
                args.target_device,
                temp_path / "target.sock",
            ),
        ]

        try:
            for role, model, device, socket_path in definitions:
                process = launch_worker(
                    worker_script,
                    role,
                    model,
                    device,
                    socket_path,
                    args.max_model_len,
                )
                workers.append((role, process, None))

            # Connect to both workers concurrently.  The worker starts model
            # loading only after accepting its socket connection, so doing
            # this sequentially would accidentally serialize Draft/Target
            # initialization.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                connect_futures = [
                    pool.submit(
                        connect_worker,
                        definitions[index][3],
                        process,
                        args.startup_timeout,
                    )
                    for index, (_, process, _) in enumerate(workers)
                ]
                for index, future in enumerate(connect_futures):
                    client = future.result()
                    role, process, _ = workers[index]
                    workers[index] = (role, process, client)

            print("[coordinator] both workers are ready", flush=True)

            clients = [(role, client) for role, _, client in workers if client]
            for role, client in clients:
                response = client.request({"cmd": "ping"})
                print(f"[{role}] {response}", flush=True)

            if args.gamma <= 0:
                raise ValueError("--gamma must be positive")
            if args.rounds <= 0:
                raise ValueError("--rounds must be positive")

            # Tokenization is CPU-side and does not touch either NPU.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.target_model,
                trust_remote_code=True,
            )

            draft_client = dict(clients)["draft"]
            target_client = dict(clients)["target"]
            prompts = args.prompts or ["请回答：1+1等于几？"]
            for prompt_index, prompt in enumerate(prompts, start=1):
                initial_prefix_ids = [
                    int(token_id)
                    for token_id in tokenizer.encode(
                        prompt,
                        add_special_tokens=True,
                    )
                ]
                prefix_ids = initial_prefix_ids
                speculative_token_ids: list[int] = []

                print(
                    f"[coordinator] ===== prompt {prompt_index}/{len(prompts)} "
                    f"{prompt!r} =====",
                    flush=True,
                )

                for round_index in range(1, args.rounds + 1):
                    old_prefix_len = len(prefix_ids)
                    prefix_ids = run_token_round(
                        draft_client,
                        target_client,
                        tokenizer,
                        prefix_ids,
                        args.gamma,
                        args.probe_max_tokens,
                        round_index,
                    )
                    speculative_token_ids.extend(prefix_ids[old_prefix_len:])

                # Compare the speculative committed tokens against a normal
                # greedy Target generation of the same number of tokens.
                # This validates the rejection/commit logic before KV-cache
                # reuse is introduced.
                baseline_response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids": initial_prefix_ids,
                        "max_tokens": len(speculative_token_ids),
                    }
                )
                baseline_token_ids = [
                    int(token_id)
                    for token_id in baseline_response.get("token_ids", [])
                ]
                is_equal = baseline_token_ids == speculative_token_ids
                print(
                    f"[coordinator] target-only token_ids={baseline_token_ids}",
                    flush=True,
                )
                print(
                    f"[coordinator] speculative token_ids="
                    f"{speculative_token_ids}",
                    flush=True,
                )
                print(
                    f"[coordinator] correctness_match={is_equal}; "
                    f"final_prefix_len={len(prefix_ids)}",
                    flush=True,
                )
                if not is_equal:
                    raise RuntimeError(
                        "greedy speculative output does not match Target-only "
                        f"output for prompt {prompt!r}"
                    )

            print("[coordinator] greedy verification correctness passed", flush=True)

        finally:
            for role, process, client in workers:
                if client is not None:
                    try:
                        stop_worker(role, client)
                    except Exception as exc:
                        print(f"[{role}] stop failed: {exc!r}", flush=True)

                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"[{role}] force terminating worker", flush=True)
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        print("[coordinator] finished", flush=True)


if __name__ == "__main__":
    main()
