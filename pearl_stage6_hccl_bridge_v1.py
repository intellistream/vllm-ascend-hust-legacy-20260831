#!/usr/bin/env python3
"""HCCL sidecar bridge for the two-process Stage-6 token path."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any


REQUEST_MAGIC = 1601
RESPONSE_MAGIC = 1602
ERROR_MAGIC = 1603


def send_message(conn: socket.socket, message: dict[str, Any]) -> None:
    conn.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))


def receive_message(reader) -> dict[str, Any] | None:
    line = reader.readline()
    return json.loads(line) if line else None


def bind_server(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    return server


def connect_unix(path: Path, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(max(1.0, deadline - time.monotonic()))
            sock.connect(str(path))
            sock.settimeout(None)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(0.1)
    raise TimeoutError(
        f"timed out connecting to local socket {path}; "
        f"last_error={last_error!r}"
    )


def send_i32(dist, values: list[int], device, dst: int) -> None:
    import torch

    if not values:
        return
    tensor = torch.tensor(values, dtype=torch.int32, device=device)
    dist.send(tensor, dst=dst)


def recv_i32(dist, count: int, device, src: int) -> list[int]:
    import torch

    if count < 0:
        raise RuntimeError(f"negative HCCL receive count: {count}")
    if count == 0:
        return []
    tensor = torch.empty(count, dtype=torch.int32, device=device)
    dist.recv(tensor, src=src)
    return [int(value) for value in tensor.cpu().tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("draft", "target"), required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--draft-socket")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    return parser.parse_args()


def init_hccl():
    # torch/torch_npu must be imported only after the parent has assigned the
    # one-device ASCEND_RT_VISIBLE_DEVICES value.
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or rank not in (0, 1):
        raise RuntimeError(
            f"Stage-6 HCCL bridge requires rank/world_size 0|1/2, "
            f"got rank={rank}, world_size={world_size}"
        )
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return torch, dist, rank, device


def run_target_bridge(
    server: socket.socket,
    dist,
    device,
    max_model_len: int,
) -> None:
    conn, _ = server.accept()
    sequence = 0
    with conn:
        with conn.makefile("r", encoding="utf-8") as reader:
            while True:
                message = receive_message(reader)
                if message is None:
                    return
                if message.get("cmd") != "draft":
                    send_message(
                        conn,
                        {
                            "status": "error",
                            "error": f"unknown HCCL bridge command: {message.get('cmd')!r}",
                        },
                    )
                    continue
                prefix = message.get("prefix_token_ids")
                gamma = int(message.get("gamma", 0))
                if not isinstance(prefix, list) or not prefix:
                    raise ValueError("prefix_token_ids must be a non-empty list")
                if len(prefix) > max_model_len:
                    raise ValueError(
                        f"prefix length {len(prefix)} exceeds max_model_len {max_model_len}"
                    )
                if gamma <= 0:
                    raise ValueError(f"gamma must be positive, got {gamma}")

                sequence += 1
                send_i32(
                    dist,
                    [REQUEST_MAGIC, sequence, len(prefix), gamma],
                    device,
                    dst=0,
                )
                send_i32(
                    dist,
                    [int(token_id) for token_id in prefix],
                    device,
                    dst=0,
                )

                header = recv_i32(dist, 3, device, src=0)
                if len(header) != 3 or header[0] not in (RESPONSE_MAGIC, ERROR_MAGIC):
                    raise RuntimeError(f"invalid HCCL response header: {header}")
                if header[1] != sequence:
                    raise RuntimeError(
                        f"HCCL response sequence mismatch: expected {sequence}, "
                        f"got {header[1]}"
                    )
                draft_len = header[2]
                if header[0] == ERROR_MAGIC:
                    send_message(
                        conn,
                        {
                            "status": "error",
                            "error": "Draft sidecar returned an HCCL bridge error",
                        },
                    )
                    continue
                if draft_len < 0 or draft_len > gamma:
                    raise RuntimeError(
                        f"invalid draft length {draft_len} for gamma {gamma}"
                    )
                draft_ids = recv_i32(dist, draft_len, device, src=0)
                send_message(
                    conn,
                    {
                        "status": "result",
                        "draft_token_ids": draft_ids,
                        "prefix_len": len(prefix),
                    },
                )
                print(
                    "[stage6-hccl] target bridge "
                    f"seq={sequence} prefix_len={len(prefix)} draft_len={draft_len}",
                    flush=True,
                )


def run_draft_bridge(
    worker_socket: Path,
    dist,
    device,
    socket_timeout: float,
) -> None:
    worker_conn = connect_unix(worker_socket, socket_timeout)
    with worker_conn:
        with worker_conn.makefile("r", encoding="utf-8") as worker_reader:
            sequence = 0
            while True:
                header = recv_i32(dist, 4, device, src=1)
                if header[0] != REQUEST_MAGIC:
                    raise RuntimeError(f"invalid HCCL request header: {header}")
                sequence = header[1]
                prefix_len = header[2]
                gamma = header[3]
                if prefix_len <= 0 or gamma <= 0:
                    raise RuntimeError(
                        f"invalid HCCL request lengths: prefix_len={prefix_len}, "
                        f"gamma={gamma}"
                    )
                prefix = recv_i32(dist, prefix_len, device, src=1)
                send_message(
                    worker_conn,
                    {
                        "cmd": "draft",
                        "request_id": f"hccl-{sequence}",
                        "prefix_token_ids": prefix,
                        "gamma": gamma,
                    },
                )
                response = receive_message(worker_reader)
                if response is None:
                    raise ConnectionError("Draft worker closed its proposal socket")
                if response.get("status") == "error":
                    send_i32(dist, [ERROR_MAGIC, sequence, 0], device, dst=1)
                    print(
                        "[stage6-hccl] draft bridge worker error "
                        f"seq={sequence}: {response.get('error')}",
                        flush=True,
                    )
                    continue
                draft_ids = [
                    int(token_id)
                    for token_id in response.get("draft_token_ids", [])[:gamma]
                ]
                send_i32(
                    dist,
                    [RESPONSE_MAGIC, sequence, len(draft_ids)],
                    device,
                    dst=1,
                )
                send_i32(dist, draft_ids, device, dst=1)
                print(
                    "[stage6-hccl] draft bridge "
                    f"seq={sequence} prefix_len={prefix_len} draft_len={len(draft_ids)}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.role == "draft" and not args.draft_socket:
        raise ValueError("draft role requires --draft-socket")
    torch, dist, rank, device = init_hccl()
    server = None
    try:
        if args.role == "target":
            if rank != 1:
                raise RuntimeError(f"target bridge must use rank 1, got {rank}")
            server = bind_server(Path(args.socket))
            print(
                "[stage6-hccl] ready role=target rank=1 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"socket={args.socket}",
                flush=True,
            )
            run_target_bridge(server, dist, device, args.max_model_len)
        else:
            if rank != 0:
                raise RuntimeError(f"draft bridge must use rank 0, got {rank}")
            print(
                "[stage6-hccl] ready role=draft rank=0 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"worker_socket={args.draft_socket}",
                flush=True,
            )
            run_draft_bridge(
                Path(args.draft_socket),
                dist,
                device,
                args.socket_timeout,
            )
    finally:
        if server is not None:
            server.close()
            try:
                Path(args.socket).unlink()
            except FileNotFoundError:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
