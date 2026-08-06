#!/usr/bin/env python3
"""HCCL sidecar for Stage-5 single-request and batch Draft proposals.

v1 carried only ``cmd=draft``.  Stage-5 batch>1 sends ``cmd=draft_batch``
with a list of request IDs, prefixes, and gamma values.  v2 keeps the v1
single-request wire format and adds an ordered batch wire format.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pearl_stage6_hccl_bridge_v1 as base


BATCH_REQUEST_MAGIC = 1604
BATCH_RESPONSE_MAGIC = 1605


def _validate_request(
    request: dict[str, Any],
    max_model_len: int,
    index: int,
) -> tuple[str, list[int], int]:
    request_id = str(request.get("request_id", f"row-{index}"))
    prefix = request.get("prefix_token_ids")
    gamma = int(request.get("gamma", 0))
    if not isinstance(prefix, list) or not prefix:
        raise ValueError(f"prefix_token_ids must be non-empty for {request_id!r}")
    if len(prefix) > max_model_len:
        raise ValueError(
            f"prefix length {len(prefix)} exceeds max_model_len {max_model_len}"
        )
    if gamma <= 0:
        raise ValueError(f"gamma must be positive for {request_id!r}")
    return request_id, [int(token_id) for token_id in prefix], gamma


def _send_single_request(
    dist,
    device,
    sequence: int,
    prefix: list[int],
    gamma: int,
) -> None:
    base.send_i32(
        dist,
        [base.REQUEST_MAGIC, sequence, len(prefix), gamma],
        device,
        dst=0,
    )
    base.send_i32(dist, prefix, device, dst=0)


def _send_batch_request(
    dist,
    device,
    sequence: int,
    requests: list[tuple[str, list[int], int]],
) -> None:
    base.send_i32(
        dist,
        [BATCH_REQUEST_MAGIC, sequence, len(requests), 0],
        device,
        dst=0,
    )
    for _request_id, prefix, gamma in requests:
        base.send_i32(dist, [len(prefix), gamma], device, dst=0)
        base.send_i32(dist, prefix, device, dst=0)


def _recv_single_response(
    dist,
    device,
    sequence: int,
) -> tuple[bool, list[int]]:
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) != 3 or header[0] not in (
        base.RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(f"invalid HCCL response header: {header}")
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL response sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False, []
    draft_len = header[2]
    if draft_len < 0:
        raise RuntimeError(f"invalid single draft length: {draft_len}")
    return True, base.recv_i32(dist, draft_len, device, src=0)


def _recv_batch_response(
    dist,
    device,
    sequence: int,
    request_ids: list[str],
) -> tuple[bool, list[list[int]]]:
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) != 3 or header[0] not in (
        BATCH_RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(f"invalid HCCL batch response header: {header}")
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL batch response sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False, []
    count = header[2]
    if count != len(request_ids):
        raise RuntimeError(
            f"HCCL batch response count mismatch: expected {len(request_ids)}, "
            f"got {count}"
        )
    result_by_index: dict[int, list[int]] = {}
    for _ in range(count):
        item_header = base.recv_i32(dist, 2, device, src=0)
        index, draft_len = item_header
        if index < 0 or index >= count:
            raise RuntimeError(f"invalid HCCL batch response index: {index}")
        if draft_len < 0:
            raise RuntimeError(f"invalid HCCL batch draft length: {draft_len}")
        result_by_index[index] = base.recv_i32(
            dist, draft_len, device, src=0
        )
    if len(result_by_index) != count:
        raise RuntimeError("duplicate or missing HCCL batch response index")
    return True, [result_by_index[index] for index in range(count)]


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
                message = base.receive_message(reader)
                if message is None:
                    return
                command = message.get("cmd")
                try:
                    if command == "draft":
                        request = {
                            "request_id": str(message.get("request_id", "target-0")),
                            "prefix_token_ids": message.get("prefix_token_ids"),
                            "gamma": message.get("gamma", 0),
                        }
                        request_id, prefix, gamma = _validate_request(
                            request, max_model_len, 0
                        )
                        sequence += 1
                        _send_single_request(dist, device, sequence, prefix, gamma)
                        ok, draft_ids = _recv_single_response(
                            dist, device, sequence
                        )
                        if not ok:
                            raise RuntimeError(
                                "Draft sidecar returned an HCCL bridge error"
                            )
                        base.send_message(
                            conn,
                            {
                                "status": "result",
                                "draft_token_ids": draft_ids[:gamma],
                                "prefix_len": len(prefix),
                            },
                        )
                        print(
                            "[stage6-hccl] target bridge "
                            f"seq={sequence} prefix_len={len(prefix)} "
                            f"draft_len={len(draft_ids[:gamma])}",
                            flush=True,
                        )
                        continue

                    if command != "draft_batch":
                        raise ValueError(f"unknown HCCL bridge command: {command!r}")
                    raw_requests = message.get("requests")
                    if not isinstance(raw_requests, list) or not raw_requests:
                        raise ValueError(
                            "draft_batch requires a non-empty requests list"
                        )
                    requests = [
                        _validate_request(item, max_model_len, index)
                        for index, item in enumerate(raw_requests)
                    ]
                    sequence += 1
                    _send_batch_request(dist, device, sequence, requests)
                    request_ids = [item[0] for item in requests]
                    ok, draft_rows = _recv_batch_response(
                        dist, device, sequence, request_ids
                    )
                    if not ok:
                        raise RuntimeError(
                            "Draft batch sidecar returned an HCCL bridge error"
                        )
                    results = [
                        {
                            "request_id": request_id,
                            "draft_token_ids": draft_ids[: requests[index][2]],
                        }
                        for index, (request_id, _prefix, _gamma) in enumerate(requests)
                        for draft_ids in [draft_rows[index]]
                    ]
                    base.send_message(
                        conn,
                        {"status": "result", "results": results},
                    )
                    print(
                        "[stage6-hccl] target bridge batch "
                        f"seq={sequence} batch_size={len(requests)} "
                        f"draft_tokens={sum(len(row) for row in draft_rows)}",
                        flush=True,
                    )
                except Exception as exc:
                    base.send_message(
                        conn,
                        {"status": "error", "error": repr(exc)},
                    )


def run_draft_bridge(
    worker_socket: Path,
    dist,
    device,
    socket_timeout: float,
) -> None:
    worker_conn = base.connect_unix(worker_socket, socket_timeout)
    with worker_conn:
        with worker_conn.makefile("r", encoding="utf-8") as worker_reader:
            while True:
                header = base.recv_i32(dist, 4, device, src=1)
                magic, sequence, count_or_prefix, gamma_or_zero = header
                if magic == base.REQUEST_MAGIC:
                    prefix_len = count_or_prefix
                    gamma = gamma_or_zero
                    if prefix_len <= 0 or gamma <= 0:
                        raise RuntimeError(
                            f"invalid HCCL request lengths: prefix_len={prefix_len}, "
                            f"gamma={gamma}"
                        )
                    prefix = base.recv_i32(dist, prefix_len, device, src=1)
                    base.send_message(
                        worker_conn,
                        {
                            "cmd": "draft",
                            "request_id": f"hccl-{sequence}",
                            "prefix_token_ids": prefix,
                            "gamma": gamma,
                        },
                    )
                    response = base.receive_message(worker_reader)
                    if response is None:
                        raise ConnectionError(
                            "Draft worker closed its proposal socket"
                        )
                    if response.get("status") == "error":
                        base.send_i32(
                            dist,
                            [base.ERROR_MAGIC, sequence, 0],
                            device,
                            dst=1,
                        )
                        continue
                    draft_ids = [
                        int(token_id)
                        for token_id in response.get("draft_token_ids", [])[:gamma]
                    ]
                    base.send_i32(
                        dist,
                        [base.RESPONSE_MAGIC, sequence, len(draft_ids)],
                        device,
                        dst=1,
                    )
                    base.send_i32(dist, draft_ids, device, dst=1)
                    print(
                        "[stage6-hccl] draft bridge "
                        f"seq={sequence} prefix_len={prefix_len} "
                        f"draft_len={len(draft_ids)}",
                        flush=True,
                    )
                    continue

                if magic != BATCH_REQUEST_MAGIC:
                    raise RuntimeError(f"invalid HCCL request header: {header}")
                batch_size = count_or_prefix
                if batch_size <= 0:
                    raise RuntimeError(f"invalid HCCL batch size: {batch_size}")
                requests: list[dict[str, Any]] = []
                for index in range(batch_size):
                    item_header = base.recv_i32(dist, 2, device, src=1)
                    prefix_len, gamma = item_header
                    if prefix_len <= 0 or gamma <= 0:
                        raise RuntimeError(
                            f"invalid HCCL batch request lengths at row {index}: "
                            f"prefix_len={prefix_len}, gamma={gamma}"
                        )
                    prefix = base.recv_i32(dist, prefix_len, device, src=1)
                    requests.append(
                        {
                            "request_id": f"hccl-{sequence}-{index}",
                            "prefix_token_ids": prefix,
                            "gamma": gamma,
                        }
                    )
                base.send_message(
                    worker_conn,
                    {"cmd": "draft_batch", "requests": requests},
                )
                response = base.receive_message(worker_reader)
                if response is None:
                    raise ConnectionError("Draft worker closed its proposal socket")
                if response.get("status") == "error":
                    base.send_i32(
                        dist,
                        [base.ERROR_MAGIC, sequence, 0],
                        device,
                        dst=1,
                    )
                    print(
                        "[stage6-hccl] draft batch worker error "
                        f"seq={sequence}: {response.get('error')}",
                        flush=True,
                    )
                    continue
                raw_results = response.get("results")
                if not isinstance(raw_results, list) or len(raw_results) != batch_size:
                    raise RuntimeError(
                        "Draft batch response must contain one result per row"
                    )
                result_by_id = {
                    str(item.get("request_id")): item
                    for item in raw_results
                    if isinstance(item, dict)
                }
                base.send_i32(
                    dist,
                    [BATCH_RESPONSE_MAGIC, sequence, batch_size],
                    device,
                    dst=1,
                )
                for index, request in enumerate(requests):
                    item = result_by_id.get(request["request_id"])
                    if item is None:
                        raise RuntimeError(
                            f"Draft batch response omitted {request['request_id']!r}"
                        )
                    draft_ids = [
                        int(token_id)
                        for token_id in item.get("draft_token_ids", [])[
                            : request["gamma"]
                        ]
                    ]
                    base.send_i32(
                        dist,
                        [index, len(draft_ids)],
                        device,
                        dst=1,
                    )
                    base.send_i32(dist, draft_ids, device, dst=1)
                print(
                    "[stage6-hccl] draft bridge batch "
                    f"seq={sequence} batch_size={batch_size}",
                    flush=True,
                )


def main() -> None:
    args = base.parse_args()
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.role == "draft" and not args.draft_socket:
        raise ValueError("draft role requires --draft-socket")
    _torch, dist, rank, device = base.init_hccl()
    server = None
    try:
        if args.role == "target":
            if rank != 1:
                raise RuntimeError(f"target bridge must use rank 1, got {rank}")
            server = base.bind_server(Path(args.socket))
            print(
                "[stage6-hccl] ready role=target rank=1 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"socket={args.socket} batch_protocol=v2",
                flush=True,
            )
            run_target_bridge(server, dist, device, args.max_model_len)
        else:
            if rank != 0:
                raise RuntimeError(f"draft bridge must use rank 0, got {rank}")
            print(
                "[stage6-hccl] ready role=draft rank=0 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"worker_socket={args.draft_socket} batch_protocol=v2",
                flush=True,
            )
            run_draft_bridge(
                Path(args.draft_socket), dist, device, args.socket_timeout
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
