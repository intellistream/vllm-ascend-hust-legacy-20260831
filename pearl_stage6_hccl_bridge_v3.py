#!/usr/bin/env python3
"""HCCL bridge v3 with stable Draft request slots.

The v2 bridge generated a new Draft request ID from the HCCL sequence on every
round. v3 assigns a stable integer slot to each Target request ID and carries
that slot over HCCL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pearl_stage6_hccl_bridge_v1 as base


SINGLE_REQUEST_MAGIC = 1606
SINGLE_RESPONSE_MAGIC = 1607
BATCH_REQUEST_MAGIC = 1608
BATCH_RESPONSE_MAGIC = 1609
COMMIT_REQUEST_MAGIC = 1612
COMMIT_RESPONSE_MAGIC = 1613
REBASE_REQUEST_MAGIC = 1614
REBASE_RESPONSE_MAGIC = 1615
COMMIT_ROW_LENGTH_ONLY = 0
COMMIT_ROW_FULL = 1


def validate_request(
    request: dict[str, Any], max_model_len: int, index: int
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


def recv_single_response(dist, device, sequence: int):
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) != 3 or header[0] not in (
        SINGLE_RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(f"invalid HCCL v3 single response header: {header}")
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL v3 response sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False, []
    draft_len = header[2]
    if draft_len < 0:
        raise RuntimeError(f"invalid HCCL v3 draft length: {draft_len}")
    return True, base.recv_i32(dist, draft_len, device, src=0)


def recv_batch_response(dist, device, sequence: int, count: int):
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) not in (3,) or header[0] not in (
        BATCH_RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(f"invalid HCCL v3 batch response header: {header}")
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL v3 batch sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False, []
    if header[2] != count:
        raise RuntimeError(
            f"HCCL v3 batch response count mismatch: expected {count}, "
            f"got {header[2]}"
        )
    rows: dict[int, list[int]] = {}
    for _ in range(count):
        index, draft_len = base.recv_i32(dist, 2, device, src=0)
        if index < 0 or index >= count or index in rows:
            raise RuntimeError(f"invalid HCCL v3 batch response index: {index}")
        if draft_len < 0:
            raise RuntimeError(f"invalid HCCL v3 batch draft length: {draft_len}")
        rows[index] = base.recv_i32(dist, draft_len, device, src=0)
    if len(rows) != count:
        raise RuntimeError("missing HCCL v3 batch response row")
    return True, [rows[index] for index in range(count)]


def recv_commit_response(dist, device, sequence: int, count: int) -> bool:
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) != 3 or header[0] not in (
        COMMIT_RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(
            f"invalid HCCL v3 commit response header: {header}"
        )
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL v3 commit sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False
    if header[2] != count:
        raise RuntimeError(
            f"HCCL v3 commit count mismatch: expected {count}, "
            f"got {header[2]}"
        )
    return True



def recv_rebase_response(
    dist, device, sequence: int, count: int
) -> bool:
    header = base.recv_i32(dist, 3, device, src=0)
    if len(header) != 3 or header[0] not in (
        REBASE_RESPONSE_MAGIC,
        base.ERROR_MAGIC,
    ):
        raise RuntimeError(
            f"invalid HCCL v3 rebase response header: {header}"
        )
    if header[1] != sequence:
        raise RuntimeError(
            f"HCCL v3 rebase sequence mismatch: expected {sequence}, "
            f"got {header[1]}"
        )
    if header[0] == base.ERROR_MAGIC:
        return False
    if header[2] != count:
        raise RuntimeError(
            f"HCCL v3 rebase count mismatch: expected {count}, "
            f"got {header[2]}"
        )
    return True


def run_target_bridge(server, dist, device, max_model_len: int) -> None:
    conn, _ = server.accept()
    sequence = 0
    next_slot = 0
    slots: dict[str, int] = {}

    def slot_for(request_id: str) -> int:
        nonlocal next_slot
        request_id = str(request_id)
        if request_id not in slots:
            slots[request_id] = next_slot
            next_slot += 1
        return slots[request_id]

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
                            "request_id": str(
                                message.get("request_id", "target-0")
                            ),
                            "prefix_token_ids": message.get("prefix_token_ids"),
                            "gamma": message.get("gamma", 0),
                        }
                        request_id, prefix, gamma = validate_request(
                            request, max_model_len, 0
                        )
                        sequence += 1
                        slot = slot_for(request_id)
                        base.send_i32(
                            dist,
                            [
                                SINGLE_REQUEST_MAGIC,
                                sequence,
                                slot,
                                len(prefix),
                                gamma,
                            ],
                            device,
                            dst=0,
                        )
                        base.send_i32(dist, prefix, device, dst=0)
                        ok, draft_ids = recv_single_response(
                            dist, device, sequence
                        )
                        if not ok:
                            raise RuntimeError(
                                "Draft sidecar returned an HCCL v3 error"
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
                            "[stage6-hccl] target bridge v3 "
                            f"seq={sequence} slot={slot} "
                            f"prefix_len={len(prefix)} "
                            f"draft_len={len(draft_ids[:gamma])}",
                            flush=True,
                        )
                        continue

                    if command == "commit_batch":
                        raw_updates = message.get("updates")
                        if not isinstance(raw_updates, list) or not raw_updates:
                            raise ValueError(
                                "commit_batch requires a non-empty updates list"
                            )

                        wire_updates = []
                        for item in raw_updates:
                            if not isinstance(item, dict):
                                raise ValueError("commit_batch update must be a dict")

                            request_id = str(item.get("request_id", ""))
                            if request_id not in slots:
                                raise ValueError(
                                    f"unknown HCCL v3 request slot: {request_id!r}"
                                )

                            length_only = item.get("length_only", False)
                            if not isinstance(length_only, bool):
                                raise ValueError(
                                    "commit_batch length_only must be boolean"
                                )

                            accepted_len = int(item["accepted_len"])
                            valid_len = int(item["valid_len"])
                            if accepted_len < 0 or valid_len < 0:
                                raise ValueError(
                                    "accepted_len and valid_len must be non-negative"
                                )

                            if length_only:
                                if item.get("prefix_token_ids") is not None:
                                    raise ValueError(
                                        "length-only HCCL commit must not carry "
                                        "prefix_token_ids"
                                    )
                                wire_updates.append(
                                    {
                                        "slot": slots[request_id],
                                        "mode": COMMIT_ROW_LENGTH_ONLY,
                                        "accepted_len": accepted_len,
                                        "valid_len": valid_len,
                                    }
                                )
                                continue

                            prefix = item.get("prefix_token_ids")
                            if not isinstance(prefix, list) or not prefix:
                                raise ValueError(
                                    "full HCCL commit requires non-empty "
                                    "prefix_token_ids"
                                )
                            prefix = [int(token_id) for token_id in prefix]

                            target_prefix_len = int(
                                item.get("target_prefix_len", len(prefix))
                            )
                            if target_prefix_len != len(prefix):
                                raise ValueError(
                                    "target_prefix_len must equal prefix length"
                                )

                            gamma = int(item.get("gamma", 0))
                            draft_len = int(item.get("draft_len", 0))
                            if gamma < 0 or draft_len < 0:
                                raise ValueError(
                                    "gamma and draft_len must be non-negative"
                                )

                            replacement = item.get("replacement_token_id")
                            replacement_token_id = (
                                -1 if replacement is None else int(replacement)
                            )
                            if replacement_token_id < -1:
                                raise ValueError(
                                    "replacement_token_id must be >= -1"
                                )

                            finished = 1 if bool(item.get("finished", False)) else 0

                            wire_updates.append(
                                {
                                    "slot": slots[request_id],
                                    "mode": COMMIT_ROW_FULL,
                                    "accepted_len": accepted_len,
                                    "valid_len": valid_len,
                                    "gamma": gamma,
                                    "draft_len": draft_len,
                                    "target_prefix_len": target_prefix_len,
                                    "replacement_token_id": replacement_token_id,
                                    "finished": finished,
                                    "prefix": prefix,
                                }
                            )

                        sequence += 1
                        base.send_i32(
                            dist,
                            [COMMIT_REQUEST_MAGIC, sequence, len(wire_updates), 0, 0],
                            device,
                            dst=0,
                        )
                        for update in wire_updates:
                            base.send_i32(
                                dist,
                                [
                                    update["slot"],
                                    update["mode"],
                                    update["accepted_len"],
                                    update["valid_len"],
                                ],
                                device,
                                dst=0,
                            )
                            if update["mode"] == COMMIT_ROW_FULL:
                                base.send_i32(
                                    dist,
                                    [
                                        update["gamma"],
                                        update["draft_len"],
                                        update["target_prefix_len"],
                                        update["replacement_token_id"],
                                        update["finished"],
                                        len(update["prefix"]),
                                    ],
                                    device,
                                    dst=0,
                                )
                                base.send_i32(
                                    dist,
                                    update["prefix"],
                                    device,
                                    dst=0,
                                )

                        ok = recv_commit_response(
                            dist, device, sequence, len(wire_updates)
                        )
                        if not ok:
                            raise RuntimeError(
                                "Draft batch sidecar returned an HCCL v3 commit error"
                            )
                        base.send_message(conn, {"status": "result"})
                        print(
                            "[stage6-hccl] target bridge commit_batch "
                            f"v3 seq={sequence} updates={len(wire_updates)}",
                            flush=True,
                        )
                        continue

                    if command == "rebase_batch":
                        raw_requests = message.get("requests")
                        if not isinstance(raw_requests, list) or not raw_requests:
                            raise ValueError(
                                "rebase_batch requires a non-empty requests list"
                            )

                        wire_requests = []
                        for index, item in enumerate(raw_requests):
                            if not isinstance(item, dict):
                                raise ValueError(
                                    f"rebase request {index} must be a dict"
                                )

                            request_id = str(item.get("request_id", ""))
                            if request_id not in slots:
                                raise ValueError(
                                    f"unknown HCCL v3 request slot: {request_id!r}"
                                )

                            prefix = item.get("prefix_token_ids")
                            if not isinstance(prefix, list) or not prefix:
                                raise ValueError(
                                    "rebase prefix_token_ids must be non-empty"
                                )

                            prefix = [int(token_id) for token_id in prefix]
                            if len(prefix) > max_model_len:
                                raise ValueError(
                                    f"rebase prefix length {len(prefix)} exceeds "
                                    f"max_model_len {max_model_len}"
                                )

                            wire_requests.append((slots[request_id], prefix))

                        sequence += 1
                        base.send_i32(
                            dist,
                            [
                                REBASE_REQUEST_MAGIC,
                                sequence,
                                len(wire_requests),
                                0,
                                0,
                            ],
                            device,
                            dst=0,
                        )

                        for slot, prefix in wire_requests:
                            base.send_i32(
                                dist,
                                [slot, len(prefix), 0],
                                device,
                                dst=0,
                            )
                            base.send_i32(dist, prefix, device, dst=0)

                        ok = recv_rebase_response(
                            dist,
                            device,
                            sequence,
                            len(wire_requests),
                        )
                        if not ok:
                            raise RuntimeError(
                                "Draft batch sidecar returned an HCCL "
                                "v3 rebase error"
                            )

                        base.send_message(conn, {"status": "result"})
                        print(
                            "[stage6-hccl] target bridge rebase_batch "
                            f"v3 seq={sequence} rows={len(wire_requests)}",
                            flush=True,
                        )
                        continue

                    if command != "draft_batch":
                        raise ValueError(
                            f"unknown HCCL bridge command: {command!r}"
                        )
                    raw_requests = message.get("requests")
                    if not isinstance(raw_requests, list) or not raw_requests:
                        raise ValueError(
                            "draft_batch requires a non-empty requests list"
                        )
                    requests = [
                        validate_request(item, max_model_len, index)
                        for index, item in enumerate(raw_requests)
                    ]
                    sequence += 1
                    base.send_i32(
                        dist,
                        [BATCH_REQUEST_MAGIC, sequence, len(requests), 0, 0],
                        device,
                        dst=0,
                    )
                    row_slots: list[int] = []
                    for request_id, prefix, gamma in requests:
                        slot = slot_for(request_id)
                        row_slots.append(slot)
                        base.send_i32(
                            dist,
                            [slot, len(prefix), gamma],
                            device,
                            dst=0,
                        )
                        base.send_i32(dist, prefix, device, dst=0)

                    ok, draft_rows = recv_batch_response(
                        dist, device, sequence, len(requests)
                    )
                    if not ok:
                        raise RuntimeError(
                            "Draft batch sidecar returned an HCCL v3 error"
                        )
                    results = [
                        {
                            "request_id": requests[index][0],
                            "draft_token_ids": draft_rows[index][
                                : requests[index][2]
                            ],
                        }
                        for index in range(len(requests))
                    ]
                    base.send_message(
                        conn, {"status": "result", "results": results}
                    )
                    print(
                        "[stage6-hccl] target bridge batch v3 "
                        f"seq={sequence} batch_size={len(requests)} "
                        f"slots={row_slots} "
                        f"draft_tokens={sum(len(row) for row in draft_rows)}",
                        flush=True,
                    )
                except Exception as exc:
                    base.send_message(
                        conn, {"status": "error", "error": repr(exc)}
                    )


def run_draft_bridge(
    worker_socket: Path, dist, device, socket_timeout: float
) -> None:
    worker_conn = base.connect_unix(worker_socket, socket_timeout)
    with worker_conn:
        with worker_conn.makefile("r", encoding="utf-8") as worker_reader:
            while True:
                header = base.recv_i32(dist, 5, device, src=1)
                magic, sequence, field2, field3, field4 = header

                if magic == SINGLE_REQUEST_MAGIC:
                    slot = field2
                    prefix_len = field3
                    gamma = field4
                    if slot < 0 or prefix_len <= 0 or gamma <= 0:
                        raise RuntimeError(
                            f"invalid HCCL v3 single request header: {header}"
                        )
                    prefix = base.recv_i32(
                        dist, prefix_len, device, src=1
                    )
                    request_id = f"hccl-slot-{slot}"
                    base.send_message(
                        worker_conn,
                        {
                            "cmd": "draft",
                            "request_id": request_id,
                            "prefix_token_ids": prefix,
                            "gamma": gamma,
                        },
                    )
                    response = base.receive_message(worker_reader)
                    if response is None or response.get("status") == "error":
                        base.send_i32(
                            dist,
                            [base.ERROR_MAGIC, sequence, 0],
                            device,
                            dst=1,
                        )
                        continue
                    draft_ids = [
                        int(x)
                        for x in response.get("draft_token_ids", [])[:gamma]
                    ]
                    base.send_i32(
                        dist,
                        [SINGLE_RESPONSE_MAGIC, sequence, len(draft_ids)],
                        device,
                        dst=1,
                    )
                    base.send_i32(dist, draft_ids, device, dst=1)
                    print(
                        "[stage6-hccl] draft bridge v3 "
                        f"seq={sequence} slot={slot} "
                        f"draft_len={len(draft_ids)}",
                        flush=True,
                    )
                    continue

                if magic == COMMIT_REQUEST_MAGIC:
                    count = field2
                    if count <= 0:
                        raise RuntimeError(
                            f"invalid HCCL v3 commit count: {count}"
                        )

                    updates = []
                    for _ in range(count):
                        row_header = base.recv_i32(
                            dist, 4, device, src=1
                        )
                        if len(row_header) != 4:
                            raise RuntimeError(
                                f"invalid HCCL v3 commit row header: {row_header}"
                            )

                        slot, mode, accepted_len, valid_len = row_header
                        if slot < 0 or accepted_len < 0 or valid_len < 0:
                            raise RuntimeError(
                                "invalid HCCL v3 commit update: "
                                f"{slot}, {accepted_len}, {valid_len}"
                            )

                        if mode == COMMIT_ROW_LENGTH_ONLY:
                            updates.append(
                                {
                                    "request_id": f"hccl-slot-{slot}",
                                    "length_only": True,
                                    "accepted_len": accepted_len,
                                    "valid_len": valid_len,
                                }
                            )
                            continue

                        if mode != COMMIT_ROW_FULL:
                            raise RuntimeError(
                                f"unknown HCCL v3 commit row mode: {mode}"
                            )

                        metadata = base.recv_i32(
                            dist, 6, device, src=1
                        )
                        if len(metadata) != 6:
                            raise RuntimeError(
                                f"invalid HCCL v3 full commit metadata: {metadata}"
                            )

                        (
                            gamma,
                            draft_len,
                            target_prefix_len,
                            replacement_token_id,
                            finished,
                            prefix_len,
                        ) = metadata

                        if (
                            gamma < 0
                            or draft_len < 0
                            or target_prefix_len <= 0
                            or prefix_len <= 0
                            or target_prefix_len != prefix_len
                            or replacement_token_id < -1
                            or finished not in (0, 1)
                        ):
                            raise RuntimeError(
                                "invalid HCCL v3 full commit metadata: "
                                f"{metadata}"
                            )

                        prefix = base.recv_i32(
                            dist, prefix_len, device, src=1
                        )

                        updates.append(
                            {
                                "request_id": f"hccl-slot-{slot}",
                                "length_only": False,
                                "accepted_len": accepted_len,
                                "valid_len": valid_len,
                                "gamma": gamma,
                                "draft_len": draft_len,
                                "target_prefix_len": target_prefix_len,
                                "replacement_token_id": (
                                    None
                                    if replacement_token_id == -1
                                    else replacement_token_id
                                ),
                                "finished": bool(finished),
                                "prefix_token_ids": prefix,
                            }
                        )

                    base.send_message(
                        worker_conn,
                        {"cmd": "commit_batch", "updates": updates},
                    )
                    response = base.receive_message(worker_reader)
                    if response is None or response.get("status") != "result":
                        base.send_i32(
                            dist,
                            [base.ERROR_MAGIC, sequence, 0],
                            device,
                            dst=1,
                        )
                        continue

                    base.send_i32(
                        dist,
                        [COMMIT_RESPONSE_MAGIC, sequence, count],
                        device,
                        dst=1,
                    )
                    print(
                        "[stage6-hccl] draft bridge commit_batch "
                        f"v3 seq={sequence} updates={count}",
                        flush=True,
                    )
                    continue

                if magic == REBASE_REQUEST_MAGIC:
                    count = field2
                    if count <= 0:
                        raise RuntimeError(
                            f"invalid HCCL v3 rebase count: {count}"
                        )

                    requests = []
                    for _ in range(count):
                        slot, prefix_len, _reserved = base.recv_i32(
                            dist, 3, device, src=1
                        )
                        if slot < 0 or prefix_len <= 0:
                            raise RuntimeError(
                                "invalid HCCL v3 rebase row: "
                                f"slot={slot} prefix_len={prefix_len}"
                            )

                        prefix = base.recv_i32(
                            dist, prefix_len, device, src=1
                        )
                        requests.append(
                            {
                                "request_id": f"hccl-slot-{slot}",
                                "prefix_token_ids": prefix,
                            }
                        )

                    base.send_message(
                        worker_conn,
                        {
                            "cmd": "rebase_batch",
                            "requests": requests,
                        },
                    )
                    response = base.receive_message(worker_reader)

                    if response is None or response.get("status") != "result":
                        base.send_i32(
                            dist,
                            [base.ERROR_MAGIC, sequence, 0],
                            device,
                            dst=1,
                        )
                        continue

                    base.send_i32(
                        dist,
                        [REBASE_RESPONSE_MAGIC, sequence, count],
                        device,
                        dst=1,
                    )
                    print(
                        "[stage6-hccl] draft bridge rebase_batch "
                        f"v3 seq={sequence} rows={count}",
                        flush=True,
                    )
                    continue

                if magic != BATCH_REQUEST_MAGIC:
                    raise RuntimeError(
                        f"invalid HCCL v3 request header: {header}"
                    )
                batch_size = field2
                if batch_size <= 0:
                    raise RuntimeError(
                        f"invalid HCCL v3 batch size: {batch_size}"
                    )
                requests: list[dict[str, Any]] = []
                for index in range(batch_size):
                    slot, prefix_len, gamma = base.recv_i32(
                        dist, 3, device, src=1
                    )
                    if slot < 0 or prefix_len <= 0 or gamma <= 0:
                        raise RuntimeError(
                            "invalid HCCL v3 batch row header: "
                            f"{slot, prefix_len, gamma}"
                        )
                    prefix = base.recv_i32(
                        dist, prefix_len, device, src=1
                    )
                    requests.append(
                        {
                            "request_id": f"hccl-slot-{slot}",
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
                    print(
                        "[stage6-hccl] draft batch worker error "
                        f"seq={sequence}: {response.get('error')}",
                        flush=True,
                    )
                    continue

                raw_results = response.get("results")
                if not isinstance(raw_results, list):
                    raise RuntimeError(
                        "Draft batch response must contain results"
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
                            "Draft batch response omitted "
                            f"{request['request_id']!r}"
                        )
                    draft_ids = [
                        int(x)
                        for x in item.get("draft_token_ids", [])[
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
                    "[stage6-hccl] draft bridge batch v3 "
                    f"seq={sequence} batch_size={batch_size} "
                    f"slots={[request['request_id'] for request in requests]}",
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
                raise RuntimeError(
                    f"target bridge must use rank 1, got {rank}"
                )
            server = base.bind_server(Path(args.socket))
            print(
                "[stage6-hccl] ready role=target rank=1 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"socket={args.socket} "
                "batch_protocol=v3 stable_slots=1",
                flush=True,
            )
            run_target_bridge(server, dist, device, args.max_model_len)
        else:
            if rank != 0:
                raise RuntimeError(
                    f"draft bridge must use rank 0, got {rank}"
                )
            print(
                "[stage6-hccl] ready role=draft rank=0 "
                f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
                f"worker_socket={args.draft_socket} "
                "batch_protocol=v3 stable_slots=1",
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
