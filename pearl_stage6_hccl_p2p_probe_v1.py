#!/usr/bin/env python3
"""Bidirectional single-node Ascend HCCL send/recv probe for cards 6 and 7."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != 2:
        raise RuntimeError(f"expected WORLD_SIZE=2, got {world_size}")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", init_method="env://")

    try:
        if rank == 0:
            outgoing = torch.tensor(
                [101, 102, 103, 104], dtype=torch.int32, device=device
            )
            dist.send(outgoing, dst=1)
            torch.npu.synchronize()

            incoming = torch.empty(4, dtype=torch.int32, device=device)
            dist.recv(incoming, src=1)
            torch.npu.synchronize()
            received = incoming.cpu().tolist()
            if received != [201, 202, 203, 204]:
                raise RuntimeError(f"rank 0 received unexpected payload: {received}")
            print(
                "HCCL_P2P_PROBE rank=0 sent=[101,102,103,104] "
                f"received={received}",
                flush=True,
            )
        else:
            incoming = torch.empty(4, dtype=torch.int32, device=device)
            dist.recv(incoming, src=0)
            torch.npu.synchronize()
            received = incoming.cpu().tolist()
            if received != [101, 102, 103, 104]:
                raise RuntimeError(f"rank 1 received unexpected payload: {received}")

            outgoing = torch.tensor(
                [201, 202, 203, 204], dtype=torch.int32, device=device
            )
            dist.send(outgoing, dst=0)
            torch.npu.synchronize()
            print(
                "HCCL_P2P_PROBE rank=1 received=[101,102,103,104] "
                "sent=[201,202,203,204]",
                flush=True,
            )

        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
