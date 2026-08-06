#!/usr/bin/env python3
"""HCCL P2P probe with the same one-visible-device topology as Stage5 workers."""

from __future__ import annotations

import os


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected WORLD_SIZE=2, got {world_size}")

    physical_device = "6" if rank == 0 else "7"
    # Set visibility before importing torch/torch_npu so each rank sees only
    # the physical device that Stage5 assigns to that worker.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_device

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dist.init_process_group(backend="hccl", init_method="env://")
    try:
        if rank == 0:
            outgoing = torch.tensor([601, 602], dtype=torch.int32, device=device)
            dist.send(outgoing, dst=1)
            incoming = torch.empty(2, dtype=torch.int32, device=device)
            dist.recv(incoming, src=1)
            received = incoming.cpu().tolist()
            if received != [701, 702]:
                raise RuntimeError(f"rank 0 received {received}")
            print(
                "HCCL_SPLIT_VISIBLE_PROBE rank=0 "
                f"visible={physical_device} device_count={torch.npu.device_count()} "
                f"received={received}",
                flush=True,
            )
        else:
            incoming = torch.empty(2, dtype=torch.int32, device=device)
            dist.recv(incoming, src=0)
            received = incoming.cpu().tolist()
            if received != [601, 602]:
                raise RuntimeError(f"rank 1 received {received}")
            outgoing = torch.tensor([701, 702], dtype=torch.int32, device=device)
            dist.send(outgoing, dst=0)
            print(
                "HCCL_SPLIT_VISIBLE_PROBE rank=1 "
                f"visible={physical_device} device_count={torch.npu.device_count()} "
                "received=[601, 602]",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
