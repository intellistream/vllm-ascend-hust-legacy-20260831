#!/usr/bin/env python3
"""Minimal single-node Ascend HCCL probe for physical cards 6 and 7.

Run with ASCEND_RT_VISIBLE_DEVICES=6,7 and torchrun with two local ranks.
The visible devices are therefore logical NPU 0 and 1 inside the two ranks.
"""

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
        value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        print(
            "HCCL_PROBE "
            f"rank={rank} local_rank={local_rank} "
            f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r} "
            f"current_device={torch.npu.current_device()} "
            f"value={value.item()}",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
