#!/usr/bin/env python3
"""Smoke test for the experimental mapped-host KV cache gather op.

This intentionally tests the custom op directly, without starting a vLLM
engine. It requires a built vllm-ascend extension that registers
torch.ops._C_ascend.kv_cache_block_gather.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-src-blocks", type=int, default=8)
    parser.add_argument("--num-dst-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--op-lib", default=os.getenv("VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_LIB"))
    parser.add_argument("--opapi-lib", default=os.getenv("VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        print(f"FAIL: unable to import torch/torch_npu: {exc}", file=sys.stderr)
        return 2

    if args.opapi_lib:
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"] = args.opapi_lib

    if args.op_lib:
        op_lib = Path(args.op_lib)
        if not op_lib.exists():
            print(f"FAIL: --op-lib does not exist: {op_lib}", file=sys.stderr)
            return 2
        torch.ops.load_library(str(op_lib))
    else:
        try:
            import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401
        except Exception as exc:
            print(f"WARN: unable to import vllm_ascend.vllm_ascend_C: {exc}", file=sys.stderr)

    try:
        gather = torch.ops._C_ascend.kv_cache_block_gather
    except AttributeError:
        print("FAIL: torch.ops._C_ascend.kv_cache_block_gather is not registered", file=sys.stderr)
        return 2

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device(args.device)
    torch.npu.set_device(device)

    shape = (args.num_src_blocks, args.block_size, args.num_heads, args.head_size)
    src_pages = torch.arange(1, 1 + int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape).to(dtype)
    out = torch.zeros(
        (args.num_dst_blocks, args.block_size, args.num_heads, args.head_size),
        dtype=dtype,
        device=device,
    )

    src_block_ids = torch.tensor([3, 1, 6, 0], dtype=torch.int32, device=device)
    dst_block_ids = torch.tensor([0, 2, 4, 7], dtype=torch.int32, device=device)

    gather(src_block_ids, src_pages, dst_block_ids, out)
    torch.npu.synchronize()

    expected = torch.zeros_like(out.cpu())
    for src_block, dst_block in zip(src_block_ids.cpu().tolist(), dst_block_ids.cpu().tolist()):
        expected[dst_block].copy_(src_pages[src_block])

    actual = out.cpu()
    if not torch.equal(actual, expected):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        print(f"FAIL: gather output mismatch, max_diff={max_diff}", file=sys.stderr)
        return 1

    print("PASS: kv_cache_block_gather smoke succeeded")
    print(f"device={args.device} dtype={args.dtype} src_shape={tuple(src_pages.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
