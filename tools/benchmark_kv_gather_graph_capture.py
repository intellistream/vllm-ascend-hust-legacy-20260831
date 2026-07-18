#!/usr/bin/env python3
"""Prototype ACLGraph capture/replay of mapped-host KV gather.

This benchmark deliberately stays outside the CPU connector.  It keeps the
three addresses that matter to ACLGraph stable for the process lifetime:

* one registered CPU host arena;
* fixed-shape NPU ``src_block_ids`` and ``dst_block_ids`` buffers;
* one fixed-shape NPU KV destination arena.

It then compares a decode-like matmul surrogate in three production-shaped
forms:

* ``graph_only``: resident KV followed by a decode graph replay;
* ``mapped_then_graph``: an eager mapped gather followed by that graph replay;
* ``graph_capture_gather_decode``: gather and surrogate captured in one graph.

The surrogate is not a model-performance claim.  It only gives the gather a
real graph consumer and makes graph-boundary/launch overhead measurable.  The
benchmark also changes the contents of the fixed source-ID buffer between
replays to prove that replay reads current IDs rather than capture-time values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def build_source_ids(
    *,
    num_cpu_blocks: int,
    selected_blocks: int,
    seed: int,
    replay_variant: int = 0,
) -> list[int]:
    """Return a deterministic fragmented mapping with a stable shape."""
    if selected_blocks <= 0:
        raise ValueError("selected_blocks must be positive")
    if selected_blocks > num_cpu_blocks:
        raise ValueError("selected_blocks cannot exceed num_cpu_blocks")
    population = list(range(num_cpu_blocks))
    random.Random(seed + replay_variant * 104729).shuffle(population)
    return population[:selected_blocks]


def validate_shape_config(
    *,
    block_bytes: int,
    element_size: int,
    selected_blocks: int,
    num_cpu_blocks: int,
    num_npu_blocks: int,
    decode_rows: int,
    surrogate_hidden: int,
    surrogate_depth: int,
) -> int:
    if block_bytes <= 0 or block_bytes % element_size:
        raise ValueError("block_bytes must be positive and divisible by dtype size")
    if selected_blocks <= 0 or selected_blocks > min(num_cpu_blocks, num_npu_blocks):
        raise ValueError("selected_blocks must fit both CPU and NPU arenas")
    elements_per_block = block_bytes // element_size
    if decode_rows <= 0 or decode_rows > selected_blocks:
        raise ValueError("decode_rows must be in [1, selected_blocks]")
    if surrogate_hidden <= 0 or surrogate_hidden > elements_per_block:
        raise ValueError("surrogate_hidden must fit one KV block")
    if surrogate_depth <= 0:
        raise ValueError("surrogate_depth must be positive")
    return elements_per_block


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def derive_comparison(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    graph_only = rows["graph_only"]["mean_ms"]
    outside = rows["mapped_then_graph"]["mean_ms"]
    captured = rows["graph_capture_gather_decode"]["mean_ms"]
    return {
        "outside_exposed_over_graph_ms": outside - graph_only,
        "captured_exposed_over_graph_ms": captured - graph_only,
        "capture_savings_ms": outside - captured,
        "capture_savings_percent": (outside - captured) / outside * 100.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-cpu-blocks", type=int, default=2048)
    parser.add_argument("--num-npu-blocks", type=int, default=512)
    parser.add_argument("--selected-blocks", type=int, default=128)
    parser.add_argument("--block-bytes", type=int, default=16384)
    parser.add_argument("--decode-rows", type=int, default=128)
    parser.add_argument("--surrogate-hidden", type=int, default=512)
    parser.add_argument("--surrogate-depth", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--replays-per-sample",
        type=int,
        default=10,
        help="Batch replays so device synchronization overhead is amortized",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--task-queue-enable",
        choices=("0", "1"),
        default="0",
        help=(
            "Set before torch_npu import. The current OpCommand custom handler "
            "is only submitted while capture is active when this is 0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("branch_development_notes/work/kv-gather-graph-capture"),
    )
    parser.add_argument("--op-lib", type=Path, default=None)
    parser.add_argument("--opapi-lib", type=Path, default=None)
    parser.add_argument("--source-git-sha", default=None)
    parser.add_argument("--source-git-branch", default=None)
    return parser.parse_args()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_operator_provenance(opapi_lib: Path | None) -> dict[str, Any]:
    if opapi_lib is None:
        return {}
    opapi_lib = opapi_lib.resolve()
    vendor_root = opapi_lib.parents[2]
    source_header = vendor_root / (
        "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/kv_cache_block_gather/kv_cache_block_gather.h"
    )
    kernel_dir = vendor_root / ("op_impl/ai_core/tbe/kernel/ascend910b/kv_cache_block_gather")
    return {
        "vendor_root": str(vendor_root),
        "opapi_library": str(opapi_lib),
        "opapi_sha256": file_sha256(opapi_lib),
        "kernel_source": str(source_header),
        "kernel_source_sha256": file_sha256(source_header),
        "kernel_objects_sha256": {
            str(path.relative_to(vendor_root)): file_sha256(path) for path in sorted(kernel_dir.glob("*.o"))
        },
    }


def load_runtime(args: argparse.Namespace):
    if args.opapi_lib is not None:
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"] = str(args.opapi_lib)

    import torch
    import torch_npu  # noqa: F401

    if args.op_lib is not None:
        torch.ops.load_library(str(args.op_lib))
    else:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

    # ACLGraph's auto-dispatch capture path needs a Meta implementation for an
    # otherwise opaque custom operator.  The production extension does not yet
    # register one for this experimental in-place/void op, so keep a benchmark-
    # local Library alive for the duration of capture and replay.
    meta_library = torch.library.Library("_C_ascend", "IMPL", "Meta")
    meta_library.impl("kv_cache_block_gather", lambda src_ids, src, dst_ids, out: None)

    return (
        torch,
        torch.ops._C_ascend.kv_cache_block_gather,
        torch.ops._C_ascend.register_kv_cache_block_gather_host_mapping,
        torch.ops._C_ascend.clear_kv_cache_block_gather_host_mappings,
        meta_library,
    )


def capture_graph(torch: Any, operation: Callable[[], Any]) -> tuple[Any, Any]:
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def measure_replays(
    torch: Any,
    operation: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    replays_per_sample: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()

    values = []
    for _ in range(samples):
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(replays_per_sample):
            operation()
        torch.npu.synchronize()
        values.append((time.perf_counter() - started) * 1000.0 / replays_per_sample)
    return summarize_ms(values)


def assert_gathered_blocks(
    torch: Any,
    *,
    source: Any,
    source_ids: Sequence[int],
    destination: Any,
) -> None:
    expected = source.index_select(0, torch.tensor(source_ids, dtype=torch.long))
    actual = destination[: len(source_ids)].cpu()
    if not torch.equal(actual, expected):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"captured gather output mismatch, max_diff={max_diff}")


def write_results(
    output_dir: Path,
    manifest: dict[str, Any],
    rows: dict[str, dict[str, float]],
    comparison: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "measurements": rows, "comparison": comparison}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Mapped-host Gather ACLGraph Capture Prototype",
        "",
        f"- git: `{manifest['git_sha']}`",
        f"- device: `{manifest['device_name']}` (`{manifest['device']}`)",
        f"- dtype: `{manifest['dtype']}`",
        f"- logical gather bytes: `{manifest['logical_gather_bytes']}`",
        f"- fixed host arena address: `{manifest['addresses']['host_arena']}`",
        f"- dynamic fixed-buffer replay validated: `{manifest['dynamic_id_replay_validated']}`",
        "",
        "| mode | mean ms | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, result in rows.items():
        lines.append(f"| {mode} | {result['mean_ms']:.4f} | {result['p50_ms']:.4f} | {result['p95_ms']:.4f} |")
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- outside exposed over graph: `{comparison['outside_exposed_over_graph_ms']:.4f} ms`",
            f"- captured exposed over graph: `{comparison['captured_exposed_over_graph_ms']:.4f} ms`",
            f"- graph capture saving: `{comparison['capture_savings_ms']:.4f} ms` "
            f"(`{comparison['capture_savings_percent']:+.2f}%`)",
            "",
            "The decode matmul stack is a dependency-preserving surrogate, not a model benchmark.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.samples <= 0 or args.replays_per_sample <= 0:
        raise ValueError("warmup must be non-negative; samples/replays must be positive")

    # OpCommand may defer its custom handler to a host task queue.  If that
    # handler runs after the torch.npu.graph context exits, the gather executes
    # once during capture but is absent from every replay.  This prototype
    # forces immediate handler submission before torch_npu is imported.
    os.environ["TASK_QUEUE_ENABLE"] = args.task_queue_enable
    torch, gather, register, clear, meta_library = load_runtime(args)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    elements_per_block = validate_shape_config(
        block_bytes=args.block_bytes,
        element_size=element_size,
        selected_blocks=args.selected_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
        num_npu_blocks=args.num_npu_blocks,
        decode_rows=args.decode_rows,
        surrogate_hidden=args.surrogate_hidden,
        surrogate_depth=args.surrogate_depth,
    )

    torch.npu.set_device(args.device)
    clear()

    # All objects below intentionally remain strongly referenced until after
    # both graphs are destroyed.  In particular, never register a temporary
    # host slice: graph replay embeds the mapped device address.
    host_arena = torch.empty((args.num_cpu_blocks, elements_per_block), dtype=dtype)
    block_values = torch.arange(args.num_cpu_blocks, dtype=torch.float32).remainder_(97).div_(97)
    host_arena.copy_(block_values[:, None])
    host_arena[:, 0] = torch.arange(args.num_cpu_blocks, dtype=dtype).remainder_(1024)
    registration = dict(register(host_arena))

    source_ids_initial = build_source_ids(
        num_cpu_blocks=args.num_cpu_blocks,
        selected_blocks=args.selected_blocks,
        seed=args.seed,
    )
    source_ids_alternate = build_source_ids(
        num_cpu_blocks=args.num_cpu_blocks,
        selected_blocks=args.selected_blocks,
        seed=args.seed,
        replay_variant=1,
    )
    src_ids = torch.tensor(source_ids_initial, dtype=torch.int32, device=args.device)
    dst_ids = torch.arange(args.selected_blocks, dtype=torch.int32, device=args.device)
    decode_kv = torch.zeros((args.num_npu_blocks, elements_per_block), dtype=dtype, device=args.device)
    captured_kv = torch.zeros((args.num_npu_blocks, elements_per_block), dtype=dtype, device=args.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    weights = []
    for _ in range(args.surrogate_depth):
        weight = torch.randn(
            (args.surrogate_hidden, args.surrogate_hidden),
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.01)
        weights.append(weight.to(dtype=dtype, device=args.device))

    def decode_surrogate(kv: Any) -> Any:
        value = kv[: args.decode_rows, : args.surrogate_hidden]
        for weight in weights:
            value = torch.relu(torch.matmul(value, weight))
        return value

    decode_graph, decode_output = capture_graph(torch, lambda: decode_surrogate(decode_kv))

    def gather_and_decode() -> Any:
        gather(src_ids, host_arena, dst_ids, captured_kv)
        return decode_surrogate(captured_kv)

    captured_graph, captured_output = capture_graph(torch, gather_and_decode)

    # Correctness of both the data dependency and mutable contents behind the
    # fixed-shape/fixed-address source-ID input.
    decode_kv.zero_()
    gather(src_ids, host_arena, dst_ids, decode_kv)
    decode_graph.replay()
    torch.npu.synchronize()
    assert_gathered_blocks(torch, source=host_arena, source_ids=source_ids_initial, destination=decode_kv)
    expected_decode = decode_output.cpu().clone()

    captured_kv.zero_()
    # NPUGraph replay may execute on its captured stream.  Make test-only
    # sentinel writes complete first; otherwise the reset itself can race the
    # graph and masquerade as a missing captured gather.
    torch.npu.synchronize()
    captured_graph.replay()
    torch.npu.synchronize()
    assert_gathered_blocks(torch, source=host_arena, source_ids=source_ids_initial, destination=captured_kv)
    torch.testing.assert_close(captured_output.cpu(), expected_decode, rtol=1e-3, atol=1e-3)

    src_ids.copy_(torch.tensor(source_ids_alternate, dtype=torch.int32, device=args.device))
    captured_kv.zero_()
    torch.npu.synchronize()
    captured_graph.replay()
    torch.npu.synchronize()
    assert_gathered_blocks(torch, source=host_arena, source_ids=source_ids_alternate, destination=captured_kv)
    captured_alternate_decode = captured_output.cpu().clone()

    decode_kv.zero_()
    torch.npu.synchronize()
    gather(src_ids, host_arena, dst_ids, decode_kv)
    decode_graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        captured_alternate_decode,
        decode_output.cpu(),
        rtol=1e-3,
        atol=1e-3,
    )
    src_ids.copy_(torch.tensor(source_ids_initial, dtype=torch.int32, device=args.device))
    torch.npu.synchronize()

    # Seed resident KV for graph_only.  Every other operation overwrites the
    # complete selected destination, so no reset is included in timed regions.
    gather(src_ids, host_arena, dst_ids, decode_kv)
    torch.npu.synchronize()

    def mapped_only() -> None:
        gather(src_ids, host_arena, dst_ids, decode_kv)

    def mapped_then_graph() -> None:
        gather(src_ids, host_arena, dst_ids, decode_kv)
        decode_graph.replay()

    operations = {
        "graph_only": decode_graph.replay,
        "mapped_only": mapped_only,
        "mapped_then_graph": mapped_then_graph,
        "graph_capture_gather_decode": captured_graph.replay,
    }
    rows = {
        name: measure_replays(
            torch,
            operation,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        for name, operation in operations.items()
    }
    comparison = derive_comparison(rows)

    manifest = {
        "argv": sys.argv,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "git_branch": args.source_git_branch or git_output("branch", "--show-current"),
        "git_sha": args.source_git_sha or git_output("rev-parse", "HEAD"),
        "device": args.device,
        "device_name": torch.npu.get_device_name(torch.device(args.device)),
        "dtype": args.dtype,
        "num_cpu_blocks": args.num_cpu_blocks,
        "num_npu_blocks": args.num_npu_blocks,
        "selected_blocks": args.selected_blocks,
        "block_bytes": args.block_bytes,
        "logical_gather_bytes": args.selected_blocks * args.block_bytes,
        "decode_rows": args.decode_rows,
        "surrogate_hidden": args.surrogate_hidden,
        "surrogate_depth": args.surrogate_depth,
        "warmup": args.warmup,
        "samples": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "task_queue_enable": args.task_queue_enable,
        "registration": registration,
        "operator_provenance": collect_operator_provenance(args.opapi_lib),
        "dynamic_id_replay_validated": True,
        "addresses": {
            "host_arena": host_arena.data_ptr(),
            "src_ids": src_ids.data_ptr(),
            "dst_ids": dst_ids.data_ptr(),
            "decode_kv": decode_kv.data_ptr(),
            "captured_kv": captured_kv.data_ptr(),
        },
    }
    write_results(args.output_dir, manifest, rows, comparison)
    print(json.dumps({"measurements": rows, "comparison": comparison}, indent=2, sort_keys=True))
    print(f"wrote {args.output_dir / 'summary.md'}")

    # Explicit ordering matters: graph replay may retain the mapped address.
    operations.clear()
    del mapped_only, mapped_then_graph, operations
    captured_graph = None
    decode_graph = None
    del meta_library
    torch.npu.synchronize()
    clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
