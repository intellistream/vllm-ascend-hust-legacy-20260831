#!/usr/bin/env python3
"""Benchmark mixed mapped-host/device KV consumption and first-use promotion.

This is a decision prototype, not a production attention benchmark.  Its
AscendC consumer performs the attention-like inner dataflow ``score = K dot Q``
for each selected block.  A block can live in mapped host memory or device GM.
The benchmark compares three steady sequence policies:

* ``device``: all selected blocks are already device resident;
* ``permanent_hybrid``: host blocks are read through the mapping every token;
* ``promote_first_use``: token 0 consumes mixed sources and writes a compact
  device cache; subsequent tokens consume that promoted cache.

The token sweep exposes the promotion amortization crossover.  Registration,
allocation, and correctness checks are outside timed regions.
"""

# The three timed closures are invoked before their surrounding parameter loop
# advances, so intentionally capture that iteration's tensors and token count.
# ruff: noqa: B023

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Layout:
    source_kinds: tuple[int, ...]
    source_block_ids: tuple[int, ...]
    host_blocks: int
    device_blocks: int


def build_layout(selected_blocks: int, host_fraction: float) -> Layout:
    if selected_blocks <= 0:
        raise ValueError("selected_blocks must be positive")
    if not 0.0 <= host_fraction <= 1.0:
        raise ValueError("host_fraction must be in [0, 1]")
    host_blocks = round(selected_blocks * host_fraction)
    device_blocks = selected_blocks - host_blocks
    # Interleave locations rather than placing one contiguous host suffix.  The
    # source IDs are dense within their respective host/device arenas.
    source_kinds: list[int] = []
    source_ids: list[int] = []
    next_host = next_device = 0
    for pair in range(selected_blocks):
        should_host = (pair * host_blocks) // selected_blocks >= next_host
        if should_host and next_host < host_blocks:
            source_kinds.append(1)
            source_ids.append(next_host)
            next_host += 1
        else:
            source_kinds.append(0)
            source_ids.append(next_device)
            next_device += 1
    return Layout(tuple(source_kinds), tuple(source_ids), host_blocks, device_blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--selected-blocks", type=int, default=128)
    parser.add_argument("--block-elems", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--host-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--op-lib", type=Path)
    parser.add_argument("--opapi-lib", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("branch_development_notes/work/hybrid-kv-promotion"))
    return parser.parse_args()


def load_runtime(args: argparse.Namespace):
    import os
    if args.opapi_lib:
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"] = str(args.opapi_lib)
    import torch
    import torch_npu
    if args.op_lib:
        torch.ops.load_library(str(args.op_lib))
    else:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
    return torch, torch_npu, torch.ops._C_ascend.kv_cache_hybrid_attention_proto


def elapsed_ms(torch: Any, operation, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize_crossovers(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, float], dict[int, dict[str, float]]] = {}
    for row in results:
        shape = (row["block_elems"], row["selected_blocks"], row["host_fraction"])
        groups.setdefault(shape, {}).setdefault(row["tokens"], {})[row["policy"]] = row["mean_ms"]

    summaries: list[dict[str, Any]] = []
    for (block_elems, selected_blocks, host_fraction), token_rows in groups.items():
        crossover = None
        comparisons = []
        for tokens, policies in sorted(token_rows.items()):
            device = policies["device"]
            permanent = policies["permanent_hybrid"]
            promotion = policies["promote_first_use"]
            if crossover is None and promotion < permanent:
                crossover = tokens
            comparisons.append({
                "tokens": tokens,
                "permanent_penalty_vs_device_percent":
                    (permanent / device - 1.0) * 100.0,
                "promotion_speedup_vs_permanent_percent":
                    (1.0 - promotion / permanent) * 100.0,
            })
        summaries.append({
            "block_elems": block_elems,
            "selected_blocks": selected_blocks,
            "host_fraction": host_fraction,
            "promotion_crossover_tokens": crossover,
            "comparisons": comparisons,
        })
    return summaries


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch, torch_npu, op = load_runtime(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    results: list[dict[str, Any]] = []

    for block_elems in args.block_elems:
        if block_elems <= 0 or block_elems % 8:
            raise ValueError("block-elems must be positive multiples of 8")
        for host_fraction in args.host_fractions:
            layout = build_layout(args.selected_blocks, host_fraction)
            # Keep at least one physical page in both arenas: empty tensors
            # cannot be host-registered and complicate a boundary-case benchmark.
            host_pages = torch.randn(max(1, layout.host_blocks), block_elems,
                                     dtype=torch.float32)
            device_pages_cpu = torch.randn(max(1, layout.device_blocks), block_elems,
                                            dtype=torch.float32)
            # The prototype kernel performs explicit flat ND addressing.  NPU
            # factory/copy heuristics may otherwise choose an internal format.
            device_pages = torch_npu.npu_format_cast(device_pages_cpu.to(device), 2)
            kinds = torch.tensor(layout.source_kinds, dtype=torch.int32, device=device)
            ids = torch.tensor(layout.source_block_ids, dtype=torch.int32, device=device)
            promoted_ids = torch.arange(args.selected_blocks, dtype=torch.int32,
                                        device=device)
            device_kinds = torch.zeros(args.selected_blocks, dtype=torch.int32,
                                       device=device)
            promote_on = torch.ones(1, dtype=torch.int32, device=device)
            promote_off = torch.zeros(1, dtype=torch.int32, device=device)
            max_tokens = max(args.tokens)
            queries_cpu = [torch.randn(block_elems, dtype=torch.float32)
                           for _ in range(max_tokens)]
            queries = [torch_npu.npu_format_cast(query.to(device), 2)
                       for query in queries_cpu]

            resolved_cpu = torch.stack([
                host_pages[source_id] if source_kind else device_pages_cpu[source_id]
                for source_kind, source_id in zip(layout.source_kinds,
                                                  layout.source_block_ids)
            ])
            all_device = torch_npu.npu_format_cast(resolved_cpu.to(device), 2)
            scores = torch_npu.npu_format_cast(
                torch.empty(args.selected_blocks, 8, dtype=torch.float32,
                            device=device), 2)
            promoted = torch_npu.npu_format_cast(
                torch.empty(args.selected_blocks, block_elems,
                            dtype=torch.float32, device=device), 2)
            scratch = torch_npu.npu_format_cast(
                torch.empty(args.selected_blocks, block_elems,
                            dtype=torch.float32, device=device), 2)

            # Validate the mixed path and the promotion payload before timing.
            op(kinds, ids, host_pages, device_pages, queries[0], promote_on,
               scores, promoted)
            torch.npu.synchronize()
            reference = resolved_cpu @ queries_cpu[0]
            actual_scores = scores.cpu()[:, 0]
            try:
                torch.testing.assert_close(actual_scores, reference, rtol=2e-4,
                                           atol=2e-3)
            except AssertionError:
                print(f"source_kinds={layout.source_kinds}")
                print(f"actual_scores={actual_scores.tolist()}")
                print(f"reference_scores={reference.tolist()}")
                raise
            torch.testing.assert_close(promoted.cpu(), resolved_cpu, rtol=0, atol=0)

            for token_count in args.tokens:
                if token_count <= 0:
                    raise ValueError("tokens must be positive")

                def device_sequence() -> None:
                    for token in range(token_count):
                        op(device_kinds, promoted_ids, host_pages, all_device,
                           queries[token], promote_off, scores, scratch)

                def permanent_hybrid_sequence() -> None:
                    for token in range(token_count):
                        op(kinds, ids, host_pages, device_pages, queries[token],
                           promote_off, scores, scratch)

                def promote_first_use_sequence() -> None:
                    op(kinds, ids, host_pages, device_pages, queries[0],
                       promote_on, scores, promoted)
                    for token in range(1, token_count):
                        op(device_kinds, promoted_ids, host_pages, promoted,
                           queries[token], promote_off, scores, scratch)

                for policy, operation in (
                    ("device", device_sequence),
                    ("permanent_hybrid", permanent_hybrid_sequence),
                    ("promote_first_use", promote_first_use_sequence),
                ):
                    samples = elapsed_ms(torch, operation, args.warmup, args.iters)
                    results.append({
                        "policy": policy,
                        "block_elems": block_elems,
                        "block_bytes": block_elems * 4,
                        "selected_blocks": args.selected_blocks,
                        "host_fraction": host_fraction,
                        "host_blocks": layout.host_blocks,
                        "device_blocks": layout.device_blocks,
                        "tokens": token_count,
                        "mean_ms": statistics.fmean(samples),
                        "p50_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "per_token_ms": statistics.fmean(samples) / token_count,
                    })
    return results


def main() -> None:
    args = parse_args()
    results = run(args)
    crossovers = summarize_crossovers(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                "results": results, "crossovers": crossovers}
    (args.output_dir / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("promotion amortization summary:")
    print(json.dumps(crossovers, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
