#!/usr/bin/env python3
"""Measure ACLGraph decode combined with CPU KV restore on Ascend.

This benchmark answers a different question from
``benchmark_kv_gather_vs_span.py``.  The older benchmark compares isolated
transfer latency.  This one compares the production-shaped compositions:

* ``graph_only``: replay a graph-captured synthetic decode-attention workload;
* ``span_only``: restore CPU KV through coalesced non-blocking ``copy_`` calls;
* ``mapped_only``: restore CPU KV through ``kv_cache_block_gather``;
* ``graph_overlap_span``: submit graph replay and span-copy on separate streams;
* ``mapped_then_graph``: submit mapped gather and graph replay serially;
* ``graph_overlap_mapped``: submit them on separate streams as a diagnostic.

The decode graph uses real query/key/value attention math, but it is not a
particular vLLM model.  Its batch, heads, context length, head dimension, and
number of repeated layers are explicit command-line parameters.  This keeps
the experiment independent and reproducible while preserving the important
ACLGraph-vs-transfer scheduling shape.

Host wall time around a device-wide synchronization is the primary metric.
NPU events provide a secondary makespan measurement.  The mapped-host custom
op adapter may defer submission outside a Python stream context on some CANN
versions, so the benchmark reports observed behavior rather than assuming the
requested overlap took place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark_kv_gather_vs_span import (
    Case,
    make_case,
    summarize_ms,
    validate_output,
)

MODE_NAMES = (
    "graph_only",
    "span_only",
    "mapped_only",
    "graph_overlap_span",
    "mapped_then_graph",
    "graph_overlap_mapped",
)


@dataclass(frozen=True)
class DecodeShape:
    batch: int
    heads: int
    context_tokens: int
    head_dim: int
    layers: int


def calculate_overlap_metrics(
    graph_ms: float,
    transfer_ms: float,
    combined_ms: float,
) -> dict[str, float]:
    """Return normalized overlap and serialization diagnostics.

    ``hidden_fraction`` is zero for fully serialized execution and one when
    the shorter operation is completely hidden by the longer one.  Values are
    intentionally not clamped: small values outside [0, 1] expose measurement
    noise, while larger excursions expose interference or timing mistakes.
    """
    if min(graph_ms, transfer_ms) <= 0:
        raise ValueError("component timings must be positive")
    serialized_ms = graph_ms + transfer_ms
    ideal_overlap_ms = max(graph_ms, transfer_ms)
    hidden_ms = serialized_ms - combined_ms
    return {
        "serialized_ms": serialized_ms,
        "ideal_overlap_ms": ideal_overlap_ms,
        "hidden_ms": hidden_ms,
        "hidden_fraction": hidden_ms / min(graph_ms, transfer_ms),
        "over_ideal_ms": combined_ms - ideal_overlap_ms,
        "over_serialized_ms": combined_ms - serialized_ms,
    }


def remove_measurement_floor(value_ms: float, noop_ms: float) -> float:
    """Remove one per-sample event/synchronization floor from a timing."""
    corrected = value_ms - noop_ms
    if corrected <= 0:
        raise ValueError(
            f"timing {value_ms} ms does not exceed no-op floor {noop_ms} ms"
        )
    return corrected


def classify_timeline(metrics: dict[str, float], tolerance_fraction: float) -> str:
    """Classify a combined measurement without claiming profiler precision."""
    if tolerance_fraction < 0:
        raise ValueError("tolerance_fraction must be non-negative")
    shorter_ms = metrics["serialized_ms"] - metrics["ideal_overlap_ms"]
    tolerance_ms = tolerance_fraction * shorter_ms
    if metrics["over_ideal_ms"] <= tolerance_ms:
        return "overlap_or_shorter"
    if abs(metrics["over_serialized_ms"]) <= tolerance_ms:
        return "serialized"
    if metrics["over_serialized_ms"] > tolerance_ms:
        return "interference_or_overhead"
    return "partial_overlap"


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--num-cpu-blocks", type=int, default=2048)
    parser.add_argument("--num-npu-blocks", type=int, default=2048)
    parser.add_argument("--selected-blocks", type=int, default=256)
    parser.add_argument("--block-bytes", type=int, nargs="+", default=[16384])
    parser.add_argument("--span-lengths", type=int, nargs="+", default=[1, 8, 256])
    parser.add_argument("--decode-batch", type=int, default=16)
    parser.add_argument("--decode-heads", type=int, default=16)
    parser.add_argument("--decode-context-tokens", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--decode-head-dim", type=int, default=128)
    parser.add_argument(
        "--decode-layers",
        type=int,
        default=4,
        help="Repeated attention operations inside one captured graph",
    )
    parser.add_argument("--graph-capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--timeline-tolerance-percent", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("branch_development_notes/work/aclgraph-kv-restore-overlap"),
    )
    parser.add_argument("--op-lib", type=Path, default=None)
    parser.add_argument("--opapi-lib", type=Path, default=None)
    parser.add_argument(
        "--source-git-sha",
        default=None,
        help="Source SHA when running from a copied, non-git benchmark bundle",
    )
    parser.add_argument("--source-git-branch", default=None)
    return parser.parse_args()


def load_runtime(args: argparse.Namespace):
    if args.opapi_lib is not None:
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"] = str(
            args.opapi_lib
        )

    import torch
    import torch_npu  # noqa: F401

    if args.op_lib is not None:
        torch.ops.load_library(str(args.op_lib))
    else:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

    return (
        torch,
        torch.ops._C_ascend.kv_cache_block_gather,
        torch.ops._C_ascend.register_kv_cache_block_gather_host_mapping,
        torch.ops._C_ascend.clear_kv_cache_block_gather_host_mappings,
    )


def validate_args(args: argparse.Namespace, element_size: int) -> None:
    positive = {
        "parts": args.parts,
        "num_cpu_blocks": args.num_cpu_blocks,
        "num_npu_blocks": args.num_npu_blocks,
        "selected_blocks": args.selected_blocks,
        "decode_batch": args.decode_batch,
        "decode_heads": args.decode_heads,
        "decode_head_dim": args.decode_head_dim,
        "decode_layers": args.decode_layers,
        "iters": args.iters,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"values must be positive: {', '.join(invalid)}")
    if args.warmup < 0 or args.graph_capture_warmup < 0:
        raise ValueError("warmup values must be non-negative")
    if args.timeline_tolerance_percent < 0:
        raise ValueError("timeline tolerance must be non-negative")
    for block_bytes in args.block_bytes:
        if block_bytes <= 0 or block_bytes % element_size:
            raise ValueError(
                f"block_bytes={block_bytes} is not divisible by dtype size "
                f"{element_size}"
            )
    for span_len in args.span_lengths:
        if span_len <= 0 or span_len > args.selected_blocks:
            raise ValueError(f"invalid span length {span_len}")
    if any(context <= 0 for context in args.decode_context_tokens):
        raise ValueError("decode context lengths must be positive")


class CapturedDecode:
    """A small graph-captured decode-attention workload with stable buffers."""

    def __init__(self, torch: Any, shape: DecodeShape, dtype: Any, device: str):
        self.torch = torch
        self.shape = shape
        self.scale = 1.0 / math.sqrt(shape.head_dim)
        tensor_shape = (shape.batch, shape.heads, shape.context_tokens, shape.head_dim)
        self.query = torch.randn(
            (shape.batch, shape.heads, 1, shape.head_dim),
            dtype=dtype,
            device=device,
        )
        self.key = torch.randn(tensor_shape, dtype=dtype, device=device)
        self.value = torch.randn(tensor_shape, dtype=dtype, device=device)
        self.graph = torch.npu.NPUGraph()
        self.output = None

    def eager(self):
        output = self.query
        transposed_key = self.key.transpose(-2, -1)
        for _ in range(self.shape.layers):
            scores = self.torch.matmul(output, transposed_key) * self.scale
            probabilities = self.torch.softmax(scores, dim=-1)
            output = self.torch.matmul(probabilities, self.value)
        return output

    def capture(self, stream: Any, warmup: int) -> None:
        with self.torch.npu.stream(stream):
            for _ in range(warmup):
                self.eager()
        stream.synchronize()
        with self.torch.npu.stream(stream), self.torch.npu.graph(self.graph):
            self.output = self.eager()
        stream.synchronize()

    def replay(self) -> None:
        self.graph.replay()


def make_transfer_operations(
    torch: Any,
    gather: Any,
    source: Any,
    out: Any,
    case: Case,
    parts: int,
    device: str,
) -> tuple[Callable[[], None], Callable[[], None]]:
    src_ids = torch.tensor(
        [cpu for cpu, _ in case.block_pairs], dtype=torch.int32, device=device
    )
    dst_ids = torch.tensor(
        [npu for _, npu in case.block_pairs], dtype=torch.int32, device=device
    )

    def span_copy() -> None:
        for cpu_start, npu_start, span_len in case.spans:
            cpu_end = cpu_start + span_len
            npu_end = npu_start + span_len
            for part in range(parts):
                out[part, npu_start:npu_end].copy_(
                    source[part, cpu_start:cpu_end], non_blocking=True
                )

    def mapped_gather() -> None:
        for part in range(parts):
            gather(src_ids, source[part], dst_ids, out[part])

    return span_copy, mapped_gather


def measure_mode(
    torch: Any,
    operation: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure a submission recipe which is responsible for stream placement."""
    for _ in range(warmup):
        operation()
        torch.npu.synchronize()

    wall_samples: list[float] = []
    event_samples: list[float] = []
    current = torch.npu.current_stream()
    for _ in range(iters):
        torch.npu.synchronize()
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record(current)
        wall_start = time.perf_counter()
        operation()
        # Recording after the recipe on the current stream only measures the
        # true makespan when the recipe joins its worker streams first.
        end_event.record(current)
        torch.npu.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        event_samples.append(start_event.elapsed_time(end_event))
    return summarize_ms(wall_samples), summarize_ms(event_samples)


def build_mode_operations(
    torch: Any,
    decode: CapturedDecode,
    span_copy: Callable[[], None],
    mapped_gather: Callable[[], None],
    compute_stream: Any,
    copy_stream: Any,
) -> dict[str, Callable[[], None]]:
    current = torch.npu.current_stream()

    def on_stream(stream: Any, operation: Callable[[], None]) -> None:
        # Anchor worker-stream work after measure_mode's start event on the
        # current stream so the reported event interval is a real makespan.
        stream.wait_stream(current)
        with torch.npu.stream(stream):
            operation()

    def join(stream: Any) -> None:
        current.wait_stream(stream)

    def graph_only() -> None:
        on_stream(compute_stream, decode.replay)
        join(compute_stream)

    def noop() -> None:
        on_stream(compute_stream, lambda: None)
        join(compute_stream)

    def span_only() -> None:
        on_stream(copy_stream, span_copy)
        join(copy_stream)

    def mapped_only() -> None:
        on_stream(compute_stream, mapped_gather)
        join(compute_stream)

    def graph_overlap_span() -> None:
        on_stream(copy_stream, span_copy)
        on_stream(compute_stream, decode.replay)
        join(copy_stream)
        join(compute_stream)

    def mapped_then_graph() -> None:
        # This is the production-shaped mapped path: wait_for_layer_load()
        # synchronizes the load stream before attention can consume that KV.
        # Keep the explicit barrier here.  Merely placing both submissions in
        # one Python stream context is insufficient on stacks where the custom
        # op handler defers its actual submission.
        on_stream(copy_stream, mapped_gather)
        copy_stream.synchronize()
        on_stream(compute_stream, decode.replay)
        join(compute_stream)

    def graph_overlap_mapped() -> None:
        on_stream(copy_stream, mapped_gather)
        on_stream(compute_stream, decode.replay)
        join(copy_stream)
        join(compute_stream)

    return {
        "noop": noop,
        "graph_only": graph_only,
        "span_only": span_only,
        "mapped_only": mapped_only,
        "graph_overlap_span": graph_overlap_span,
        "mapped_then_graph": mapped_then_graph,
        "graph_overlap_mapped": graph_overlap_mapped,
    }


def make_row(
    case: Case,
    decode_shape: DecodeShape,
    measurements: dict[str, tuple[dict[str, float], dict[str, float]]],
    tolerance_fraction: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block_bytes": case.block_bytes,
        "selected_blocks": case.selected_blocks,
        "requested_span_len": case.requested_span_len,
        "span_count": case.span_count,
        "mean_span_len": case.mean_span_len,
        **{f"decode_{key}": value for key, value in asdict(decode_shape).items()},
    }
    noop_wall, noop_event = measurements["noop"]
    row["measurement_noop_wall_mean_ms"] = noop_wall["mean_ms"]
    row["measurement_noop_event_mean_ms"] = noop_event["mean_ms"]
    for mode in MODE_NAMES:
        wall, event = measurements[mode]
        row[f"{mode}_wall_mean_ms"] = wall["mean_ms"]
        row[f"{mode}_wall_p50_ms"] = wall["p50_ms"]
        row[f"{mode}_wall_p95_ms"] = wall["p95_ms"]
        row[f"{mode}_event_mean_ms"] = event["mean_ms"]

    for transfer_name, combined_name in (
        ("span", "graph_overlap_span"),
        ("mapped", "mapped_then_graph"),
        ("mapped", "graph_overlap_mapped"),
    ):
        wall_metrics = calculate_overlap_metrics(
            remove_measurement_floor(
                row["graph_only_wall_mean_ms"],
                row["measurement_noop_wall_mean_ms"],
            ),
            remove_measurement_floor(
                row[f"{transfer_name}_only_wall_mean_ms"],
                row["measurement_noop_wall_mean_ms"],
            ),
            remove_measurement_floor(
                row[f"{combined_name}_wall_mean_ms"],
                row["measurement_noop_wall_mean_ms"],
            ),
        )
        wall_prefix = f"{combined_name}_wall_timeline"
        row.update(
            {f"{wall_prefix}_{key}": value for key, value in wall_metrics.items()}
        )
        row[f"{wall_prefix}_classification"] = classify_timeline(
            wall_metrics, tolerance_fraction
        )
        event_metrics = calculate_overlap_metrics(
            remove_measurement_floor(
                row["graph_only_event_mean_ms"],
                row["measurement_noop_event_mean_ms"],
            ),
            remove_measurement_floor(
                row[f"{transfer_name}_only_event_mean_ms"],
                row["measurement_noop_event_mean_ms"],
            ),
            remove_measurement_floor(
                row[f"{combined_name}_event_mean_ms"],
                row["measurement_noop_event_mean_ms"],
            ),
        )
        event_prefix = f"{combined_name}_event_timeline"
        row.update(
            {f"{event_prefix}_{key}": value for key, value in event_metrics.items()}
        )
        row[f"{event_prefix}_classification"] = classify_timeline(
            event_metrics, tolerance_fraction
        )

    span_pipeline = row["graph_overlap_span_wall_mean_ms"]
    mapped_pipeline = row["mapped_then_graph_wall_mean_ms"]
    row["mapped_pipeline_gain_percent"] = (
        (span_pipeline - mapped_pipeline) / span_pipeline * 100.0
    )
    row["mapped_pipeline_faster"] = mapped_pipeline < span_pipeline
    row["validated"] = True
    return row


def write_results(
    output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "results.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        with (output_dir / "results.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# ACLGraph Decode + KV Restore Overlap",
        "",
        f"- git_sha: `{manifest['git_sha']}`",
        f"- device: `{manifest['device']}`",
        "",
        "| context | span len | spans | graph ms | span ms | mapped ms | "
        "graph || span ms | mapped → graph ms | graph || mapped ms | "
        "mapped pipeline gain | span timeline | mapped parallel timeline |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | :--- | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {decode_context_tokens} | {requested_span_len} | {span_count} | "
            "{graph_only_wall_mean_ms:.3f} | {span_only_wall_mean_ms:.3f} | "
            "{mapped_only_wall_mean_ms:.3f} | "
            "{graph_overlap_span_wall_mean_ms:.3f} | "
            "{mapped_then_graph_wall_mean_ms:.3f} | "
            "{graph_overlap_mapped_wall_mean_ms:.3f} | "
            "{mapped_pipeline_gain_percent:+.1f}% | "
            "{graph_overlap_span_event_timeline_classification} | "
            "{graph_overlap_mapped_event_timeline_classification} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    torch, gather, register, clear = load_runtime(args)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    validate_args(args, element_size)

    torch.npu.set_device(args.device)
    current = torch.npu.current_stream(device=args.device)
    compute_stream = torch.npu.Stream(device=args.device)
    copy_stream = torch.npu.Stream(device=args.device)
    clear()

    manifest = {
        "argv": sys.argv,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cann": getattr(torch.version, "cann", ""),
        "git_branch": args.source_git_branch
        or git_output("branch", "--show-current"),
        "git_sha": args.source_git_sha or git_output("rev-parse", "HEAD"),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": args.device,
        "dtype": args.dtype,
        "parts": args.parts,
        "num_cpu_blocks": args.num_cpu_blocks,
        "num_npu_blocks": args.num_npu_blocks,
        "selected_blocks": args.selected_blocks,
        "block_bytes": args.block_bytes,
        "span_lengths": args.span_lengths,
        "decode_batch": args.decode_batch,
        "decode_heads": args.decode_heads,
        "decode_context_tokens": args.decode_context_tokens,
        "decode_head_dim": args.decode_head_dim,
        "decode_layers": args.decode_layers,
        "graph_capture_warmup": args.graph_capture_warmup,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "timeline_tolerance_percent": args.timeline_tolerance_percent,
        "mapping_registration": [],
    }

    rows: list[dict[str, Any]] = []
    live_sources = []
    try:
        for block_bytes in args.block_bytes:
            elements_per_block = block_bytes // element_size
            source = torch.empty(
                (args.parts, args.num_cpu_blocks, elements_per_block), dtype=dtype
            )
            for part in range(args.parts):
                source[part].fill_(part + 1)
            source[:, :, 0] = torch.arange(
                args.num_cpu_blocks, dtype=dtype
            ).remainder_(1024)
            out = torch.empty(
                (args.parts, args.num_npu_blocks, elements_per_block),
                dtype=dtype,
                device=args.device,
            )
            live_sources.append(source)
            for part in range(args.parts):
                started = time.perf_counter()
                registration = dict(register(source[part]))
                registration.update(
                    wall_ms=(time.perf_counter() - started) * 1000.0,
                    block_bytes=block_bytes,
                    part=part,
                )
                manifest["mapping_registration"].append(registration)

            for context_tokens in args.decode_context_tokens:
                shape = DecodeShape(
                    batch=args.decode_batch,
                    heads=args.decode_heads,
                    context_tokens=context_tokens,
                    head_dim=args.decode_head_dim,
                    layers=args.decode_layers,
                )
                decode = CapturedDecode(torch, shape, dtype, args.device)
                decode.capture(compute_stream, args.graph_capture_warmup)
                current.wait_stream(compute_stream)
                torch.npu.synchronize()

                for index, span_len in enumerate(args.span_lengths):
                    case = make_case(
                        block_bytes=block_bytes,
                        selected_blocks=args.selected_blocks,
                        requested_span_len=span_len,
                        num_cpu_blocks=args.num_cpu_blocks,
                        num_npu_blocks=args.num_npu_blocks,
                        seed=args.seed + index,
                    )
                    span_copy, mapped_gather = make_transfer_operations(
                        torch,
                        gather,
                        source,
                        out,
                        case,
                        args.parts,
                        args.device,
                    )
                    operations = build_mode_operations(
                        torch,
                        decode,
                        span_copy,
                        mapped_gather,
                        compute_stream,
                        copy_stream,
                    )
                    measurements = {}
                    print(
                        f"context={context_tokens} block_bytes={block_bytes} "
                        f"span_len={span_len} spans={case.span_count}",
                        flush=True,
                    )
                    for mode in ("noop", *MODE_NAMES):
                        with torch.npu.stream(current):
                            out.zero_()
                        torch.npu.synchronize()
                        measurements[mode] = measure_mode(
                            torch,
                            operations[mode],
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        if mode not in ("noop", "graph_only"):
                            backend = "span" if "span" in mode else "mapped"
                            validate_output(torch, source, out, case, backend)
                    rows.append(
                        make_row(
                            case,
                            shape,
                            measurements,
                            args.timeline_tolerance_percent / 100.0,
                        )
                    )

        write_results(args.output_dir, manifest, rows)
    finally:
        clear()
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
