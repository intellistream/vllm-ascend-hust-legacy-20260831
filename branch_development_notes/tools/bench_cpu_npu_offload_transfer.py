#!/usr/bin/env python3
"""Benchmark worker-local CPU<->NPU KV offload transfers.

This tool exercises CpuNpuOffloadingHandler directly with synthetic KV cache
tensors. It is intended as the phase-2 copy baseline before adding the mapped
host-gather backend to the worker-local offload path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


DTYPE_NAMES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CpuNpuOffloadingHandler transfer_async paths."
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("branch_development_notes/work/cpu_npu_transfer_results"),
    )
    parser.add_argument("--num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--num-cpu-blocks", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--block-bytes", type=int, nargs="+",
                        default=[4096, 16384, 65536, 1048576])
    parser.add_argument("--selected-blocks", type=int, nargs="+",
                        default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument("--patterns", nargs="+", default=["random"],
                        choices=["sequential", "reverse", "stride", "random"])
    parser.add_argument("--directions", nargs="+", default=["h2d", "d2h", "bidirectional"],
                        choices=["h2d", "d2h", "bidirectional"])
    parser.add_argument("--h2d-backend", choices=["copy", "mapped"], default="copy")
    parser.add_argument(
        "--custom-opp-path",
        type=Path,
        default=None,
        help=(
            "Custom OPP vendor directory for mapped H2D, usually "
            "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend. "
            "If omitted, the tool auto-detects the editable build output."
        ),
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_NAMES), default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N expanded cases.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write manifest and expanded cases.")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values) * 1000.0 if values else math.nan,
        "p50_ms": percentile(values, 50) * 1000.0,
        "p90_ms": percentile(values, 90) * 1000.0,
        "p95_ms": percentile(values, 95) * 1000.0,
        "p99_ms": percentile(values, 99) * 1000.0,
    }


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name, "")
    parts = [part for part in current.split(":") if part]
    if value not in parts:
        os.environ[name] = ":".join([value, *parts]) if parts else value


def default_custom_opp_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "vllm-ascend"


def configure_mapped_backend_env(args: argparse.Namespace) -> dict[str, str]:
    if args.h2d_backend != "mapped":
        return {}

    custom_opp = args.custom_opp_path or default_custom_opp_path()
    configured: dict[str, str] = {}
    if custom_opp.exists():
        custom_opp_str = str(custom_opp)
        prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_str)
        configured["ASCEND_CUSTOM_OPP_PATH"] = os.environ["ASCEND_CUSTOM_OPP_PATH"]

        opapi_lib = custom_opp / "op_api" / "lib" / "libcust_opapi.so"
        if opapi_lib.exists():
            os.environ.setdefault(
                "VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB", str(opapi_lib)
            )
            configured["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"] = os.environ[
                "VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB"
            ]
    return configured


def build_block_ids(
    *,
    count: int,
    limit: int,
    pattern: str,
    rng: "np.random.Generator",
) -> "np.ndarray":
    import numpy as np

    if count > limit:
        raise ValueError(f"count {count} exceeds block limit {limit}")
    if pattern == "sequential":
        ids = np.arange(count, dtype=np.int64)
    elif pattern == "reverse":
        ids = np.arange(limit - 1, limit - count - 1, -1, dtype=np.int64)
    elif pattern == "stride":
        stride = max(1, limit // count)
        ids = (np.arange(count, dtype=np.int64) * stride) % limit
    elif pattern == "random":
        ids = rng.choice(limit, size=count, replace=False).astype(np.int64)
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return ids


def expanded_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for block_bytes in args.block_bytes:
        for selected_blocks in args.selected_blocks:
            for pattern in args.patterns:
                for direction in args.directions:
                    cases.append(
                        {
                            "case_name": "worker_local_copy",
                            "direction": direction,
                            "block_bytes": block_bytes,
                            "selected_blocks": selected_blocks,
                            "pattern": pattern,
                        }
                    )
    if args.limit is not None:
        cases = cases[:args.limit]
    return cases


def make_manifest(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_sha": run_git(["rev-parse", "HEAD"]),
        "config": {
            "device_id": args.device_id,
            "num_gpu_blocks": args.num_gpu_blocks,
            "num_cpu_blocks": args.num_cpu_blocks,
            "layers": args.layers,
            "dtype": args.dtype,
            "h2d_backend": args.h2d_backend,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "custom_opp_path": str(args.custom_opp_path)
            if args.custom_opp_path
            else "",
        },
        "mapped_backend_env": {
            key: os.environ[key]
            for key in (
                "ASCEND_CUSTOM_OPP_PATH",
                "VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB",
            )
            if key in os.environ
        },
        "cases": cases,
        "dry_run": args.dry_run,
    }


def import_runtime():
    import torch
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
    from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler

    if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
        raise RuntimeError(
            "torch.ops._C_ascend.swap_blocks_batch is not registered. "
            "Run this tool with a vllm_ascend_C extension built from this "
            "branch, for example inside the reproduction Docker after "
            "`python3 -m pip install -e . --no-build-isolation`."
        )
    return torch, CPULoadStoreSpec, GPULoadStoreSpec, CpuNpuOffloadingHandler


def make_handler(args: argparse.Namespace, block_bytes: int):
    if args.h2d_backend == "mapped":
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER"] = "1"
    else:
        os.environ["VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER"] = "0"

    torch, _, _, CpuNpuOffloadingHandler = import_runtime()
    torch.npu.set_device(f"npu:{args.device_id}")

    dtype = getattr(torch, DTYPE_NAMES[args.dtype])
    element_size = torch.empty((), dtype=dtype).element_size()
    if block_bytes % element_size != 0:
        raise ValueError(
            f"block_bytes={block_bytes} is not divisible by {args.dtype} size {element_size}"
        )
    elems_per_block = block_bytes // element_size

    gpu_caches = {}
    for layer_idx in range(args.layers):
        gpu_caches[f"layer_{layer_idx}"] = torch.empty(
            (2, args.num_gpu_blocks, elems_per_block),
            dtype=dtype,
            device=f"npu:{args.device_id}",
        )

    handler = CpuNpuOffloadingHandler(
        gpu_block_size=1,
        cpu_block_size=1,
        num_cpu_blocks=args.num_cpu_blocks,
        gpu_caches=gpu_caches,
        attn_backends={},
    )
    return torch, handler


def clear_mapped_host_mappings(torch: Any, args: argparse.Namespace) -> None:
    if args.h2d_backend != "mapped":
        return
    clear_op = getattr(
        torch.ops._C_ascend, "clear_kv_cache_block_gather_host_mappings", None
    )
    if clear_op is not None:
        cleared = clear_op()
        if cleared:
            print(f"cleared {cleared} mapped host ranges", flush=True)


def wait_for_jobs(handler: Any, job_ids: set[int]) -> list[Any]:
    handler.wait(job_ids)
    deadline = time.monotonic() + 30.0
    results: list[Any] = []
    seen: set[int] = set()
    while len(seen) < len(job_ids):
        for result in handler.get_finished():
            if result.job_id in job_ids and result.job_id not in seen:
                results.append(result)
                seen.add(result.job_id)
        if len(seen) == len(job_ids):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for jobs {sorted(job_ids - seen)}")
        time.sleep(0.001)
    return results


def run_one_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    case_index: int,
) -> list[dict[str, Any]]:
    import numpy as np

    torch, CPULoadStoreSpec, GPULoadStoreSpec, _ = import_runtime()
    clear_mapped_host_mappings(torch, args)
    _, handler = make_handler(args, case["block_bytes"])

    rng = np.random.default_rng(args.seed + case_index)
    cpu_ids = build_block_ids(
        count=case["selected_blocks"],
        limit=args.num_cpu_blocks,
        pattern=case["pattern"],
        rng=rng,
    )
    gpu_ids = build_block_ids(
        count=case["selected_blocks"],
        limit=args.num_gpu_blocks,
        pattern=case["pattern"],
        rng=rng,
    )

    job_id = case_index * 100000

    def issue(direction: str, current_job_id: int) -> set[int]:
        if direction == "h2d":
            spec = (CPULoadStoreSpec(cpu_ids), GPULoadStoreSpec(gpu_ids))
            handler.transfer_async(current_job_id, spec)
            return {current_job_id}
        if direction == "d2h":
            spec = (GPULoadStoreSpec(gpu_ids), CPULoadStoreSpec(cpu_ids))
            handler.transfer_async(current_job_id, spec)
            return {current_job_id}
        if direction == "bidirectional":
            h2d_spec = (CPULoadStoreSpec(cpu_ids), GPULoadStoreSpec(gpu_ids))
            d2h_spec = (GPULoadStoreSpec(gpu_ids), CPULoadStoreSpec(cpu_ids))
            handler.transfer_async(current_job_id, h2d_spec)
            handler.transfer_async(current_job_id + 1, d2h_spec)
            return {current_job_id, current_job_id + 1}
        raise ValueError(f"unsupported direction: {direction}")

    for warmup_idx in range(args.warmup):
        ids = issue(case["direction"], job_id + warmup_idx * 10)
        wait_for_jobs(handler, ids)

    per_direction_times: dict[str, list[float]] = {}
    per_direction_bytes: dict[str, int] = {}
    wall_times: list[float] = []

    for iter_idx in range(args.iters):
        current_job_id = job_id + 1000 + iter_idx * 10
        wall_start = time.perf_counter()
        ids = issue(case["direction"], current_job_id)
        results = wait_for_jobs(handler, ids)
        torch.npu.synchronize()
        wall_times.append(time.perf_counter() - wall_start)
        for result in results:
            direction = "h2d" if result.transfer_type == ("CPU", "NPU") else "d2h"
            if case["direction"] == "bidirectional":
                direction = f"bidirectional_{direction}"
            per_direction_times.setdefault(direction, []).append(result.transfer_time)
            per_direction_bytes[direction] = int(result.transfer_size)

    rows: list[dict[str, Any]] = []
    for direction, values in sorted(per_direction_times.items()):
        num_bytes = per_direction_bytes[direction]
        stats = summarize(values)
        mean_s = stats["mean_ms"] / 1000.0
        gbps = (num_bytes / mean_s / 1e9) if mean_s > 0 else math.nan
        row = {
            **case,
            "measured_direction": direction,
            "device_id": args.device_id,
            "num_gpu_blocks": args.num_gpu_blocks,
            "num_cpu_blocks": args.num_cpu_blocks,
            "layers": args.layers,
            "dtype": args.dtype,
            "h2d_backend": args.h2d_backend,
            "warmup": args.warmup,
            "iters": args.iters,
            "bytes_per_transfer": num_bytes,
            "gbps": gbps,
            "wall_mean_ms": statistics.fmean(wall_times) * 1000.0,
            "status": "pass",
            **stats,
        }
        rows.append(row)

    if case["direction"] == "bidirectional":
        total_bytes = sum(per_direction_bytes.values())
        stats = summarize(wall_times)
        mean_s = stats["mean_ms"] / 1000.0
        rows.append(
            {
                **case,
                "measured_direction": "bidirectional_combined_wall",
                "device_id": args.device_id,
                "num_gpu_blocks": args.num_gpu_blocks,
                "num_cpu_blocks": args.num_cpu_blocks,
                "layers": args.layers,
                "dtype": args.dtype,
                "h2d_backend": args.h2d_backend,
                "warmup": args.warmup,
                "iters": args.iters,
                "bytes_per_transfer": total_bytes,
                "gbps": (total_bytes / mean_s / 1e9) if mean_s > 0 else math.nan,
                "wall_mean_ms": stats["mean_ms"],
                "status": "pass",
                **stats,
            }
        )

    return rows


def write_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_dir / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# CPU/NPU Offload Transfer Results",
        "",
        f"- git_branch: `{manifest.get('git_branch', '')}`",
        f"- git_sha: `{manifest.get('git_sha', '')}`",
        f"- dry_run: `{manifest.get('dry_run')}`",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| direction | h2d_backend | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| {measured_direction} | {h2d_backend} | {block_bytes} | "
                "{selected_blocks} | {pattern} | {mean_ms:.3f} | "
                "{p95_ms:.3f} | {p99_ms:.3f} | {gbps:.2f} | "
                "{status} |".format(**row)
            )
    else:
        lines.append("No measured rows.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    configured_env = configure_mapped_backend_env(args)
    if configured_env:
        print(
            "mapped backend env: "
            + ", ".join(f"{key}={value}" for key, value in configured_env.items()),
            flush=True,
        )
    cases = expanded_cases(args)
    manifest = make_manifest(args, cases)
    if args.dry_run:
        write_outputs(args.output_dir, manifest, [])
        print(f"dry-run: expanded {len(cases)} cases")
        print(f"wrote {args.output_dir / 'manifest.json'}")
        return 0

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case}", flush=True)
        rows.extend(run_one_case(args, case, idx))
    write_outputs(args.output_dir, manifest, rows)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
