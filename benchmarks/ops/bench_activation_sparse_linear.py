import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from vllm_ascend.ops.sparse_linear import (
    _custom_op_enabled,
    activation_sparse_linear,
    activation_sparse_linear_direct,
    activation_sparse_linear_direct_t,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=3584)
    parser.add_argument("--output-dim", type=int, default=18944)
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument(
        "--threshold-mode",
        choices=["scalar", "row_topk"],
        default="scalar",
        help=(
            "scalar matches precomputed TEAL thresholds; row_topk matches "
            "La RoSA's per-row top-k threshold generation."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--inclusive", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--max-sparse-err",
        type=float,
        default=None,
        help="Fail if packed sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-err",
        type=float,
        default=None,
        help="Fail if direct sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--max-direct-t-err",
        type=float,
        default=None,
        help="Fail if direct_t sparse max abs error exceeds this value.",
    )
    parser.add_argument(
        "--min-packed-total-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / packed_total_ms is below this value.",
    )
    parser.add_argument(
        "--min-packed-total-with-threshold-speedup",
        type=float,
        default=None,
        help=(
            "Fail if dense_ms / packed_total_with_threshold_ms is below this "
            "value. Online threshold cost is included for row_topk mode."
        ),
    )
    parser.add_argument(
        "--min-packed-compute-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / packed_compute_ms is below this value.",
    )
    parser.add_argument(
        "--min-direct-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / direct_sparse_ms is below this value.",
    )
    parser.add_argument(
        "--min-direct-t-speedup",
        type=float,
        default=None,
        help="Fail if dense_ms / direct_t_sparse_ms is below this value.",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip the direct sparse kernel and benchmark only the packed path.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def synchronize() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    return (time.perf_counter() - start) / iters


def topk_keep(input_dim: int, sparsity: float) -> int:
    keep = int(input_dim * (1.0 - sparsity))
    if keep < 1:
        raise ValueError(
            "row_topk threshold mode requires at least one kept activation; "
            f"got input_dim={input_dim}, sparsity={sparsity}."
        )
    return keep


def build_threshold(
    x: torch.Tensor,
    sparsity: float,
    threshold_mode: str,
) -> torch.Tensor:
    x_abs = x.abs().to(dtype=torch.float32)
    if threshold_mode == "scalar":
        return torch.quantile(x_abs.flatten(), sparsity).reshape(())
    if threshold_mode == "row_topk":
        keep = topk_keep(x.shape[-1], sparsity)
        topk_values, _ = torch.topk(x_abs, keep, dim=-1)
        return topk_values[..., -1].contiguous()
    raise ValueError(f"Unsupported threshold mode: {threshold_mode}")


def threshold_for_mask(threshold: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if threshold.numel() == 1:
        return threshold.reshape(())
    return threshold.reshape(x.shape[0], 1)


def main() -> None:
    args = parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is required for this benchmark.")
    if not _custom_op_enabled():
        raise RuntimeError("Ascend custom ops must be enabled for this benchmark.")
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in [0, 1).")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("npu")
    torch.manual_seed(0)

    x = torch.randn(args.batch_size, args.input_dim, device=device, dtype=dtype)
    weight = torch.randn(args.output_dim, args.input_dim, device=device, dtype=dtype)
    weight_t = weight.t().contiguous()
    topk_keep_count = (
        topk_keep(args.input_dim, args.sparsity)
        if args.threshold_mode == "row_topk"
        else None
    )
    threshold = build_threshold(x, args.sparsity, args.threshold_mode)
    effective_inclusive = args.inclusive or args.threshold_mode == "row_topk"
    compare = torch.ge if effective_inclusive else torch.gt

    dense_out = torch.nn.functional.linear(x, weight)
    direct_out = None
    if not args.skip_direct:
        direct_out = activation_sparse_linear_direct(
            x,
            weight,
            threshold,
            inclusive=effective_inclusive,
        )
    packed_out = activation_sparse_linear(
        x,
        weight,
        threshold,
        inclusive=effective_inclusive,
        weight_t=weight_t,
    )
    direct_t_out = activation_sparse_linear_direct_t(
        x,
        weight_t,
        threshold,
        inclusive=effective_inclusive,
    )
    threshold_mask = threshold_for_mask(threshold, x)
    masked_x = torch.where(
        compare(x.abs().to(dtype=torch.float32), threshold_mask),
        x,
        torch.zeros_like(x),
    )
    expected_sparse = torch.nn.functional.linear(masked_x, weight)
    synchronize()

    sparse_max_abs_err = (
        packed_out.to(dtype=torch.float32) -
        expected_sparse.to(dtype=torch.float32)
    ).abs().max().item()
    direct_max_abs_err = None
    if direct_out is not None:
        direct_max_abs_err = (
            direct_out.to(dtype=torch.float32) -
            expected_sparse.to(dtype=torch.float32)
        ).abs().max().item()
    direct_t_max_abs_err = (
        direct_t_out.to(dtype=torch.float32) -
        expected_sparse.to(dtype=torch.float32)
    ).abs().max().item()
    dense_sparse_max_abs_delta = (
        dense_out.to(dtype=torch.float32) -
        expected_sparse.to(dtype=torch.float32)
    ).abs().max().item()

    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold.to(dtype=torch.float32, device=x.device).contiguous(),
        effective_inclusive,
    )
    synchronize()

    dense_time = bench(
        lambda: torch.nn.functional.linear(x, weight),
        args.warmup,
        args.iters,
    )
    threshold_time = bench(
        lambda: build_threshold(x, args.sparsity, args.threshold_mode),
        args.warmup,
        args.iters,
    )
    direct_sparse_time = None
    if not args.skip_direct:
        direct_sparse_time = bench(
            lambda: activation_sparse_linear_direct(
                x,
                weight,
                threshold,
                inclusive=effective_inclusive,
            ),
            args.warmup,
            args.iters,
        )
    direct_t_sparse_time = bench(
        lambda: activation_sparse_linear_direct_t(
            x,
            weight_t,
            threshold,
            inclusive=effective_inclusive,
        ),
        args.warmup,
        args.iters,
    )
    pack_time = bench(
        lambda: torch.ops._C_ascend.activation_sparse_pack(
            x.contiguous(),
            threshold.to(dtype=torch.float32, device=x.device).contiguous(),
            effective_inclusive,
        ),
        args.warmup,
        args.iters,
    )
    packed_compute_time = bench(
        lambda: torch.ops._C_ascend.activation_sparse_linear_packed_t(
            values,
            indices,
            counts,
            weight_t,
        ),
        args.warmup,
        args.iters,
    )
    packed_total_time = bench(
        lambda: activation_sparse_linear(
            x,
            weight,
            threshold,
            inclusive=effective_inclusive,
            weight_t=weight_t,
        ),
        args.warmup,
        args.iters,
    )
    threshold_online = args.threshold_mode == "row_topk"
    online_threshold_time = threshold_time if threshold_online else 0.0
    packed_total_with_threshold_time = packed_total_time + online_threshold_time
    density = (
        compare(x.abs().to(dtype=torch.float32), threshold_mask)
        .float()
        .mean()
        .item()
    )
    direct_speedup = (
        None if direct_sparse_time is None else dense_time / direct_sparse_time
    )
    direct_t_speedup = dense_time / direct_t_sparse_time
    packed_total_speedup = dense_time / packed_total_time
    packed_compute_speedup = dense_time / packed_compute_time
    packed_total_with_threshold_speedup = (
        dense_time / packed_total_with_threshold_time
    )
    failures = []
    if not math.isfinite(sparse_max_abs_err):
        failures.append(
            f"packed sparse max abs error is not finite: {sparse_max_abs_err}"
        )
    if (
        args.max_sparse_err is not None
        and (
            not math.isfinite(sparse_max_abs_err)
            or sparse_max_abs_err > args.max_sparse_err
        )
    ):
        failures.append(
            "packed sparse max abs error "
            f"{sparse_max_abs_err:.6g} > {args.max_sparse_err:.6g}"
        )
    if direct_max_abs_err is not None and not math.isfinite(direct_max_abs_err):
        failures.append(
            f"direct sparse max abs error is not finite: {direct_max_abs_err}"
        )
    if not math.isfinite(direct_t_max_abs_err):
        failures.append(
            f"direct_t sparse max abs error is not finite: {direct_t_max_abs_err}"
        )
    if (
        args.max_direct_err is not None
        and direct_max_abs_err is not None
        and (
            not math.isfinite(direct_max_abs_err)
            or direct_max_abs_err > args.max_direct_err
        )
    ):
        failures.append(
            "direct sparse max abs error "
            f"{direct_max_abs_err:.6g} > {args.max_direct_err:.6g}"
        )
    if (
        args.max_direct_t_err is not None
        and (
            not math.isfinite(direct_t_max_abs_err)
            or direct_t_max_abs_err > args.max_direct_t_err
        )
    ):
        failures.append(
            "direct_t sparse max abs error "
            f"{direct_t_max_abs_err:.6g} > {args.max_direct_t_err:.6g}"
        )
    if args.max_direct_err is not None and direct_max_abs_err is None:
        failures.append("--max-direct-err cannot be used with --skip-direct")
    if (
        args.min_packed_total_speedup is not None
        and packed_total_speedup < args.min_packed_total_speedup
    ):
        failures.append(
            "packed total speedup "
            f"{packed_total_speedup:.6g} < {args.min_packed_total_speedup:.6g}"
        )
    if (
        args.min_packed_total_with_threshold_speedup is not None
        and packed_total_with_threshold_speedup
        < args.min_packed_total_with_threshold_speedup
    ):
        failures.append(
            "packed total with threshold speedup "
            f"{packed_total_with_threshold_speedup:.6g} < "
            f"{args.min_packed_total_with_threshold_speedup:.6g}"
        )
    if (
        args.min_packed_compute_speedup is not None
        and packed_compute_speedup < args.min_packed_compute_speedup
    ):
        failures.append(
            "packed compute speedup "
            f"{packed_compute_speedup:.6g} < "
            f"{args.min_packed_compute_speedup:.6g}"
        )
    if (
        args.min_direct_speedup is not None
        and direct_speedup is not None
        and direct_speedup < args.min_direct_speedup
    ):
        failures.append(
            "direct speedup "
            f"{direct_speedup:.6g} < {args.min_direct_speedup:.6g}"
        )
    if args.min_direct_speedup is not None and direct_speedup is None:
        failures.append("--min-direct-speedup cannot be used with --skip-direct")
    if (
        args.min_direct_t_speedup is not None
        and direct_t_speedup < args.min_direct_t_speedup
    ):
        failures.append(
            "direct_t speedup "
            f"{direct_t_speedup:.6g} < {args.min_direct_t_speedup:.6g}"
        )

    result = {
        "batch_size": args.batch_size,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "dtype": args.dtype,
        "threshold_mode": args.threshold_mode,
        "requested_inclusive": args.inclusive,
        "inclusive": effective_inclusive,
        "topk_keep": topk_keep_count,
        "threshold_online": threshold_online,
        "requested_sparsity": args.sparsity,
        "measured_density": density,
        "measured_sparsity": 1.0 - density,
        "nnz_min": int(counts.min().item()),
        "nnz_max": int(counts.max().item()),
        "nnz_mean": float(counts.to(dtype=torch.float32).mean().item()),
        "sparse_max_abs_err": sparse_max_abs_err,
        "direct_max_abs_err": direct_max_abs_err,
        "direct_t_max_abs_err": direct_t_max_abs_err,
        "dense_sparse_max_abs_delta": dense_sparse_max_abs_delta,
        "weight_t_cached": True,
        "dense_ms": dense_time * 1000.0,
        "threshold_ms": threshold_time * 1000.0,
        "online_threshold_ms": online_threshold_time * 1000.0,
        "direct_sparse_ms": (
            None if direct_sparse_time is None else direct_sparse_time * 1000.0
        ),
        "direct_t_sparse_ms": direct_t_sparse_time * 1000.0,
        "pack_ms": pack_time * 1000.0,
        "packed_compute_ms": packed_compute_time * 1000.0,
        "packed_total_ms": packed_total_time * 1000.0,
        "packed_total_with_threshold_ms": (
            packed_total_with_threshold_time * 1000.0
        ),
        "direct_speedup": direct_speedup,
        "direct_t_speedup": direct_t_speedup,
        "packed_total_speedup": packed_total_speedup,
        "packed_total_with_threshold_speedup": (
            packed_total_with_threshold_speedup
        ),
        "packed_compute_speedup": packed_compute_speedup,
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    if failures:
        print("activation_sparse_linear benchmark failed gates:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
