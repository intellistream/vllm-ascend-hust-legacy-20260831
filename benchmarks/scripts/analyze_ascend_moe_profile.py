#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FUSION_TYPES = {
    "Add",
    "AddRmsNormBias",
    "Cast",
    "Reshape",
    "RmsNorm",
    "Slice",
    "SwiGlu",
    "Transpose",
}
ROUTING_TYPES = {
    "MoeGatingTopK",
    "MoeInitRoutingCustom",
    "MoeTokenPermute",
    "MoeTokenUnpermute",
    "Sort",
    "TopK",
}
PREFILL_TYPES = {
    "FusedInferAttentionScore",
    "FlashAttentionScore",
    "MatMul",
    "MatMulV2",
    "ReshapeAndCache",
    "ReshapeAndCacheNdKernel",
}


def _float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _resolve_output_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name == "ASCEND_PROFILER_OUTPUT":
        return candidate
    nested = candidate / "ASCEND_PROFILER_OUTPUT"
    if nested.exists():
        return nested
    return candidate


def _load_benchmark(path: str | Path | None) -> dict[str, float | str]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    with source.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}

    keys = (
        "label",
        "median_ttft_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "output_throughput",
        "total_token_throughput",
        "request_throughput",
    )
    result: dict[str, float | str] = {}
    for key in keys:
        if key not in payload:
            continue
        if key == "label":
            result[key] = str(payload[key])
        else:
            result[key] = _float(payload[key])
    return result


def _summarize_ops(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    hotspots = []
    total_us = sum(_float(row.get("Total Time(us)")) for row in rows)
    for row in rows:
        time_us = _float(row.get("Total Time(us)"))
        if time_us <= 0:
            continue
        ratio = _float(row.get("Ratio(%)"))
        if ratio == 0 and total_us > 0:
            ratio = time_us / total_us * 100.0
        hotspots.append({
            "name": row.get("OP Type", ""),
            "core_type": row.get("Core Type", ""),
            "count": _int(row.get("Count")),
            "total_us": time_us,
            "avg_us": _float(row.get("Avg Time(us)")),
            "max_us": _float(row.get("Max Time(us)")),
            "ratio_percent": ratio,
        })
    hotspots.sort(key=lambda item: item["total_us"], reverse=True)
    return hotspots


def _summarize_kernels(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "type": "",
            "count": 0,
            "duration_us": 0.0,
            "wait_us": 0.0,
            "max_us": 0.0,
            "cube_sum": 0.0,
            "cube_count": 0,
            "mte_us": 0.0,
        })
    top_by_duration = []
    short_kernel_counts: dict[str, int] = defaultdict(int)
    total_duration_us = 0.0
    total_wait_us = 0.0
    total_mte_us = 0.0

    for row in rows:
        op_type = row.get("Type") or row.get("Name", "")
        duration_us = _float(row.get("Duration(us)"))
        wait_us = _float(row.get("Wait Time(us)"))
        cube = _float(row.get("cube_utilization(%)"))
        mte_us = (
            _float(row.get("aic_mte1_time(us)"))
            + _float(row.get("aic_mte2_time(us)"))
            + _float(row.get("aiv_mte2_time(us)"))
            + _float(row.get("aiv_mte3_time(us)"))
        )

        total_duration_us += duration_us
        total_wait_us += wait_us
        total_mte_us += mte_us
        bucket = by_type[op_type]
        bucket["type"] = op_type
        bucket["count"] += 1
        bucket["duration_us"] += duration_us
        bucket["wait_us"] += wait_us
        bucket["max_us"] = max(bucket["max_us"], duration_us)
        bucket["mte_us"] += mte_us
        if cube > 0:
            bucket["cube_sum"] += cube
            bucket["cube_count"] += 1
        if duration_us and duration_us < 50.0:
            short_kernel_counts[op_type] += 1
        top_by_duration.append({
            "name": row.get("Name", ""),
            "type": op_type,
            "core": row.get("Accelerator Core", ""),
            "duration_us": duration_us,
            "wait_us": wait_us,
            "cube_utilization_percent": cube,
        })

    type_rows = []
    for item in by_type.values():
        count = item["count"]
        cube_count = item.pop("cube_count")
        cube_sum = item.pop("cube_sum")
        item["avg_us"] = item["duration_us"] / count if count else 0.0
        item["avg_cube_utilization_percent"] = cube_sum / cube_count if cube_count else 0.0
        type_rows.append(item)

    type_rows.sort(key=lambda item: item["duration_us"], reverse=True)
    top_by_duration.sort(key=lambda item: item["duration_us"], reverse=True)

    return {
        "total_duration_us": total_duration_us,
        "total_wait_us": total_wait_us,
        "wait_ratio_percent": total_wait_us / total_duration_us * 100.0 if total_duration_us else 0.0,
        "mte_ratio_percent": total_mte_us / total_duration_us * 100.0 if total_duration_us else 0.0,
        "by_type": type_rows[:20],
        "top_by_duration": top_by_duration[:20],
        "short_kernel_counts": dict(
            sorted(short_kernel_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
    }


def _summarize_operators(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    operators = []
    for row in rows:
        host_us = _float(row.get("Host Total Duration(us)"))
        device_us = _float(row.get("Device Total Duration(us)"))
        if host_us <= 0 and device_us <= 0:
            continue
        operators.append({
            "name": row.get("Name", ""),
            "host_total_us": host_us,
            "device_total_us": device_us,
            "device_aicore_total_us": _float(row.get("Device Total Duration With AICore(us)")),
        })
    operators.sort(key=lambda item: max(item["host_total_us"], item["device_total_us"]), reverse=True)
    return operators[:30]


def _summarize_step_trace(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            if key and key not in {"Device_id", "Step"}:
                totals[key] += _float(value)
    stage = totals.get("Stage", 0.0)
    if stage:
        totals["Free Ratio(%)"] = totals.get("Free", 0.0) / stage * 100.0
        totals["Computing Ratio(%)"] = totals.get("Computing", 0.0) / stage * 100.0
    return dict(totals)


def _op_by_name(op_hotspots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in op_hotspots}


def _add_opportunity(
    items: list[dict[str, Any]],
    category: str,
    stage: str,
    evidence: str,
    recommendation: str,
    expected_metric: str,
):
    items.append({
        "category": category,
        "stage": stage,
        "evidence": evidence,
        "recommendation": recommendation,
        "expected_metric": expected_metric,
    })


def _build_opportunities(
    phase: str,
    op_hotspots: list[dict[str, Any]],
    kernel_summary: dict[str, Any],
    step_trace: dict[str, float],
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    ops = _op_by_name(op_hotspots)

    grouped = ops.get("GroupedMatmul")
    if grouped:
        expected = "TPOT" if phase == "decode" else "TTFT and TPOT"
        _add_opportunity(
            opportunities,
            "moe_grouped_matmul",
            "Decode" if phase != "prefill" else "Prefill spillover",
            (
                f"GroupedMatmul uses {grouped['total_us']:.1f} us "
                f"across {grouped['count']} calls ({grouped['ratio_percent']:.1f}%)."
            ),
            (
                "Inspect token-per-expert distribution and grouped matmul shapes; "
                "prioritize tiling, expert batching, and stable-shape grouped matmul paths."
            ),
            expected,
        )

    prefill_total = sum(ops[name]["total_us"] for name in PREFILL_TYPES if name in ops)
    all_total = sum(item["total_us"] for item in op_hotspots)
    if prefill_total > 0 and (phase in {"mixed", "prefill"} or prefill_total / max(all_total, 1.0) > 0.15):
        _add_opportunity(
            opportunities,
            "prefill_attention_or_kv",
            "Prefill",
            f"Prefill-heavy ops account for {prefill_total:.1f} us.",
            (
                "Compare attention, large matmul, RoPE, and KV-cache write kernels by shape; "
                "optimize long-prompt TTFT before applying decode-only fusion."
            ),
            "TTFT",
        )

    fusion_total = sum(ops[name]["total_us"] for name in FUSION_TYPES if name in ops)
    fusion_count = sum(ops[name]["count"] for name in FUSION_TYPES if name in ops)
    if fusion_total > 0 or fusion_count >= 10:
        _add_opportunity(
            opportunities,
            "fusion_candidate",
            "Prefill and Decode",
            f"Fusion candidate ops total {fusion_total:.1f} us across {fusion_count} calls.",
            (
                "Look for adjacent RMSNorm, residual add, bias, SwiGLU, cast, slice, and reshape chains; "
                "merge high-frequency short kernels when tensor shapes match."
            ),
            "TTFT and TPOT",
        )

    routing_total = sum(ops[name]["total_us"] for name in ROUTING_TYPES if name in ops)
    routing_count = sum(ops[name]["count"] for name in ROUTING_TYPES if name in ops)
    if routing_total > 0:
        _add_opportunity(
            opportunities,
            "moe_routing",
            "Decode",
            f"Routing ops total {routing_total:.1f} us across {routing_count} calls.",
            (
                "Profile top-k, routing init, token permute, and unpermute together; "
                "consider fusing routing post-processing with token reorder."
            ),
            "TPOT",
        )

    if kernel_summary.get("wait_ratio_percent", 0.0) >= 10.0:
        _add_opportunity(
            opportunities,
            "scheduler_wait",
            "Prefill and Decode",
            (
                f"Cumulative kernel wait/duration is {kernel_summary['wait_ratio_percent']:.1f}%. "
                "This can exceed 100% because Ascend wait time is summed across kernels and streams."
            ),
            (
                "Check stream dependencies, host launch gaps, hidden synchronizations, "
                "and whether small kernels serialize the decode loop."
            ),
            "TTFT and TPOT",
        )

    if kernel_summary.get("mte_ratio_percent", 0.0) >= 20.0:
        _add_opportunity(
            opportunities,
            "memory_movement",
            "Prefill and Decode",
            f"MTE time ratio is {kernel_summary['mte_ratio_percent']:.1f}%.",
            (
                "Inspect MTE2/MTE3-heavy kernels for memory-layout, format conversion, "
                "and avoidable data movement."
            ),
            "TTFT and TPOT",
        )

    for item in kernel_summary.get("by_type", []):
        if item["type"] == "GroupedMatmul" and item.get("avg_cube_utilization_percent", 0.0) and item[
                "avg_cube_utilization_percent"] < 70.0:
            _add_opportunity(
                opportunities,
                "low_cube_utilization",
                "Decode",
                f"GroupedMatmul average cube utilization is {item['avg_cube_utilization_percent']:.1f}%.",
                (
                    "Treat this as a shape or batching problem before adding more fusion; "
                    "small expert batches often underuse cube."
                ),
                "TPOT",
            )

    free_ratio = step_trace.get("Free Ratio(%)", 0.0)
    if free_ratio >= 20.0:
        _add_opportunity(
            opportunities,
            "service_or_batching_gap",
            "Mixed workload",
            f"Step trace free ratio is {free_ratio:.1f}%.",
            (
                "Correlate profiler windows with request arrival, scheduler batching, and benchmark concurrency; "
                "this may be a service-level gap rather than an operator gap."
            ),
            "TTFT and throughput",
        )

    return opportunities


def analyze_profile(
    phase: str,
    profiler_output: str | Path,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(profiler_output)
    op_rows = _read_csv(output_dir / "op_statistic.csv")
    kernel_rows = _read_csv(output_dir / "kernel_details.csv")
    operator_rows = _read_csv(output_dir / "operator_details.csv")
    step_rows = _read_csv(output_dir / "step_trace_time.csv")

    op_hotspots = _summarize_ops(op_rows)
    kernel_summary = _summarize_kernels(kernel_rows)
    operator_hotspots = _summarize_operators(operator_rows)
    step_trace = _summarize_step_trace(step_rows)
    opportunities = _build_opportunities(phase, op_hotspots, kernel_summary, step_trace)

    return {
        "phase": phase,
        "profiler_output": str(output_dir),
        "benchmark": _load_benchmark(benchmark_path),
        "op_hotspots": op_hotspots[:20],
        "operator_hotspots": operator_hotspots,
        "kernel_summary": kernel_summary,
        "step_trace": step_trace,
        "optimization_opportunities": opportunities,
    }


def _fmt_us(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} ms"
    return f"{value:.1f} us"


def _render_phase(report: dict[str, Any]) -> str:
    phase = report["phase"]
    focus = {
        "mixed": "Macro focus: end-to-end TTFT, TPOT, throughput, and phase balance.",
        "prefill": "TTFT focus: long-prompt attention, large matmul, RoPE, and KV-cache writes.",
        "decode": "TPOT focus: MoE routing, grouped matmul, short kernels, and stream wait.",
    }.get(phase, "Stage focus: inspect profiler hotspots.")

    lines = [f"## {phase}", "", focus, ""]
    benchmark = report.get("benchmark") or {}
    if benchmark:
        metrics = []
        for key in ("median_ttft_ms", "median_tpot_ms", "output_throughput", "total_token_throughput"):
            if key in benchmark:
                metrics.append(f"{key}={benchmark[key]}")
        if metrics:
            lines.extend(["Benchmark: " + ", ".join(metrics), ""])

    lines.extend(["Top OP types:", ""])
    lines.extend(["| OP Type | Count | Total | Avg | Ratio |", "|---|---:|---:|---:|---:|"])
    for item in report["op_hotspots"][:10]:
        lines.append(
            f"| {item['name']} | {item['count']} | {_fmt_us(item['total_us'])} | "
            f"{_fmt_us(item['avg_us'])} | {item['ratio_percent']:.1f}% |")

    lines.extend(["", "Top kernels:", ""])
    lines.extend(["| Kernel Type | Count | Total | Avg | Wait | Cube |", "|---|---:|---:|---:|---:|---:|"])
    for item in report["kernel_summary"]["by_type"][:10]:
        lines.append(
            f"| {item['type']} | {item['count']} | {_fmt_us(item['duration_us'])} | "
            f"{_fmt_us(item['avg_us'])} | {_fmt_us(item['wait_us'])} | "
            f"{item['avg_cube_utilization_percent']:.1f}% |")

    lines.extend(["", "Optimization opportunities:", ""])
    if report["optimization_opportunities"]:
        for item in report["optimization_opportunities"]:
            lines.append(
                f"- {item['category']} ({item['stage']}, {item['expected_metric']}): "
                f"{item['evidence']} {item['recommendation']}")
    else:
        lines.append("- No heuristic opportunities found; inspect trace_view.json manually.")
    lines.append("")
    return "\n".join(lines)


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Ascend MoE Profile Report",
        "",
        "This report is built from Ascend PyTorch Profiler CSV files. Use the mixed phase as the macro view, "
        "then use the prefill and decode phases to separate TTFT and TPOT optimization work.",
        "",
    ]
    for report in reports:
        lines.append(_render_phase(report))
    return "\n".join(lines)


def _parse_phase_arg(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("phase spec must be phase:profiler_output[:benchmark_json]")
    phase = parts[0]
    output = Path(parts[1])
    benchmark = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    return phase, output, benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Ascend PyTorch Profiler output for Qwen3 MoE prefill/decode optimization.")
    parser.add_argument(
        "--phase",
        action="append",
        type=_parse_phase_arg,
        required=True,
        help="Phase input as phase:ASCEND_PROFILER_OUTPUT[:benchmark_json]. Can be used multiple times.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path for machine-readable report JSON.")
    parser.add_argument("--markdown-output", type=Path, help="Optional path for Markdown report.")
    args = parser.parse_args()

    reports = [analyze_profile(phase, output, benchmark) for phase, output, benchmark in args.phase]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    markdown = render_markdown(reports)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
