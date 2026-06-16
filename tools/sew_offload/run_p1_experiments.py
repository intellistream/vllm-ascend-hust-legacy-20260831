# SPDX-License-Identifier: Apache-2.0
"""Run materialized SEW-MoE P1 experiment matrices."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from typing import Any, Iterator

from tools.sew_offload.collect_moe_trace import (
    csv_set,
    load_manifest,
)
from tools.sew_offload.compare_smoke_outputs import (
    load_outputs_jsonl,
    write_comparison_summary,
)
from tools.sew_offload.run_fixed_slot_smoke import (
    SEW_OFFLOAD_ENV_VARS,
    load_config,
    load_inline_prompts_jsonl,
    make_inline_request,
    override_request_max_output_tokens,
    run_smoke,
)

EXPERIMENT_ENV_VARS = (*SEW_OFFLOAD_ENV_VARS, "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH")


def run_experiment_matrix(
    args: argparse.Namespace,
    *,
    config: dict[str, Any] | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matrix = _load_matrix(args.matrix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config) if config is None else config
    requests = _load_requests(args, config) if requests is None else requests

    results = []
    for experiment in matrix.get("experiments", []):
        case_args = _args_for_experiment(args, experiment, output_dir)
        with _patched_env(experiment.get("env") or {}, clear_keys=EXPERIMENT_ENV_VARS):
            summary = run_smoke(case_args, config, requests)
        result = {
            "name": experiment.get("name", ""),
            "description": experiment.get("description", ""),
            "env": experiment.get("env") or {},
            "evidence": experiment.get("evidence") or {},
            "summary": summary,
        }
        gate_summary = _compute_bucket_fast_path_gate_summary([result])
        if gate_summary["total"]:
            result["compute_bucket_fast_path_gate_summary"] = gate_summary
        results.append(result)
    _attach_correctness_comparisons(results, output_dir)
    _attach_baseline_deltas(results)

    aggregate = {
        "version": 1,
        "source_matrix": str(args.matrix),
        "source_plan": matrix.get("source_plan", ""),
        "output_dir": str(output_dir),
        "experiments": results,
        "correctness_vs_baseline": _correctness_vs_baseline(results),
        "compute_bucket_fast_path_gate_summary": _compute_bucket_fast_path_gate_summary(results),
        "best_by_output_throughput_tok_s": _best_by_throughput(results),
        "throughput_delta_vs_baseline": _throughput_delta_vs_baseline(results),
        "throughput_delta_vs_baseline_correct_only": _throughput_delta_vs_baseline_correct_only(results),
        "recommended_correct_experiment": _recommended_correct_experiment(results),
    }
    (output_dir / "p1_experiment_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _load_matrix(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("experiment matrix must be a JSON object")
    return payload


def _args_for_experiment(
    base_args: argparse.Namespace,
    experiment: dict[str, Any],
    output_dir: Path,
) -> argparse.Namespace:
    case_args = copy.copy(base_args)
    name = str(experiment.get("name") or "experiment")
    env = experiment.get("env") or {}
    case_args.experiment_name = name
    case_args.output_dir = str(output_dir / name)
    case_args.mode = _mode_from_env(env)
    case_args.num_slots = _int(env.get("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"), default=0)
    case_args.layered_runtime = env.get("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME") == "1"
    case_args.fanout_threshold = _int(env.get("VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"), default=0)
    case_args.compute_bucket_plan_path = str(env.get("VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH") or "")
    return case_args


def _mode_from_env(env: dict[str, str]) -> str:
    if env.get("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY") == "1":
        return "trace_only"
    if env.get("VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH") and not env.get("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"):
        return "compute_bucket_fast_path"
    if env.get("VLLM_ASCEND_MOE_OFFLOAD_ENABLED") != "1":
        return "no_offload"
    return "fixed_slot_sync"


@contextmanager
def _patched_env(env: dict[str, str], *, clear_keys: tuple[str, ...] = ()) -> Iterator[None]:
    managed_keys = set(clear_keys) | set(env)
    previous = {key: os.environ.get(key) for key in managed_keys}
    try:
        for key in managed_keys:
            os.environ.pop(key, None)
        for key, value in env.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _best_by_throughput(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    best = max(
        results,
        key=lambda item: _float((item.get("summary") or {}).get("output_throughput_tok_s")),
    )
    summary = best.get("summary") or {}
    return {
        "name": best.get("name", ""),
        "output_throughput_tok_s": _float(summary.get("output_throughput_tok_s")),
        "status": summary.get("status", ""),
    }


def _attach_correctness_comparisons(results: list[dict[str, Any]], output_dir: Path) -> None:
    baseline = next((item for item in results if item.get("name") == "baseline"), None)
    if baseline is None:
        return
    baseline_outputs_path = output_dir / "baseline" / "outputs.jsonl"
    if not baseline_outputs_path.exists():
        return
    baseline_outputs = load_outputs_jsonl(baseline_outputs_path)
    for item in results:
        name = str(item.get("name") or "")
        if name == "baseline" or not name:
            continue
        candidate_outputs_path = output_dir / name / "outputs.jsonl"
        if not candidate_outputs_path.exists():
            item["correctness_vs_baseline"] = {
                "status": "missing",
                "reason": "candidate outputs.jsonl not found",
                "candidate_outputs": str(candidate_outputs_path),
            }
            continue
        comparison_path = output_dir / name / "correctness_compare.json"
        item["correctness_vs_baseline"] = write_comparison_summary(
            baseline_outputs=baseline_outputs,
            candidate_outputs=load_outputs_jsonl(candidate_outputs_path),
            output_path=comparison_path,
        )


def _attach_baseline_deltas(results: list[dict[str, Any]]) -> None:
    baseline = next((item for item in results if item.get("name") == "baseline"), None)
    baseline_summary = (baseline or {}).get("summary") or {}
    baseline_throughput = _float(baseline_summary.get("output_throughput_tok_s"))
    for item in results:
        summary = item.get("summary") or {}
        throughput = _float(summary.get("output_throughput_tok_s"))
        delta = throughput - baseline_throughput
        delta_percent = (delta / baseline_throughput * 100.0) if baseline_throughput else 0.0
        item["relative_to_baseline"] = {
            "baseline_name": (baseline or {}).get("name", ""),
            "baseline_output_throughput_tok_s": baseline_throughput,
            "output_throughput_tok_s_delta": delta,
            "output_throughput_tok_s_delta_percent": delta_percent,
        }


def _correctness_vs_baseline(results: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    failed = 0
    missing = 0
    for item in results:
        if item.get("name") == "baseline":
            continue
        comparison = item.get("correctness_vs_baseline")
        if not comparison:
            continue
        checked += 1
        status = comparison.get("status")
        if status == "missing":
            missing += 1
        elif status != "ok":
            failed += 1
    return {
        "status": "ok" if failed == 0 and missing == 0 else "failed",
        "checked": checked,
        "failed": failed,
        "missing": missing,
    }


def _compute_bucket_fast_path_gate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    events = []
    for item in results:
        profile = ((item.get("summary") or {}).get("moe_offload_profile") or {})
        for event in profile.get("events") or []:
            if event.get("name") == "compute_bucket_fast_path_gate":
                events.append(event)

    total = len(events)
    enabled = 0
    reasons: dict[str, int] = {}
    bucket_ids: dict[str, int] = {}
    original_counts = []
    compact_counts = []
    compaction_ratios = []
    for event in events:
        payload = event.get("payload") or {}
        if bool(payload.get("enabled")):
            enabled += 1
            original_count = _float(payload.get("original_expert_count"))
            compact_count = _float(payload.get("compact_expert_count"))
            if original_count:
                original_counts.append(original_count)
                compact_counts.append(compact_count)
                compaction_ratios.append(compact_count / original_count)
            bucket_id = payload.get("bucket_id")
            if bucket_id is not None:
                key = str(bucket_id)
                bucket_ids[key] = bucket_ids.get(key, 0) + 1
        reason = str(payload.get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "total": total,
        "enabled": enabled,
        "fallback": total - enabled,
        "enabled_percent": (enabled / total * 100.0) if total else 0.0,
        "avg_original_expert_count": _mean(original_counts),
        "avg_compact_expert_count": _mean(compact_counts),
        "avg_compaction_ratio": _mean(compaction_ratios),
        "reasons": dict(sorted(reasons.items())),
        "bucket_ids": dict(sorted(bucket_ids.items())),
    }


def _throughput_delta_vs_baseline(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        summary = item.get("summary") or {}
        relative = item.get("relative_to_baseline") or {}
        rows.append({
            "name": item.get("name", ""),
            "status": summary.get("status", ""),
            "output_throughput_tok_s": _float(summary.get("output_throughput_tok_s")),
            "output_throughput_tok_s_delta": _float(relative.get("output_throughput_tok_s_delta")),
            "output_throughput_tok_s_delta_percent": _float(relative.get("output_throughput_tok_s_delta_percent")),
        })
    return sorted(
        rows,
        key=lambda item: (
            item["output_throughput_tok_s_delta_percent"],
            item["output_throughput_tok_s_delta"],
            item["output_throughput_tok_s"],
        ),
        reverse=True,
    )


def _throughput_delta_vs_baseline_correct_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in _throughput_delta_vs_baseline(results)
        if _is_correct_experiment(next((item for item in results if item.get("name") == row["name"]), {}))
    ]


def _recommended_correct_experiment(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct_rows = [row for row in _throughput_delta_vs_baseline_correct_only(results) if row.get("name") != "baseline"]
    if not correct_rows:
        return {}
    row = correct_rows[0]
    item = next((candidate for candidate in results if candidate.get("name") == row["name"]), {})
    comparison = item.get("correctness_vs_baseline") or {}
    return {
        "name": row["name"],
        "status": row["status"],
        "correctness_status": comparison.get("status", "baseline"),
        "output_throughput_tok_s": row["output_throughput_tok_s"],
        "output_throughput_tok_s_delta": row["output_throughput_tok_s_delta"],
        "output_throughput_tok_s_delta_percent": row["output_throughput_tok_s_delta_percent"],
    }


def _is_correct_experiment(item: dict[str, Any]) -> bool:
    if item.get("name") == "baseline":
        return True
    comparison = item.get("correctness_vs_baseline")
    return bool(comparison) and comparison.get("status") == "ok"


def _load_requests(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    if getattr(args, "inline_prompts_jsonl", None) is not None:
        requests = load_inline_prompts_jsonl(
            args.inline_prompts_jsonl,
            default_max_output_tokens=args.inline_max_output_tokens,
        )
    elif getattr(args, "inline_prompt", None) is not None:
        requests = [
            make_inline_request(
                prompt=args.inline_prompt,
                max_output_tokens=args.inline_max_output_tokens,
            )
        ]
    else:
        buckets = csv_set(args.buckets) or None
        manifest = Path(args.manifest or config["dataset"]["manifest_path"])
        args.manifest = str(manifest)
        requests = load_manifest(manifest, buckets, args.max_requests)
    return override_request_max_output_tokens(
        requests,
        max_output_tokens=getattr(args, "override_max_output_tokens", None),
    )


def _float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _int(value: Any, *, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--config", default="docs/sew-offload/benchmark_config.yaml")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--manifest")
    parser.add_argument("--inline-prompt")
    parser.add_argument("--inline-prompts-jsonl")
    parser.add_argument("--inline-max-output-tokens", type=int, default=1)
    parser.add_argument("--override-max-output-tokens", type=int)
    parser.add_argument("--buckets", default="short_chat")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resident-layer-ids", default="")
    parser.add_argument("--release-original-expert-weights", action="store_true")
    parser.add_argument("--compute-bucket-plan-path", default="")
    parser.add_argument("--offload-backend", default="prefetch")
    parser.add_argument("--offload-group-size", type=int, default=4)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument("--offload-params", default="experts")
    parser.add_argument("--with-native-offload-backend", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run_experiment_matrix(parse_args())
    print("P1_EXPERIMENT_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
