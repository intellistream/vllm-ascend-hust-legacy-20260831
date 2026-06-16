# SPDX-License-Identifier: Apache-2.0
"""Materialize SEW-MoE P1 plan artifacts into benchmark experiment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def materialize_experiments(
    plan_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(plan_path)
    with source.open(encoding="utf-8") as f:
        plan = json.load(f)

    experiments = [_baseline_experiment()]
    compute_plan = _first_compute_bucket_plan(plan)
    if compute_plan is not None:
        experiments.append(_compute_bucket_trace_only_experiment(source, compute_plan))
        experiments.append(_compute_bucket_fast_path_experiment(source, compute_plan))

    slot_plan = _first_slot_sweep_plan(plan)
    if slot_plan is not None:
        experiments.append(_fixed_slot_experiment(slot_plan))
    if compute_plan is not None and slot_plan is not None:
        experiments.append(_combined_compute_bucket_and_slot_experiment(source, compute_plan, slot_plan))

    matrix = {
        "version": 1,
        "source_plan": str(source),
        "experiments": experiments,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix


def _baseline_experiment() -> dict[str, Any]:
    return {
        "name": "baseline",
        "description": "Current non-offload baseline with SEW knobs disabled.",
        "env": {},
        "expected_effect": "reference throughput and correctness",
    }


def _compute_bucket_evidence(compute_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": compute_plan.get("phase", "unknown"),
        "bucket_count": len(compute_plan.get("buckets", [])),
        "coverage_percent": compute_plan.get("coverage_percent", 0.0),
        "fallback_percent": compute_plan.get("fallback_percent", 100.0),
    }


def _compute_bucket_trace_only_experiment(plan_path: Path, compute_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "p1_compute_bucket_trace_only",
        "description": "Enable trace-only P1-C grouped signature classification before the existing MLP fallback.",
        "env": {
            "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
            "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "1",
            "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": str(plan_path),
        },
        "evidence": _compute_bucket_evidence(compute_plan),
        "expected_effect": "classification overhead and fast-path eligibility measurement; math path remains fallback",
    }


def _compute_bucket_fast_path_experiment(plan_path: Path, compute_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "p1_compute_bucket_fast_path",
        "description": "Enable P1-C grouped signature classification and allow the unquantized active-expert compaction path.",
        "env": {
            "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": str(plan_path),
        },
        "evidence": _compute_bucket_evidence(compute_plan),
        "expected_effect": "measure stable grouped-shape fast-path impact without fixed-slot offload",
    }


def _fixed_slot_experiment(plan: dict[str, Any]) -> dict[str, Any]:
    sweep = plan.get("slot_sweep_result") or {}
    num_slots = _int(sweep.get("recommended_num_slots"))
    env = {
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": str(num_slots),
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY": "lru",
        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": str(num_slots),
    }
    return {
        "name": "p1_fixed_slot_recommended",
        "description": "Run fixed-slot offload with the slot budget recommended by trace-backed sweep.",
        "env": env,
        "evidence": {
            "phase": plan.get("phase", "unknown"),
            "slot_sweep_result_json": plan.get("slot_sweep_result_json", ""),
            "recommended_num_slots": num_slots,
            "recommended_miss_count": _int(sweep.get("recommended_miss_count")),
            "recommended_host_to_hbm_bytes": _int(sweep.get("recommended_host_to_hbm_bytes")),
            "recommended_prefetchable_miss_count": _int(sweep.get("recommended_prefetchable_miss_count")),
            "recommended_exposed_miss_count": _int(sweep.get("recommended_exposed_miss_count")),
            "recommended_prefetchable_host_to_hbm_bytes": _int(
                sweep.get("recommended_prefetchable_host_to_hbm_bytes")
            ),
            "recommended_exposed_host_to_hbm_bytes": _int(sweep.get("recommended_exposed_host_to_hbm_bytes")),
        },
        "expected_effect": "lower HBM residency with measured transfer/miss budget",
    }


def _combined_compute_bucket_and_slot_experiment(
    plan_path: Path,
    compute_plan: dict[str, Any],
    slot_plan: dict[str, Any],
) -> dict[str, Any]:
    compute_case = _compute_bucket_fast_path_experiment(plan_path, compute_plan)
    slot_case = _fixed_slot_experiment(slot_plan)
    env = dict(slot_case["env"])
    env["VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH"] = str(plan_path)
    evidence = dict(compute_case["evidence"])
    evidence.update(slot_case["evidence"])
    return {
        "name": "p1_compute_bucket_plus_fixed_slot",
        "description": (
            "Run the trace-backed P1-C compute bucket gate together with the "
            "recommended P1-T fixed-slot offload budget."
        ),
        "env": env,
        "evidence": evidence,
        "expected_effect": "measure whether stable grouped-shape gating composes with fixed-slot offload",
    }


def _first_compute_bucket_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    for item in plan.get("plans", []):
        compute_plan = item.get("compute_bucket_plan")
        if compute_plan:
            return compute_plan
    return None


def _first_slot_sweep_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    for item in plan.get("plans", []):
        if item.get("slot_sweep_result"):
            return item
    return None


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = materialize_experiments(args.plan, output_path=args.output)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
