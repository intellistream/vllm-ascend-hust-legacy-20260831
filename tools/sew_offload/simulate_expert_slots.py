# SPDX-License-Identifier: Apache-2.0
"""Replay SEW-Offload MoE traces through a fixed-slot simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_ascend.moe_offload.slot_simulator import ExpertSizeTable, SlotSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--num-slots", type=int)
    parser.add_argument(
        "--slot-range",
        help="Inclusive slot sweep as START:STOP[:STEP]. Use instead of --num-slots.",
    )
    parser.add_argument("--policy", default="lru", choices=("lru", "sticky_layer_lru"))
    parser.add_argument("--expert-bytes", type=int, default=14_680_064)
    parser.add_argument("--host-to-hbm-bandwidth-gbps", type=float, default=24.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.num_slots is None and args.slot_range is None:
        parser.error("one of --num-slots or --slot-range is required")
    if args.num_slots is not None and args.slot_range is not None:
        parser.error("--num-slots and --slot-range are mutually exclusive")
    return args


def parse_slot_range(value: str) -> list[int]:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("slot range must be START:STOP[:STEP]")
    start = int(parts[0])
    stop = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    if start <= 0 or stop <= 0:
        raise ValueError("slot range values must be positive")
    if step <= 0:
        raise ValueError("slot range step must be positive")
    if stop < start:
        raise ValueError("slot range stop must be greater than or equal to start")
    return list(range(start, stop + 1, step))


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    records = load_trace(args.trace)
    simulator = SlotSimulator(
        size_table=ExpertSizeTable(default_expert_bytes=args.expert_bytes),
        host_to_hbm_bandwidth_gbps=args.host_to_hbm_bandwidth_gbps,
    )
    if args.slot_range:
        summaries = [
            simulator.replay(records, num_slots=num_slots, policy_name=args.policy).to_jsonable()
            for num_slots in parse_slot_range(args.slot_range)
        ]
        best = min(
            summaries,
            key=lambda item: (
                item["host_to_hbm_bytes"],
                item["miss_count"],
                item["num_slots"],
            ),
        )
        summary = {
            "trace": str(args.trace),
            "policy": args.policy,
            "slot_range": args.slot_range,
            "recommended_num_slots": best["num_slots"],
            "recommended_host_to_hbm_bytes": best["host_to_hbm_bytes"],
            "recommended_miss_count": best["miss_count"],
            "recommended_prefetchable_miss_count": best["prefetchable_miss_count"],
            "recommended_exposed_miss_count": best["exposed_miss_count"],
            "recommended_prefetchable_host_to_hbm_bytes": best["prefetchable_host_to_hbm_bytes"],
            "recommended_exposed_host_to_hbm_bytes": best["exposed_host_to_hbm_bytes"],
            "sweep": summaries,
        }
    else:
        summary = simulator.replay(records, num_slots=args.num_slots, policy_name=args.policy).to_jsonable()
        summary["trace"] = str(args.trace)
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print("SIM_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
