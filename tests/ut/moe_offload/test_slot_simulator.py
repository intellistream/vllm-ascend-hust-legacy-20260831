#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import json
import subprocess
import sys
from pathlib import Path

from vllm_ascend.moe_offload.slot_simulator import ExpertSizeTable, SlotSimulator
from vllm_ascend.moe_offload.trace_collector import TraceRecord


def test_slot_simulator_counts_hits_misses_evictions_and_bytes():
    records = [
        TraceRecord(0, 0, "decode", 1, 1, 8, (0, 1), {0: 1, 1: 1}),
        TraceRecord(0, 1, "decode", 1, 1, 8, (1, 2), {1: 1, 2: 1}),
    ]
    size_table = ExpertSizeTable(default_expert_bytes=10)

    summary = SlotSimulator(size_table=size_table).replay(records, num_slots=2, policy_name="lru")

    assert summary.total_records == 2
    assert summary.hit_count == 1
    assert summary.miss_count == 3
    assert summary.eviction_count == 1
    assert summary.host_to_hbm_bytes == 30
    assert summary.phase_opportunity_count == 1


def test_slot_simulator_accepts_jsonable_records():
    records = [
        {
            "layer_id": 0,
            "step_id": 0,
            "mode": "decode",
            "num_tokens": 1,
            "top_k": 2,
            "num_experts": 8,
            "active_experts": [0, 1],
            "expert_token_counts": {"0": 1, "1": 1},
        }
    ]

    summary = SlotSimulator().replay(records, num_slots=4, policy_name="lru")

    assert summary.to_jsonable()["miss_count"] == 2


def test_simulate_expert_slots_cli_outputs_json_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "layer_id": 0,
                        "step_id": 0,
                        "mode": "decode",
                        "num_tokens": 1,
                        "top_k": 2,
                        "num_experts": 8,
                        "active_experts": [0, 1],
                        "expert_token_counts": {"0": 1, "1": 1},
                    }
                ),
                json.dumps(
                    {
                        "layer_id": 0,
                        "step_id": 1,
                        "mode": "decode",
                        "num_tokens": 1,
                        "top_k": 2,
                        "num_experts": 8,
                        "active_experts": [1, 2],
                        "expert_token_counts": {"1": 1, "2": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "tools/sew_offload/simulate_expert_slots.py",
            "--trace",
            str(trace_path),
            "--num-slots",
            "2",
            "--expert-bytes",
            "10",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["hit_count"] == 1
    assert summary["miss_count"] == 3
    assert summary["host_to_hbm_bytes"] == 30
