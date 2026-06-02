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

from vllm_ascend.moe_offload.layered_strategy import LayeredStrategyAnalyzer


def _record(layer_id, active_experts, num_tokens=1):
    return {
        "layer_id": layer_id,
        "step_id": layer_id,
        "mode": "unknown",
        "num_tokens": num_tokens,
        "top_k": 2,
        "num_experts": 8,
        "active_experts": active_experts,
        "expert_token_counts": {str(expert): 1 for expert in active_experts},
    }


def test_layered_strategy_routes_high_fanout_to_full_weight_path():
    records = [
        _record(0, [0, 1, 2], num_tokens=16),
        _record(0, [0, 1], num_tokens=1),
        _record(0, [0, 1], num_tokens=1),
    ]

    summary = LayeredStrategyAnalyzer(num_slots=2).analyze(records)

    assert summary.total_records == 3
    assert summary.full_weight_records == 1
    assert summary.slot_cache_records == 2
    assert summary.slot_cache_miss_count == 2
    assert summary.slot_cache_hit_count == 2
    assert summary.full_weight_layer_ids == (0,)
    assert summary.layers_requiring_full_weight == 1


def test_layered_strategy_can_keep_decode_slots_per_layer():
    records = [
        _record(0, [0, 1], num_tokens=1),
        _record(1, [0, 1], num_tokens=1),
        _record(0, [0, 1], num_tokens=1),
    ]

    global_summary = LayeredStrategyAnalyzer(num_slots=2, cache_scope="global").analyze(records)
    per_layer_summary = LayeredStrategyAnalyzer(num_slots=2, cache_scope="per_layer").analyze(records)

    assert global_summary.slot_cache_hit_count == 0
    assert per_layer_summary.slot_cache_hit_count == 2
    assert per_layer_summary.slot_cache_miss_count == 4


def test_layered_strategy_cli_outputs_json_summary(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(_record(0, [0, 1, 2], num_tokens=16)),
                json.dumps(_record(0, [0, 1], num_tokens=1)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/sew_offload/analyze_layered_strategy.py",
            "--trace",
            str(trace_path),
            "--num-slots",
            "2",
            "--cache-scope",
            "per_layer",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "LAYERED_STRATEGY_SUMMARY " in result.stdout
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["full_weight_records"] == 1
    assert summary["slot_cache_records"] == 1
    assert summary["full_weight_layer_ids"] == [0]
    assert summary["cache_scope"] == "per_layer"
