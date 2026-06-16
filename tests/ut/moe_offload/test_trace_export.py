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

import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_ascend.moe_offload.trace_collector import TraceCollector


def test_trace_collector_exports_jsonl_records(tmp_path):
    collector = TraceCollector(max_records=4)
    collector.record(
        layer_id=3,
        step_id=11,
        topk_ids=torch.tensor([[0, 7], [7, 31]], dtype=torch.int32),
        num_experts=128,
        mode="decode",
    )
    collector.record(
        layer_id=4,
        step_id=12,
        topk_ids=torch.tensor([[2, 2]], dtype=torch.int32),
        num_experts=128,
        mode="prefill",
    )

    jsonl = collector.to_jsonl()
    lines = jsonl.splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "layer_id": 3,
        "step_id": 11,
        "mode": "decode",
        "source": "logical_topk",
        "num_tokens": 2,
        "top_k": 2,
        "num_logical_experts": 128,
        "fanout": 3,
        "active_experts": [0, 7, 31],
        "expert_token_counts": {"0": 1, "7": 2, "31": 1},
        "group_list_type": None,
        "group_list_signature": None,
        "physical_expert_count": None,
    }
    assert json.loads(lines[1])["expert_token_counts"] == {"2": 2}

    output_path = tmp_path / "nested" / "trace.jsonl"
    collector.write_jsonl(output_path)

    assert output_path.read_text(encoding="utf-8") == jsonl


def test_runtime_exports_trace_jsonl(tmp_path):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))
    runtime.trace_routing(
        layer_id=5,
        topk_ids=torch.tensor([[1, 3]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        num_experts=8,
        mode="decode",
    )

    output_path = tmp_path / "trace.jsonl"
    num_records = runtime.export_trace(output_path)

    exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert num_records == 1
    assert exported[0]["layer_id"] == 5
    assert exported[0]["expert_token_counts"] == {"1": 1, "3": 1}


def test_runtime_appends_trace_jsonl_when_trace_path_is_set(tmp_path, monkeypatch):
    trace_path = tmp_path / "child-process-trace.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH", str(trace_path))
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))

    runtime.trace_routing(
        layer_id=6,
        topk_ids=torch.tensor([[1, 7], [7, 8]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.6, 0.4], [0.9, 0.1]], dtype=torch.float32),
        num_experts=16,
        mode="decode",
    )

    exported = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert exported == [
        {
            "active_experts": [1, 7, 8],
            "expert_token_counts": {"1": 1, "7": 2, "8": 1},
            "layer_id": 6,
            "mode": "decode",
            "source": "logical_topk",
            "num_logical_experts": 16,
            "num_tokens": 2,
            "fanout": 3,
            "step_id": 0,
            "top_k": 2,
            "group_list_type": None,
            "group_list_signature": None,
            "physical_expert_count": None,
        }
    ]


def test_disabled_runtime_exports_empty_trace(tmp_path):
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=False, trace_only=False))
    runtime.trace_routing(
        layer_id=0,
        topk_ids=torch.tensor([[1]], dtype=torch.int32),
        topk_weights=torch.tensor([[1.0]], dtype=torch.float32),
        num_experts=2,
    )

    output_path = tmp_path / "empty.jsonl"
    num_records = runtime.export_trace(output_path)

    assert num_records == 0
    assert output_path.read_text(encoding="utf-8") == ""
