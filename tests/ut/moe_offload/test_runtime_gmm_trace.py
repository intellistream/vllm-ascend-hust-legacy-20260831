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


def test_gmm_trace_writes_grouped_record_when_offload_is_disabled(tmp_path):
    trace_path = tmp_path / "gmm_trace.jsonl"
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=False,
            trace_only=False,
            gmm_trace_path=str(trace_path),
        )
    )

    group_list = torch.tensor([2, 0, 3, 0], dtype=torch.int64)
    returned = runtime.trace_grouped_active_experts(
        layer_id=4,
        group_list=group_list,
        group_list_type=1,
        physical_expert_count=4,
        mode="decode",
    )

    assert returned is group_list
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["source"] == "grouped_dispatch"
    assert records[0]["layer_id"] == 4
    assert records[0]["mode"] == "decode"
    assert records[0]["group_list_type"] == 1
    assert records[0]["group_list_signature"] == "counts:2,0,3,0"
    assert records[0]["physical_expert_count"] == 4
    assert records[0]["fanout"] == 2


def test_gmm_trace_skips_grouped_record_during_graph_capture(tmp_path, monkeypatch):
    trace_path = tmp_path / "gmm_trace.jsonl"
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=False,
            trace_only=False,
            gmm_trace_path=str(trace_path),
        )
    )
    monkeypatch.setattr(
        "vllm_ascend.moe_offload.runtime._is_current_graph_capturing",
        lambda: True,
    )
    group_list = torch.tensor([2, 0, 3, 0], dtype=torch.int64)

    returned = runtime.trace_grouped_active_experts(
        layer_id=4,
        group_list=group_list,
        group_list_type=1,
        physical_expert_count=4,
        mode="decode",
    )

    assert returned is group_list
    assert not trace_path.exists()
