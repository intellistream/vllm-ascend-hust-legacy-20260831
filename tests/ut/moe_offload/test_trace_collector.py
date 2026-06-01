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

import torch

from vllm_ascend.moe_offload.trace_collector import TraceCollector


def test_trace_collector_records_active_experts_and_counts():
    collector = TraceCollector(max_records=4)
    topk_ids = torch.tensor([[0, 2], [2, 3], [3, 3]], dtype=torch.int32)

    record = collector.record(
        layer_id=7,
        step_id=11,
        topk_ids=topk_ids,
        num_experts=4,
        mode="decode",
    )

    assert record.layer_id == 7
    assert record.step_id == 11
    assert record.mode == "decode"
    assert record.num_tokens == 3
    assert record.top_k == 2
    assert record.active_experts == (0, 2, 3)
    assert record.expert_token_counts == {0: 1, 2: 2, 3: 3}
    assert collector.latest_for_layer(7) == record


def test_trace_collector_keeps_bounded_history():
    collector = TraceCollector(max_records=2)

    for step_id in range(3):
        collector.record(
            layer_id=0,
            step_id=step_id,
            topk_ids=torch.tensor([[step_id]], dtype=torch.int64),
            num_experts=8,
        )

    records = collector.records()

    assert [record.step_id for record in records] == [1, 2]
    assert collector.latest_for_layer(0).step_id == 2


def test_trace_collector_export_is_json_serializable():
    collector = TraceCollector(max_records=4)
    collector.record(
        layer_id=1,
        step_id=2,
        topk_ids=torch.tensor([[1, 2]], dtype=torch.int32),
        num_experts=4,
    )

    exported = collector.to_jsonable()

    assert exported == [
        {
            "layer_id": 1,
            "step_id": 2,
            "mode": "unknown",
            "num_tokens": 1,
            "top_k": 2,
            "num_experts": 4,
            "active_experts": [1, 2],
            "expert_token_counts": {"1": 1, "2": 1},
        }
    ]
