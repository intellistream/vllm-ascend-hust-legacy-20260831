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

from unittest.mock import MagicMock, patch
import json

import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.compute_bucket import ComputeBucketDecisionPath
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime, get_moe_offload_runtime, reset_moe_offload_runtime
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod


def test_trace_only_runtime_records_without_mutating_topk_tensors():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))
    topk_ids = torch.tensor([[0, 2], [2, 3]], dtype=torch.int32)
    topk_weights = torch.randn(2, 2)

    returned_ids, returned_weights = runtime.trace_logical_active_experts(
        layer_id=5,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_logical_experts=4,
    )

    assert returned_ids is topk_ids
    assert returned_weights is topk_weights
    assert runtime.trace_collector.latest_for_layer(5).expert_token_counts == {0: 1, 2: 2, 3: 1}


def test_runtime_traces_logical_and_grouped_active_experts_without_mutation():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))
    topk_ids = torch.tensor([[0, 2], [2, 3]], dtype=torch.int32)
    topk_weights = torch.randn(2, 2)
    group_list = torch.tensor([1, 2, 0, 1], dtype=torch.int64)

    returned_ids, returned_weights = runtime.trace_logical_active_experts(
        layer_id=5,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_logical_experts=4,
        mode="decode",
    )
    returned_group_list = runtime.trace_grouped_active_experts(
        layer_id=5,
        group_list=group_list,
        group_list_type=1,
        physical_expert_count=4,
        mode="decode",
    )

    records = runtime.trace_collector.records()
    assert returned_ids is topk_ids
    assert returned_weights is topk_weights
    assert returned_group_list is group_list
    assert [record.source for record in records] == ["logical_topk", "grouped_dispatch"]
    assert records[0].step_id == records[1].step_id
    assert records[0].fanout == 3
    assert records[1].expert_token_counts == {0: 1, 1: 2, 3: 1}


def test_runtime_classifies_grouped_compute_bucket_when_plan_is_configured(tmp_path):
    plan_path = tmp_path / "sew_moe_p1_plan.json"
    plan_path.write_text(
        json.dumps({
            "version": 1,
            "plans": [
                {
                    "phase": "decode",
                    "target": "P1-C",
                    "compute_bucket_plan": {
                        "version": 1,
                        "phase": "decode",
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "signature": "counts:1,2,1",
                                "sample_count": 8,
                                "coverage_percent": 80.0,
                            }
                        ],
                    },
                }
            ],
        }),
        encoding="utf-8",
    )
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            trace_only=True,
            compute_bucket_plan_path=str(plan_path),
        ))
    group_list = torch.tensor([1, 2, 1], dtype=torch.int64)

    decision = runtime.classify_grouped_compute_bucket(
        layer_id=9,
        group_list=group_list,
        group_list_type=1,
        phase="decode",
    )

    assert decision.path is ComputeBucketDecisionPath.BUCKET
    assert decision.bucket_id == 0
    assert decision.signature == "counts:1,2,1"
    summary = runtime.profiling_summary()
    assert summary["events"][-1]["name"] == "compute_bucket_decision"
    assert summary["events"][-1]["payload"]["path"] == "bucket"
    assert summary["events"][-1]["payload"]["bucket_id"] == 0


def test_runtime_compute_bucket_classifier_is_disabled_without_plan():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))

    decision = runtime.classify_grouped_compute_bucket(
        layer_id=9,
        group_list=torch.tensor([1, 2, 1], dtype=torch.int64),
        group_list_type=1,
        phase="decode",
    )

    assert decision is None
    assert runtime.profiling_summary()["events"] == []


def test_runtime_records_compute_bucket_fast_path_gate():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))

    runtime.record_compute_bucket_fast_path_gate(
        layer_id=9,
        enabled=True,
        reason="eligible",
        bucket_id=3,
        signature="counts:1,2,1",
        original_expert_count=3,
        compact_expert_count=3,
    )

    event = runtime.profiling_summary()["events"][-1]
    assert event["name"] == "compute_bucket_fast_path_gate"
    assert event["layer_id"] == 9
    assert event["payload"] == {
        "enabled": True,
        "reason": "eligible",
        "bucket_id": 3,
        "signature": "counts:1,2,1",
        "original_expert_count": 3,
        "compact_expert_count": 3,
    }


def test_global_runtime_is_disabled_by_default(monkeypatch):
    reset_moe_offload_runtime()
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", raising=False)

    runtime = get_moe_offload_runtime()

    assert runtime.config.enabled is False
    assert runtime.trace_collector.records() == []


def test_moe_apply_hook_records_trace_and_preserves_backend_result(monkeypatch):
    reset_moe_offload_runtime()
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "1")

    moe_config = MagicMock()
    moe_config.has_bias = False
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.moe = moe_config
    method.dynamic_eplb = False
    layer = MagicMock()
    layer.layer_id = 3
    layer.zero_expert_num = 0
    layer.zero_expert_type = None
    layer.n_shared_experts = 0
    layer.vllm_config.model_config = MagicMock(enable_return_routed_experts=False)
    layer.w13_weight = torch.randn(4, 8, 16)
    layer.w2_weight = torch.randn(4, 16, 8)
    layer.w13_bias = None
    layer.w2_bias = None

    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 4)
    selected_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)
    selected_ids = torch.tensor([[0, 2], [2, 3]], dtype=torch.int32)
    backend_result = torch.randn(2, 8)

    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    with (
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.get_moe_num_logical_experts",
            return_value=4,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.select_experts",
            return_value=(selected_weights, selected_ids),
        ),
        patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
    ):
        extra_ctx.moe_comm_method = mock_comm_method
        extra_ctx.moe_comm_type = object()

        result = method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=router_logits,
            renormalize=True,
            num_experts=4,
        )

    runtime = get_moe_offload_runtime()
    record = runtime.trace_collector.latest_for_layer(3)

    assert result is backend_result
    assert record.expert_token_counts == {0: 1, 2: 2, 3: 1}
    assert mock_comm_method.fused_experts.call_count == 1
