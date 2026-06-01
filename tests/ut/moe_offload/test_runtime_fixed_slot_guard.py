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

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.moe_offload import PreparedSlotWeights
from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime


def _mock_layer(layer_id: int = 0, num_experts: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        w13_weight=torch.arange(num_experts * 2 * 4, dtype=torch.float32).reshape(num_experts, 2, 4),
        w2_weight=torch.arange(num_experts * 4 * 2, dtype=torch.float32).reshape(num_experts, 4, 2),
    )


def test_runtime_reports_fixed_slot_mode_only_when_non_trace_slots_enabled():
    assert not MoeOffloadRuntime(MoeOffloadConfig(enabled=False, trace_only=False, num_slots=8)).should_use_fixed_slots
    assert not MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True, num_slots=8)).should_use_fixed_slots
    assert not MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=0)).should_use_fixed_slots

    assert MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=8)).should_use_fixed_slots


def test_fixed_slot_prepare_weights_fails_closed_until_log2phy_remap_is_implemented():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=8))

    with pytest.raises(NotImplementedError, match="num_logical_experts"):
        runtime.prepare_weights_for_execution(layer_id=0, active_experts=(0, 1))


def test_runtime_prepares_fixed_slot_plan_with_sync_load_and_log2phy():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    layer = _mock_layer(layer_id=4, num_experts=3)
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    prepared = runtime.prepare_fixed_slot_plan(
        layer_id=4,
        active_experts=(1, 2, 1),
        num_logical_experts=3,
        device=torch.device("cpu"),
    )

    assert isinstance(prepared, PreparedSlotWeights)
    assert prepared.log2phy.tolist() == [-1, 0, 1]
    assert torch.equal(prepared.w1[0], layer.w13_weight[1])
    assert torch.equal(prepared.w2[0], layer.w2_weight[1])
    assert torch.equal(prepared.w1[1], layer.w13_weight[2])
    assert torch.equal(prepared.w2[1], layer.w2_weight[2])


def test_runtime_rejects_active_working_set_larger_than_slot_budget_before_loading():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=1))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=2), slot_device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="exceeds num_slots"):
        runtime.prepare_fixed_slot_plan(
            layer_id=0,
            active_experts=(0, 1),
            num_logical_experts=2,
            device=torch.device("cpu"),
        )


def test_runtime_rejects_out_of_range_active_expert_before_host_lookup():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=3), slot_device=torch.device("cpu"))

    with pytest.raises(ValueError, match="out of range.*num_logical_experts=3.*expert_ids=\\[5\\]"):
        runtime.prepare_fixed_slot_plan(
            layer_id=0,
            active_experts=(1, 5),
            num_logical_experts=3,
            device=torch.device("cpu"),
        )


def test_runtime_reports_fixed_slot_memory_ledger_without_releasing_original_weights():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    layer = _mock_layer(layer_id=7, num_experts=3)

    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))

    ledger = runtime.memory_ledger()
    original_bytes = layer.w13_weight.numel() * layer.w13_weight.element_size()
    original_bytes += layer.w2_weight.numel() * layer.w2_weight.element_size()
    slot_bytes = 2 * layer.w13_weight[0].numel() * layer.w13_weight.element_size()
    slot_bytes += 2 * layer.w2_weight[0].numel() * layer.w2_weight.element_size()

    assert ledger.registered_layers == 1
    assert ledger.host_experts == 3
    assert ledger.original_expert_weight_bytes == original_bytes
    assert ledger.host_store_bytes == original_bytes
    assert ledger.slot_bank_bytes == slot_bytes
    assert ledger.original_expert_weights_retained


def test_runtime_release_readiness_rejects_current_correctness_prototype():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=3), slot_device=torch.device("cpu"))

    plan = runtime.plan_original_weight_release(
        expected_layer_ids=(0,),
        default_path_preserved=False,
        host_store_is_complete=False,
    )

    assert not plan.ready
    assert "default_path_not_preserved" in plan.blockers
    assert "host_store_not_marked_complete" in plan.blockers
    assert "original_expert_weights_still_retained" in plan.blockers


def test_runtime_release_readiness_accepts_explicit_safe_preconditions():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=3), slot_device=torch.device("cpu"))

    plan = runtime.plan_original_weight_release(
        expected_layer_ids=(0,),
        default_path_preserved=True,
        allow_retained_original_weights=True,
    )

    assert plan.ready
    assert plan.blockers == ()
    assert plan.layers_ready == (0,)


def test_runtime_release_readiness_rejects_missing_registered_layers():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=3), slot_device=torch.device("cpu"))

    plan = runtime.plan_original_weight_release(
        expected_layer_ids=(0, 1),
        default_path_preserved=True,
        allow_retained_original_weights=True,
    )

    assert not plan.ready
    assert "layers_not_registered:[1]" in plan.blockers
    assert "host_store_missing_layers:[1]" in plan.blockers
    assert plan.layers_ready == ()


def test_runtime_release_readiness_uses_host_store_self_check_by_default():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=3), slot_device=torch.device("cpu"))
    runtime._host_store._weights.pop(next(key for key in runtime._host_store._weights if key.layer_id == 0 and key.expert_id == 1))

    plan = runtime.plan_original_weight_release(
        expected_layer_ids=(0,),
        default_path_preserved=True,
        allow_retained_original_weights=True,
    )

    assert not plan.ready
    assert "host_store_missing_experts:layer=0,experts=[1]" in plan.blockers
