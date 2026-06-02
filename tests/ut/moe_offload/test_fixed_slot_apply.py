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
from contextlib import nullcontext

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime, reset_moe_offload_runtime
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendMoERunner,
    AscendUnquantizedFusedMoEMethod,
    _fixed_slot_device_for_processed_weight,
)


def _enable_fixed_slots(monkeypatch, *, num_slots: int = 2):
    reset_moe_offload_runtime()
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "0")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", str(num_slots))


def _enable_layered_runtime(monkeypatch, *, num_slots: int = 2, fanout_threshold: int = 2):
    _enable_fixed_slots(monkeypatch, num_slots=num_slots)
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD", str(fanout_threshold))


def _method(*, has_bias: bool = False):
    moe_config = MagicMock()
    moe_config.has_bias = has_bias
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.moe = moe_config
    method.dynamic_eplb = False
    return method


def _layer():
    layer = MagicMock()
    layer.layer_id = 6
    layer.zero_expert_num = 0
    layer.zero_expert_type = None
    layer.n_shared_experts = 0
    layer.vllm_config.model_config = MagicMock(enable_return_routed_experts=False)
    layer.w13_weight = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    layer.w2_weight = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    layer.w13_bias = None
    layer.w2_bias = None
    return layer


def _apply_with_fixed_slots(method, layer, mock_comm_method, *, moe_comm_type=MoECommType.ALLGATHER, **kwargs):
    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 3)
    selected_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)
    selected_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)

    with (
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_moe_num_logical_experts", return_value=3),
        patch("vllm_ascend.ops.fused_moe.fused_moe.select_experts", return_value=(selected_weights, selected_ids)),
        patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
    ):
        extra_ctx.moe_comm_method = mock_comm_method
        extra_ctx.moe_comm_type = moe_comm_type

        return method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=router_logits,
            renormalize=True,
            num_experts=3,
            **kwargs,
        )


def _apply_with_selected_ids(method, layer, mock_comm_method, selected_ids, *, moe_comm_type=MoECommType.ALLGATHER, **kwargs):
    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 3)
    selected_weights = torch.ones_like(selected_ids, dtype=torch.float32)

    with (
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_moe_num_logical_experts", return_value=3),
        patch("vllm_ascend.ops.fused_moe.fused_moe.select_experts", return_value=(selected_weights, selected_ids)),
        patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
    ):
        extra_ctx.moe_comm_method = mock_comm_method
        extra_ctx.moe_comm_type = moe_comm_type

        return method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=selected_ids.size(1),
            router_logits=router_logits,
            renormalize=True,
            num_experts=3,
            **kwargs,
        )


def test_fixed_slot_apply_passes_slot_weights_log2phy_and_physical_count(monkeypatch):
    _enable_fixed_slots(monkeypatch)
    method = _method()
    layer = _layer()
    get_moe_offload_runtime().register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    backend_result = torch.randn(2, 8)
    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_fixed_slots(method, layer, mock_comm_method)
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert fused_input.weights.w1 is layer.w13_weight
    assert fused_input.weights.w2 is layer.w2_weight
    assert fused_input.routing.log2phy is None
    assert fused_input.routing.physical_expert_count is None
    assert fused_input.offload.enabled is True
    assert fused_input.offload.layer_id == layer.layer_id
    assert fused_input.offload.num_logical_experts == 3

    reset_moe_offload_runtime()


def test_layered_runtime_low_fanout_uses_slot_cache_path(monkeypatch):
    _enable_layered_runtime(monkeypatch, num_slots=2, fanout_threshold=2)
    method = _method()
    layer = _layer()
    get_moe_offload_runtime().register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    backend_result = torch.randn(2, 8)
    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_selected_ids(
        method,
        layer,
        mock_comm_method,
        torch.tensor([[1, 2], [2, 1]], dtype=torch.int32),
    )
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert fused_input.weights.w1 is layer.w13_weight
    assert fused_input.routing.log2phy is None
    assert fused_input.offload.enabled is True
    assert fused_input.offload.layer_id == layer.layer_id
    assert fused_input.offload.num_logical_experts == 3
    reset_moe_offload_runtime()


def test_layered_runtime_high_fanout_uses_full_weight_path(monkeypatch):
    _enable_layered_runtime(monkeypatch, num_slots=2, fanout_threshold=2)
    method = _method()
    layer = _layer()
    get_moe_offload_runtime().register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    backend_result = torch.randn(2, 8)
    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_selected_ids(
        method,
        layer,
        mock_comm_method,
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
    )
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert fused_input.weights.w1 is layer.w13_weight
    assert fused_input.weights.w2 is layer.w2_weight
    assert fused_input.routing.log2phy is None
    assert fused_input.routing.physical_expert_count is None
    assert fused_input.offload.enabled is True
    reset_moe_offload_runtime()


def test_layered_runtime_apply_defers_fail_closed_to_fused_experts_boundary(monkeypatch):
    _enable_layered_runtime(monkeypatch, num_slots=2, fanout_threshold=2)
    method = _method()
    layer = _layer()
    runtime = get_moe_offload_runtime()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    runtime._released_original_weight_layers.add(layer.layer_id)
    mock_comm_method = MagicMock()

    backend_result = torch.randn(2, 8)
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_selected_ids(
        method,
        layer,
        mock_comm_method,
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
    )
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert fused_input.offload.enabled is True
    assert mock_comm_method.fused_experts.call_count == 1
    reset_moe_offload_runtime()


def test_default_apply_preserves_original_weights_and_routing(monkeypatch):
    reset_moe_offload_runtime()
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", raising=False)
    method = _method()
    layer = _layer()
    backend_result = torch.randn(2, 8)
    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_fixed_slots(method, layer, mock_comm_method)
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert fused_input.weights.w1 is layer.w13_weight
    assert fused_input.weights.w2 is layer.w2_weight
    assert fused_input.routing.log2phy is None
    assert fused_input.routing.physical_expert_count is None

    reset_moe_offload_runtime()


def test_fixed_slot_apply_rejects_backend_device_mismatch(monkeypatch):
    _enable_fixed_slots(monkeypatch)
    layer = _layer()
    runtime = get_moe_offload_runtime()
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    prepared = runtime.prepare_fixed_slot_plan(
        layer_id=layer.layer_id,
        active_experts=(1, 2),
        num_logical_experts=3,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="backend device mismatch"):
        prepared.validate_backend_ready(expected_device_type="npu")
    reset_moe_offload_runtime()


def test_fixed_slot_registration_uses_current_npu_for_cpu_offloaded_weights():
    weight = torch.empty(1)

    with patch("torch.npu.current_device", return_value=3):
        slot_device = _fixed_slot_device_for_processed_weight(weight)

    assert slot_device == torch.device("npu", 3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expert_map": torch.tensor([0, 1, 2], dtype=torch.int32)}, "expert_map"),
        ({"global_redundant_expert_num": 1}, "redundant experts"),
    ],
)
def test_fixed_slot_apply_rejects_unsupported_routing_modes(monkeypatch, kwargs, match):
    _enable_fixed_slots(monkeypatch)
    mock_comm_method = MagicMock()

    with pytest.raises(NotImplementedError, match=match):
        _apply_with_fixed_slots(_method(), _layer(), mock_comm_method, **kwargs)

    assert mock_comm_method.fused_experts.call_count == 0
    reset_moe_offload_runtime()


def test_fixed_slot_apply_constrains_profile_force_load_balance_to_slot_budget(monkeypatch):
    _enable_fixed_slots(monkeypatch, num_slots=2)
    method = _method()
    layer = _layer()
    get_moe_offload_runtime().register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    backend_result = torch.randn(2, 8)
    mock_comm_method = MagicMock()
    mock_comm_method.fused_experts.return_value = backend_result

    result = _apply_with_fixed_slots(
        method,
        layer,
        mock_comm_method,
        enable_force_load_balance=True,
    )
    fused_input = mock_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]

    assert result is backend_result
    assert sorted(set(fused_input.topk_ids.flatten().tolist())) == [0, 1]
    assert fused_input.routing.log2phy is None
    assert fused_input.offload.enabled is True
    reset_moe_offload_runtime()


def test_fixed_slot_apply_rejects_profile_force_load_balance_when_topk_exceeds_slots(monkeypatch):
    _enable_fixed_slots(monkeypatch, num_slots=1)
    mock_comm_method = MagicMock()

    with pytest.raises(RuntimeError, match="num_slots"):
        _apply_with_fixed_slots(
            _method(),
            _layer(),
            mock_comm_method,
            enable_force_load_balance=True,
        )

    assert mock_comm_method.fused_experts.call_count == 0
    reset_moe_offload_runtime()


def test_fixed_slot_apply_rejects_non_allgather_comm(monkeypatch):
    _enable_fixed_slots(monkeypatch)
    mock_comm_method = MagicMock()

    with pytest.raises(NotImplementedError, match="AllGather"):
        _apply_with_fixed_slots(_method(), _layer(), mock_comm_method, moe_comm_type=MoECommType.MC2)

    assert mock_comm_method.fused_experts.call_count == 0
    reset_moe_offload_runtime()


def test_fixed_slot_apply_rejects_bias_until_slot_bias_is_supported(monkeypatch):
    _enable_fixed_slots(monkeypatch)
    mock_comm_method = MagicMock()

    with pytest.raises(NotImplementedError, match="bias"):
        _apply_with_fixed_slots(_method(has_bias=True), _layer(), mock_comm_method)

    assert mock_comm_method.fused_experts.call_count == 0
    reset_moe_offload_runtime()


def test_fixed_slot_apply_rejects_zero_expert_path(monkeypatch):
    _enable_fixed_slots(monkeypatch)
    layer = _layer()
    layer.zero_expert_num = 1
    layer.zero_expert_type = "zero"
    mock_comm_method = MagicMock()

    with pytest.raises(NotImplementedError, match="zero expert"):
        _apply_with_fixed_slots(_method(), layer, mock_comm_method)

    assert mock_comm_method.fused_experts.call_count == 0
    reset_moe_offload_runtime()


def test_ascend_moe_runner_runs_internal_gate_before_layer_forward():
    hidden_states = torch.randn(2, 8)
    placeholder_router_logits = hidden_states
    gate_logits = torch.randn(2, 3)
    layer = MagicMock()
    layer.forward_impl.return_value = torch.randn(2, 8)
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.gate = MagicMock(return_value=(gate_logits, None))
    runner._shared_experts = None
    runner._sequence_parallel_context = MagicMock(return_value=nullcontext())

    runner._forward_impl(
        layer,
        hidden_states,
        placeholder_router_logits,
        shared_experts_input=None,
    )

    runner.gate.assert_called_once_with(hidden_states)
    layer.forward_impl.assert_called_once_with(hidden_states, gate_logits)
