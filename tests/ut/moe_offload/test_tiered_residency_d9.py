#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime
from vllm_ascend.moe_offload.tiered_residency import parse_comma_separated_ints
from tools.sew_offload.estimate_fixed_slot_memory import compare_slot_budget_models


def _mock_layer(layer_id: int = 0, num_experts: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        w13_weight=torch.arange(num_experts * 2 * 4, dtype=torch.float32).reshape(num_experts, 2, 4),
        w2_weight=torch.arange(num_experts * 4 * 2, dtype=torch.float32).reshape(num_experts, 4, 2),
    )


def test_parse_comma_separated_ints():
    assert parse_comma_separated_ints("") == frozenset()
    assert parse_comma_separated_ints("0, 2 ,4") == frozenset({0, 2, 4})


def test_resident_layer_skips_fixed_slot_plan():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2, resident_layer_ids=frozenset({0}))
    )
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0), slot_device=torch.device("cpu"))

    assert runtime.should_use_fixed_slot_plan_for_layer(0) is False
    with pytest.raises(RuntimeError, match="resident layer"):
        runtime.prepare_fixed_slot_plan(
            layer_id=0,
            active_experts=(0,),
            num_logical_experts=3,
            device=torch.device("cpu"),
        )


def test_release_original_weights_clears_ledger_for_non_resident_layer():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            trace_only=False,
            num_slots=2,
            release_original_expert_weights=True,
        )
    )
    layer = _mock_layer(layer_id=1, num_experts=3)
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    assert runtime.memory_ledger().original_expert_weight_bytes > 0

    plan = runtime.release_original_expert_weights_if_ready(layer)
    assert plan.ready
    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert runtime.memory_ledger().original_expert_weight_bytes == 0


def test_release_disabled_by_default():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=False, num_slots=2))
    layer = _mock_layer(layer_id=0)
    runtime.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    plan = runtime.release_original_expert_weights_if_ready(layer)
    assert not plan.ready
    assert "release_original_expert_weights_disabled" in plan.blockers


def test_compare_slot_budget_models_reports_global_vs_per_layer():
    summary = compare_slot_budget_models(
        num_layers=48,
        num_experts_per_layer=128,
        expert_bytes=100,
        num_slots=8,
        resident_layer_count=4,
        original_weights_retained=False,
    )
    assert summary["per_layer_slot_bank"]["capacity_model"] == "per_layer_slot_bank"
    assert summary["global_slot_bank"]["slot_bank_bytes"] == 8 * 100
    assert summary["per_layer_slot_bank"]["slot_bank_bytes"] == 48 * 8 * 100
    assert summary["global_slot_bank_within_offload_budget"] is True