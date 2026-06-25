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
"""CPU-only correctness checks for fixed-slot staging + remap (the path SHARED
by both eager and captured SEW execution).

Motivation: eager-SEW (H) diverges from no-offload baseline IDENTICALLY to
captured SEW (G). Since eager never reads the persistent log2phy buffer, the
defect must live in the slot staging / remap that both paths share. These tests
isolate whether the *weight-level* round-trip is lossless and whether the remap
math is correct, on CPU, without an NPU.

If these PASS, the divergence is NOT in weight-staging/remap and must be either
(a) a numerical kernel difference (slot-packed grouped matmul vs resident), or
(b) a surrounding-pipeline consistency bug (gate weights / group_list / dispatch).
"""
from types import SimpleNamespace

import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.expert_key import ExpertKey
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime


def _mock_layer(layer_id: int, num_experts: int, h: int = 8, i: int = 6) -> SimpleNamespace:
    # Distinct, non-trivial per-expert weights so any mis-mapping is detectable.
    torch.manual_seed(1234 + layer_id)
    return SimpleNamespace(
        layer_id=layer_id,
        w13_weight=torch.randn(num_experts, 2 * i, h, dtype=torch.float32),
        w2_weight=torch.randn(num_experts, h, i, dtype=torch.float32),
    )


def _runtime(num_slots: int, num_experts: int, layer_id: int = 0):
    rt = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, trace_only=False, num_slots=num_slots, graph_compatible_offload=True)
    )
    layer = _mock_layer(layer_id, num_experts)
    rt.register_layer_for_fixed_slots(layer, slot_device=torch.device("cpu"))
    return rt, layer


def test_staged_slot_weights_are_elementwise_lossless():
    """slot_bank.w13_slots[log2phy[e]] must equal original w13_weight[e] exactly."""
    num_experts = 8
    rt, layer = _runtime(num_slots=num_experts, num_experts=num_experts)
    rt.stage_fixed_slot_plan(
        layer_id=0, active_experts=tuple(range(num_experts)), num_logical_experts=num_experts
    )
    log2phy = rt.log2phy_buffer(0)
    bank = rt._slot_banks[0]
    for e in range(num_experts):
        slot_id = int(log2phy[e])
        assert slot_id >= 0, f"expert {e} unmapped"
        assert torch.equal(bank.w13_slots[slot_id], layer.w13_weight[e]), f"w13 mismatch expert {e}"
        assert torch.equal(bank.w2_slots[slot_id], layer.w2_weight[e]), f"w2 mismatch expert {e}"


def test_log2phy_is_a_permutation_when_all_experts_fit():
    num_experts = 8
    rt, _ = _runtime(num_slots=num_experts, num_experts=num_experts)
    rt.stage_fixed_slot_plan(
        layer_id=0, active_experts=tuple(range(num_experts)), num_logical_experts=num_experts
    )
    log2phy = rt.log2phy_buffer(0)
    slots = sorted(int(log2phy[e]) for e in range(num_experts))
    assert slots == list(range(num_experts)), f"log2phy not a permutation: {slots}"


def test_remap_recovers_correct_expert_weights_via_gather():
    """The end-to-end invariant the MoE kernel relies on: gathering slot weights
    by the remapped physical ids reproduces the per-token expert weights that a
    resident (identity) layout would have used."""
    num_experts = 8
    rt, layer = _runtime(num_slots=num_experts, num_experts=num_experts)
    rt.stage_fixed_slot_plan(
        layer_id=0, active_experts=tuple(range(num_experts)), num_logical_experts=num_experts
    )
    log2phy = rt.log2phy_buffer(0)
    bank = rt._slot_banks[0]

    # A fake routing: tokens pick experts [3,0,7,5].
    topk_ids = torch.tensor([3, 0, 7, 5], dtype=torch.long)
    phys = log2phy[topk_ids].long()
    gathered_w13 = bank.w13_slots[phys]
    expected_w13 = layer.w13_weight[topk_ids]
    assert torch.equal(gathered_w13, expected_w13), "remap+gather does not recover resident weights"
