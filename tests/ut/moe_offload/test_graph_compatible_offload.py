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
"""Option 2: graph-compatible offload via decision/execution decoupling.

These tests prove the core primitives that let ACLGraph capture a MoE offload
layer without the forbidden device->host sync:

- The persistent log2phy buffer has a STABLE address across staging calls
  (in-place update, not re-allocation) -- the attn-param-style hoisting.
- stage_fixed_slot_plan (eager) writes the real decision into that buffer.
- capture_safe_slot_weights (capture path) points routing at the fixed slot
  tensors + the fixed log2phy buffer with NO host sync and NO H2D staging.

CPU-only; no NPU required.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime


def _mock_layer(layer_id: int = 0, num_experts: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        w13_weight=torch.arange(num_experts * 2 * 4, dtype=torch.float32).reshape(num_experts, 2, 4),
        w2_weight=torch.arange(num_experts * 4 * 2, dtype=torch.float32).reshape(num_experts, 4, 2),
    )


def _make_runtime(num_slots: int = 2, num_experts: int = 4) -> tuple[MoeOffloadRuntime, int]:
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, trace_only=False, num_slots=num_slots, graph_compatible_offload=True)
    )
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=num_experts), slot_device=torch.device("cpu"))
    return runtime, num_experts


def test_log2phy_buffer_allocated_at_register_with_logical_expert_size():
    runtime, num_experts = _make_runtime(num_slots=2, num_experts=4)
    buf = runtime.log2phy_buffer(0)
    assert buf is not None
    assert buf.shape == (num_experts,)
    assert buf.dtype == torch.int32
    # initialized to the -1 sentinel (no expert mapped yet)
    assert torch.equal(buf, torch.full((num_experts,), -1, dtype=torch.int32))


def test_stage_updates_log2phy_buffer_in_place_stable_address():
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    buf = runtime.log2phy_buffer(0)
    addr_before = buf.data_ptr()

    prepared = runtime.stage_fixed_slot_plan(layer_id=0, active_experts=(1, 2), num_logical_experts=4)

    # The persistent buffer address is unchanged -> graph can capture against it.
    assert runtime.log2phy_buffer(0).data_ptr() == addr_before
    # stage returns the persistent buffer itself, not a fresh allocation.
    assert prepared.log2phy.data_ptr() == addr_before
    # contents now reflect the decision: experts 1,2 mapped to slots, others -1.
    log2phy = runtime.log2phy_buffer(0)
    assert int(log2phy[1]) >= 0
    assert int(log2phy[2]) >= 0
    assert int(log2phy[0]) == -1
    assert int(log2phy[3]) == -1


def test_restage_reuses_same_buffer_address():
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    runtime.stage_fixed_slot_plan(layer_id=0, active_experts=(1, 2), num_logical_experts=4)
    addr1 = runtime.log2phy_buffer(0).data_ptr()
    # second decode step, different active set within slot budget
    runtime.stage_fixed_slot_plan(layer_id=0, active_experts=(2, 3), num_logical_experts=4)
    addr2 = runtime.log2phy_buffer(0).data_ptr()
    assert addr1 == addr2  # stable address across steps == replayable


def test_capture_safe_weights_point_at_fixed_buffers_no_active_set():
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    # stage first so slots are populated
    runtime.stage_fixed_slot_plan(layer_id=0, active_experts=(1, 2), num_logical_experts=4)

    capture = runtime.capture_safe_slot_weights(layer_id=0)
    assert capture is not None
    # log2phy IS the persistent buffer (same address) -> no fresh allocation,
    # no host sync to build it.
    assert capture.log2phy.data_ptr() == runtime.log2phy_buffer(0).data_ptr()
    # w1/w2 are the fixed slot backing tensors.
    bank = runtime._slot_banks[0]
    assert capture.w1.data_ptr() == bank.w13_slots.data_ptr()
    assert capture.w2.data_ptr() == bank.w2_slots.data_ptr()
    assert capture.physical_expert_count == 2


def test_capture_safe_weights_none_for_unregistered_layer():
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    assert runtime.capture_safe_slot_weights(layer_id=99) is None


def test_stage_refuses_during_capture(monkeypatch):
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    import vllm_ascend.moe_offload.runtime as rt

    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must run eager"):
        runtime.stage_fixed_slot_plan(layer_id=0, active_experts=(1, 2), num_logical_experts=4)
