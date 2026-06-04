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

"""MVP-D.11 phase split unit tests."""

import json
import math
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.moe_offload.phase_split import (
    MoEPhase,
    MoEPhasePlan,
    PhaseSplitProfileEvent,
    _build_phase_group_list,
    _extract_phase_tokens,
    _scatter_phase_output,
    _slice_expert_weights,
    _write_phase_split_profile_jsonl,
    compute_expert_token_slices,
    execute_phased_mlp,
    plan_hit_miss_phases,
)
from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
    MoEMlpComputeInput,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dummy_weights(num_experts: int, hidden_size: int, ffn_size: int) -> MoEWeights:
    """Create dummy weights with deterministic values per expert."""
    w1 = torch.zeros(num_experts, ffn_size * 2, hidden_size)
    w2 = torch.zeros(num_experts, hidden_size, ffn_size)
    for e in range(num_experts):
        # Each expert's weights encode its expert id so output is distinguishable.
        w1[e] = float(e + 1)
        w2[e] = float(e + 1)
    return MoEWeights(w1=w1, w2=w2)


def _dummy_mlp_fn(mlp_compute_input):
    """Deterministic MLP that uses actual weight values for equivalence testing.

    Uses square weight matrices for simplicity:
        out = x @ w1[e] @ w2[e]   →  [c, H]

    This depends on the actual weight *values*, so slicing the weight tensors
    correctly yields identical per-token results to the full un-sliced call.
    """
    w1 = mlp_compute_input.weights.w1  # [num_experts, H, H]
    w2 = mlp_compute_input.weights.w2  # [num_experts, H, H]
    hidden_states = mlp_compute_input.hidden_states  # [T, H]
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type

    hidden_size = w1.size(-1)

    if group_list_type == 1:
        counts = [int(c) for c in group_list.cpu().tolist()]
    else:
        cumsum = [int(c) for c in group_list.cpu().tolist()]
        counts = [cumsum[0]] + [cumsum[i] - cumsum[i - 1] for i in range(1, len(cumsum))]

    outputs: list[torch.Tensor] = []
    offset = 0
    for expert_idx, count in enumerate(counts):
        if int(count) == 0:
            continue
        token_slice = hidden_states[offset : offset + int(count)]  # [c, H]
        # out = x @ w1 @ w2   all [H, H]
        out = token_slice @ w1[expert_idx] @ w2[expert_idx]  # [c, H]
        outputs.append(out)
        offset += int(count)

    if not outputs:
        return torch.empty(0, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# compute_expert_token_slices
# ---------------------------------------------------------------------------


class TestComputeExpertTokenSlices:
    def test_type1_count_mode(self):
        group_list = torch.tensor([3, 0, 5, 2], dtype=torch.int32)
        slices = compute_expert_token_slices(group_list, group_list_type=1)
        assert slices == [(0, 3), (3, 3), (3, 8), (8, 10)]

    def test_type0_cumsum_mode(self):
        group_list = torch.tensor([3, 3, 8, 10], dtype=torch.int32)
        slices = compute_expert_token_slices(group_list, group_list_type=0)
        assert slices == [(0, 3), (3, 3), (3, 8), (8, 10)]

    def test_empty_group(self):
        group_list = torch.empty(0, dtype=torch.int32)
        slices = compute_expert_token_slices(group_list, group_list_type=1)
        assert slices == []

    def test_unsupported_type_raises(self):
        group_list = torch.tensor([3, 5], dtype=torch.int32)
        with pytest.raises(ValueError, match="Unsupported group_list_type"):
            compute_expert_token_slices(group_list, group_list_type=2)


# ---------------------------------------------------------------------------
# plan_hit_miss_phases
# ---------------------------------------------------------------------------


class TestPlanHitMissPhases:
    def test_all_hit_single_phase(self):
        slices = [(0, 3), (3, 5), (5, 8)]
        expert_ids = (0, 1, 2)
        readiness = {0: True, 1: True, 2: True}
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness)
        assert plan.total_phases == 1
        assert plan.hit_phases == 1
        assert plan.miss_phases == 0
        assert plan.reason == "all_hit"

    def test_all_miss_single_phase(self):
        slices = [(0, 3), (3, 5)]
        expert_ids = (0, 1)
        readiness = {0: False, 1: False}
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness)
        assert plan.total_phases == 1
        assert plan.miss_phases == 1
        assert plan.hit_phases == 0

    def test_hit_miss_split(self):
        slices = [(0, 3), (3, 5), (5, 8)]
        expert_ids = (0, 1, 2)
        readiness = {0: True, 1: False, 2: True}
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness)
        assert plan.total_phases == 2
        assert plan.hit_phases == 1
        assert plan.miss_phases == 1
        # Hit phase should have experts 0, 2
        hit_phase = next(p for p in plan.phases if p.is_hit)
        assert set(hit_phase.expert_indices) == {0, 2}
        # Miss phase should have expert 1
        miss_phase = next(p for p in plan.phases if not p.is_hit)
        assert set(miss_phase.expert_indices) == {1}

    def test_none_readiness_single_phase(self):
        slices = [(0, 3), (3, 5)]
        expert_ids = (0, 1)
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=None)
        assert plan.total_phases == 1
        assert plan.hit_phases == 1
        assert plan.reason == "single_phase"

    def test_max_phases_one_forces_single(self):
        slices = [(0, 3), (3, 5)]
        expert_ids = (0, 1)
        readiness = {0: True, 1: False}
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness, max_phases=1)
        assert plan.total_phases == 1
        assert plan.reason == "single_phase"

    def test_mismatch_lengths_raises(self):
        slices = [(0, 3)]
        expert_ids = (0, 1)
        with pytest.raises(ValueError, match="Mismatched lengths"):
            plan_hit_miss_phases(slices, expert_ids, slot_readiness={0: True, 1: True})

    def test_empty_experts_fallback(self):
        slices = []
        expert_ids = tuple()
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness={})
        assert plan.total_phases == 1
        assert plan.total_tokens == 0

    def test_jsonable_output(self):
        slices = [(0, 3), (3, 5)]
        expert_ids = (0, 1)
        readiness = {0: True, 1: False}
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness)
        data = plan.to_jsonable()
        assert data["total_phases"] == 2
        assert len(data["phases"]) == 2


# ---------------------------------------------------------------------------
# Token extraction / group_list / weight slicing helpers
# ---------------------------------------------------------------------------


class TestPhaseHelpers:
    def test_extract_phase_tokens_contiguous(self):
        hidden = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = _extract_phase_tokens(hidden, ((0, 2),))
        assert result.shape == (2, 1)
        assert torch.equal(result, hidden[:2])

    def test_extract_phase_tokens_noncontiguous(self):
        hidden = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = _extract_phase_tokens(hidden, ((0, 1), (3, 5)))
        assert result.shape == (3, 1)
        assert torch.equal(result, torch.tensor([[1.0], [4.0], [5.0]]))

    def test_extract_phase_tokens_empty(self):
        hidden = torch.tensor([[1.0], [2.0]])
        result = _extract_phase_tokens(hidden, tuple())
        assert result.numel() == 0

    def test_build_phase_group_list_type1(self):
        group_list = torch.tensor([3, 5, 2], dtype=torch.int32)
        result = _build_phase_group_list(group_list, 1, (0, 2))
        assert torch.equal(result, torch.tensor([3, 2], dtype=torch.int32))

    def test_build_phase_group_list_type0(self):
        group_list = torch.tensor([3, 8, 10], dtype=torch.int32)
        result = _build_phase_group_list(group_list, 0, (0, 2))
        assert torch.equal(result, torch.tensor([3, 5], dtype=torch.int32))

    def test_slice_expert_weights(self):
        w1 = torch.randn(4, 8, 8)
        w2 = torch.randn(4, 8, 8)
        weights = MoEWeights(w1=w1, w2=w2)
        sliced = _slice_expert_weights(weights, (0, 2))
        assert sliced.w1.shape == (2, 8, 8)
        assert sliced.w2.shape == (2, 8, 8)
        assert torch.equal(sliced.w1, w1[[0, 2]])
        assert torch.equal(sliced.w2, w2[[0, 2]])

    def test_slice_expert_weights_with_none_fields(self):
        w1 = torch.randn(3, 8, 8)
        w2 = torch.randn(3, 8, 8)
        weights = MoEWeights(w1=w1, w2=w2, w1_bias=None, w2_bias=None)
        sliced = _slice_expert_weights(weights, (1,))
        assert sliced.w1.shape == (1, 8, 8)
        assert sliced.w1_bias is None

    def test_scatter_phase_output(self):
        full = torch.zeros(5, 2)
        phase_out = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _scatter_phase_output(full, phase_out, ((0, 1), (3, 5)))
        assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
        assert torch.equal(result[1], torch.tensor([0.0, 0.0]))
        assert torch.equal(result[2], torch.tensor([0.0, 0.0]))
        assert torch.equal(result[3], torch.tensor([3.0, 4.0]))
        assert torch.equal(result[4], torch.tensor([5.0, 6.0]))


# ---------------------------------------------------------------------------
# execute_phased_mlp — equivalence tests
# ---------------------------------------------------------------------------


class TestExecutePhasedMlpEquivalence:
    """Core D.11 equivalence: single-phase vs multi-phase must be element-wise identical."""

    def _make_mlp_input(
        self,
        num_experts: int,
        tokens_per_expert: list[int],
        hidden_size: int = 4,
        ffn_size: int = 6,
    ) -> MoEMlpComputeInput:
        total_tokens = sum(tokens_per_expert)
        hidden_states = torch.randn(total_tokens, hidden_size)

        # Build group_list type 1 (count mode)
        group_list = torch.tensor(tokens_per_expert, dtype=torch.int32)

        # Square weights for simplified equivalence testing: [N, H, H].
        w1 = torch.randn(num_experts, hidden_size, hidden_size)
        w2 = torch.randn(num_experts, hidden_size, hidden_size)

        return MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(w1=w1, w2=w2),
            quant=MoEQuantParams(quant_type=None, comm_quant_mode=None, mxfp=None),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

    def test_single_phase_equivalent_to_direct(self):
        """Single-phase path should produce the same output as direct MLP call."""
        mlp_input = self._make_mlp_input(3, [2, 3, 1])

        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        phase_plan = plan_hit_miss_phases(slices, (0, 1, 2), slot_readiness={0: True, 1: True, 2: True})

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=phase_plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )

        assert direct_out.shape == phased_out.shape
        assert torch.allclose(direct_out, phased_out)

    def test_two_phase_equivalent_to_direct(self):
        """Hit/miss split should produce element-wise identical output as direct MLP."""
        mlp_input = self._make_mlp_input(4, [2, 0, 3, 1])

        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        # Experts 0,2 hit; 1,3 miss  (expert 1 has 0 tokens, won't appear in output)
        readiness = {0: True, 1: False, 2: True, 3: False}
        phase_plan = plan_hit_miss_phases(slices, (0, 1, 2, 3), slot_readiness=readiness)

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=phase_plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )

        assert direct_out.shape == phased_out.shape
        assert torch.allclose(direct_out, phased_out)

    def test_all_miss_equivalent_to_direct(self):
        """Single miss-only phase should also be equivalent."""
        mlp_input = self._make_mlp_input(2, [3, 2])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        readiness = {0: False, 1: False}
        phase_plan = plan_hit_miss_phases(slices, (0, 1), slot_readiness=readiness)

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=phase_plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )

        assert direct_out.shape == phased_out.shape
        assert torch.allclose(direct_out, phased_out)

    def test_single_expert_equivalence(self):
        """Single expert should work."""
        mlp_input = self._make_mlp_input(1, [5])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        phase_plan = plan_hit_miss_phases(slices, (0,), slot_readiness={0: True})

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=phase_plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )

        assert torch.allclose(direct_out, phased_out)

    def test_all_zero_token_experts(self):
        """Experts with 0 tokens should not cause errors."""
        mlp_input = self._make_mlp_input(3, [0, 0, 0])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        plan = plan_hit_miss_phases(slices, (0, 1, 2), slot_readiness={0: True, 1: False, 2: True})

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )
        assert direct_out.shape == phased_out.shape

    def test_group_list_type0_equivalence(self):
        """Cumsum mode (type 0) should also work."""
        num_experts = 3
        counts = [2, 3, 1]
        total = sum(counts)
        hidden_size = 4
        hidden_states = torch.randn(total, hidden_size)

        cumsum = torch.tensor([2, 5, 6], dtype=torch.int32)  # type 0
        w1 = torch.randn(num_experts, hidden_size, hidden_size)
        w2 = torch.randn(num_experts, hidden_size, hidden_size)

        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=cumsum,
            group_list_type=0,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(w1=w1, w2=w2),
            quant=MoEQuantParams(quant_type=None, comm_quant_mode=None, mxfp=None),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        slices = compute_expert_token_slices(cumsum, 0)
        assert slices == [(0, 2), (2, 5), (5, 6)]

        readiness = {0: True, 1: False, 2: True}
        plan = plan_hit_miss_phases(slices, (0, 1, 2), slot_readiness=readiness)

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )

        assert direct_out.shape == phased_out.shape
        assert torch.allclose(direct_out, phased_out)


# ---------------------------------------------------------------------------
# Profile event
# ---------------------------------------------------------------------------


class TestPhaseSplitProfileEvent:
    def test_jsonable_output(self):
        event = PhaseSplitProfileEvent(
            name="test_event",
            layer_id=5,
            seconds=0.123,
            phase_plan_jsonable={"total_phases": 2},
        )
        data = event.to_jsonable()
        assert data["event"] == "phase_split"
        assert data["name"] == "test_event"
        assert data["layer_id"] == 5
        assert data["phase_plan"]["total_phases"] == 2

    def test_with_fail_reason(self):
        event = PhaseSplitProfileEvent(
            name="phase_split_fail_closed",
            layer_id=0,
            seconds=0.0,
            fail_reason="phase_split_requires_AllGather",
        )
        data = event.to_jsonable()
        assert data["fail_reason"] == "phase_split_requires_AllGather"
        assert "phase_plan" not in data

    def test_jsonl_write(self, tmp_path, monkeypatch):
        import vllm_ascend.envs as _envs

        profile_path = tmp_path / "profile.jsonl"
        monkeypatch.setattr(_envs, "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH", str(profile_path))

        event = PhaseSplitProfileEvent(
            name="phase_split_plan",
            layer_id=3,
            seconds=0.001,
            phase_plan_jsonable={"total_phases": 1},
        )
        _write_phase_split_profile_jsonl(event)

        assert profile_path.exists()
        lines = profile_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["name"] == "phase_split_plan"
        assert data["layer_id"] == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPhaseSplitEdgeCases:
    def test_repeated_expert_in_phase(self):
        """Repeated experts should not break slicing (dedup is caller's responsibility)."""
        slices = [(0, 3), (3, 5), (5, 8)]
        expert_ids = (0, 0, 2)  # duplicate
        readiness = {0: True, 2: True}
        # Should not raise; correctness depends on caller dedup.
        plan = plan_hit_miss_phases(slices, expert_ids, slot_readiness=readiness)
        assert plan.total_phases >= 1

    def test_missing_slice_in_phase(self):
        """Every expert must have a corresponding slice — mismatch is caller's bug."""
        slices = [(0, 3)]
        expert_ids = (0, 1)
        with pytest.raises(ValueError, match="Mismatched lengths"):
            plan_hit_miss_phases(slices, expert_ids, slot_readiness={0: True, 1: True})
