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
    WaveStager,
    _build_phase_group_list,
    _extract_phase_tokens,
    _scatter_phase_output,
    _slice_expert_weights,
    _write_phase_split_profile_jsonl,
    compute_expert_token_slices,
    execute_phased_mlp,
    build_b2_wave_routing,
    build_wave_expert_map,
    plan_capacity_bounded_phases,
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


# ---------------------------------------------------------------------------
# B2: plan_capacity_bounded_phases
# ---------------------------------------------------------------------------


class TestPlanCapacityBoundedPhases:
    def test_fits_in_one_wave_single_phase(self):
        slices = [(0, 3), (3, 5), (5, 8)]
        expert_ids = (0, 1, 2)
        plan = plan_capacity_bounded_phases(slices, expert_ids, num_slots=4)
        assert plan.total_phases == 1
        assert plan.reason == "capacity_single_wave"
        assert plan.total_tokens == 8

    def test_exact_capacity_single_wave(self):
        slices = [(0, 3), (3, 5)]
        expert_ids = (0, 1)
        plan = plan_capacity_bounded_phases(slices, expert_ids, num_slots=2)
        assert plan.total_phases == 1
        assert plan.reason == "capacity_single_wave"

    def test_two_waves_even_split(self):
        slices = [(0, 1), (1, 2), (2, 3), (3, 4)]
        expert_ids = (10, 11, 12, 13)
        plan = plan_capacity_bounded_phases(slices, expert_ids, num_slots=2)
        assert plan.total_phases == 2
        assert plan.reason == "capacity_bounded_waves"
        assert plan.hit_phases == 0
        assert plan.miss_phases == 2
        assert plan.phases[0].expert_indices == (10, 11)
        assert plan.phases[1].expert_indices == (12, 13)
        assert plan.phases[0].token_slices == ((0, 1), (1, 2))
        assert plan.phases[1].token_slices == ((2, 3), (3, 4))

    def test_uneven_last_wave_remainder(self):
        # N=5, num_slots=2 -> waves of [2,2,1]
        slices = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        expert_ids = (0, 1, 2, 3, 4)
        plan = plan_capacity_bounded_phases(slices, expert_ids, num_slots=2)
        assert plan.total_phases == 3
        assert [len(p.expert_indices) for p in plan.phases] == [2, 2, 1]
        assert plan.phases[2].expert_indices == (4,)

    def test_num_slots_one_per_wave(self):
        slices = [(0, 2), (2, 4), (4, 6)]
        expert_ids = (7, 8, 9)
        plan = plan_capacity_bounded_phases(slices, expert_ids, num_slots=1)
        assert plan.total_phases == 3
        assert all(len(p.expert_indices) == 1 for p in plan.phases)

    def test_invalid_num_slots_raises(self):
        with pytest.raises(ValueError, match="num_slots must be greater than 0"):
            plan_capacity_bounded_phases([(0, 1)], (0,), num_slots=0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            plan_capacity_bounded_phases([(0, 1)], (0, 1), num_slots=2)


# ---------------------------------------------------------------------------
# B2: execute_phased_mlp with capacity waves + stage_wave_fn hook
# ---------------------------------------------------------------------------


class TestCapacityWaveExecution:
    """B2 equivalence: wave-streamed prefill must equal single-phase output, and
    the stage hook must fire once per wave with that wave's expert ids."""

    def _make_mlp_input(self, num_experts, tokens_per_expert, hidden_size=4):
        total_tokens = sum(tokens_per_expert)
        hidden_states = torch.randn(total_tokens, hidden_size)
        group_list = torch.tensor(tokens_per_expert, dtype=torch.int32)
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

    def test_waves_equivalent_to_direct(self):
        # 5 experts, num_slots=2 -> 3 waves; output must match single-phase.
        mlp_input = self._make_mlp_input(5, [2, 1, 3, 1, 2])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        plan = plan_capacity_bounded_phases(slices, (0, 1, 2, 3, 4), num_slots=2)
        assert plan.total_phases == 3

        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
        )
        assert direct_out.shape == phased_out.shape
        assert torch.allclose(direct_out, phased_out)

    def test_stage_hook_fires_per_wave(self):
        mlp_input = self._make_mlp_input(4, [1, 1, 1, 1])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        # expert_indices are positions into group_list/weights (0..N-1), the same
        # contract as plan_hit_miss_phases; the stage hook receives those positions.
        plan = plan_capacity_bounded_phases(slices, (0, 1, 2, 3), num_slots=2)

        staged: list[tuple[int, ...]] = []
        execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
            stage_wave_fn=lambda ids: staged.append(tuple(ids)),
        )
        # Two waves, staged in order with the wave's expert positions.
        assert staged == [(0, 1), (2, 3)]

    def test_stage_hook_runs_even_for_single_wave(self):
        # Single wave (fits capacity) but stage_wave_fn present -> hook still fires.
        mlp_input = self._make_mlp_input(2, [2, 3])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        plan = plan_capacity_bounded_phases(slices, (0, 1), num_slots=4)
        assert plan.total_phases == 1

        staged: list[tuple[int, ...]] = []
        direct_out = _dummy_mlp_fn(mlp_input)
        phased_out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
            stage_wave_fn=lambda ids: staged.append(tuple(ids)),
        )
        assert staged == [(0, 1)]
        assert torch.allclose(direct_out, phased_out)


# ---------------------------------------------------------------------------
# B2 keystone: wave-partition + per-wave mask + cross-wave accumulate == single
# pass over all experts. This is the MATHEMATICAL core of the independent B2
# prefill path (the per-wave dispatch+matmul+combine loop). It is dispatcher-
# independent: it models the full MoE sum directly so we can prove correctness on
# CPU before touching the live fused_experts path / NPU.
#
# MoE output for a token:  out[t] = sum_{e in topk(t)} gate(t,e) * expert_e(x[t])
# Addition is associative/commutative, so partitioning the active experts into
# disjoint waves and summing each wave's (token,expert) contributions yields the
# identical result -> wave-streamed prefill is exactly equal to single-pass.
# ---------------------------------------------------------------------------


def _moe_full_reference(x, topk_ids, topk_weights, expert_weights):
    """Single-pass MoE: out[t] = sum_j w[t,j] * (x[t] @ W[topk_ids[t,j]])."""
    T, H = x.shape
    out = torch.zeros(T, H, dtype=x.dtype)
    k = topk_ids.shape[1]
    for t in range(T):
        for j in range(k):
            e = int(topk_ids[t, j])
            out[t] += float(topk_weights[t, j]) * (x[t] @ expert_weights[e])
    return out


def _moe_wave_accumulate(x, topk_ids, topk_weights, expert_weights, num_slots):
    """Wave-streamed MoE: partition the active experts into <=num_slots waves;
    each wave sums only the (token,expert) pairs whose expert is in that wave;
    accumulate across waves. Models the independent B2 prefill semantics."""
    T, H = x.shape
    k = topk_ids.shape[1]
    active = sorted({int(e) for e in topk_ids.flatten().tolist()})
    out = torch.zeros(T, H, dtype=x.dtype)
    for wave_start in range(0, len(active), num_slots):
        wave_experts = set(active[wave_start : wave_start + num_slots])
        assert len(wave_experts) <= num_slots  # capacity invariant
        wave_out = torch.zeros(T, H, dtype=x.dtype)
        for t in range(T):
            for j in range(k):
                e = int(topk_ids[t, j])
                if e in wave_experts:  # per-wave mask
                    wave_out[t] += float(topk_weights[t, j]) * (x[t] @ expert_weights[e])
        out += wave_out  # cross-wave accumulate
    return out


class TestB2WaveAccumulateEquivalence:
    def _setup(self, T=6, H=4, E=12, k=3, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(T, H)
        expert_weights = torch.randn(E, H, H)
        # Each token picks k distinct experts; weights are arbitrary positive.
        topk_ids = torch.zeros(T, k, dtype=torch.long)
        for t in range(T):
            perm = torch.randperm(E)[:k]
            topk_ids[t] = perm
        topk_weights = torch.rand(T, k)
        return x, topk_ids, topk_weights, expert_weights

    def test_waves_equal_single_pass_basic(self):
        x, ids, w, W = self._setup(E=12, k=3)
        ref = _moe_full_reference(x, ids, w, W)
        # active union is up to 12 experts; num_slots=4 -> up to 3 waves.
        got = _moe_wave_accumulate(x, ids, w, W, num_slots=4)
        assert torch.allclose(ref, got, atol=1e-6)

    def test_waves_equal_single_pass_remainder(self):
        # active ~ up to 10, num_slots=4 -> waves [4,4,2]
        x, ids, w, W = self._setup(T=8, E=10, k=4, seed=1)
        ref = _moe_full_reference(x, ids, w, W)
        got = _moe_wave_accumulate(x, ids, w, W, num_slots=4)
        assert torch.allclose(ref, got, atol=1e-6)

    def test_waves_equal_single_pass_num_slots_one(self):
        # Extreme: one expert per wave (worst-case HBM pressure).
        x, ids, w, W = self._setup(T=5, E=8, k=2, seed=2)
        ref = _moe_full_reference(x, ids, w, W)
        got = _moe_wave_accumulate(x, ids, w, W, num_slots=1)
        assert torch.allclose(ref, got, atol=1e-6)

    def test_waves_equal_single_pass_fits_one_wave(self):
        # num_slots >= active union -> single wave, still equal.
        x, ids, w, W = self._setup(T=4, E=6, k=2, seed=3)
        ref = _moe_full_reference(x, ids, w, W)
        got = _moe_wave_accumulate(x, ids, w, W, num_slots=64)
        assert torch.allclose(ref, got, atol=1e-6)


# ---------------------------------------------------------------------------
# B2: build_wave_expert_map + dispatcher-drop accumulate equivalence
# ---------------------------------------------------------------------------


class TestBuildWaveExpertMap:
    def test_maps_wave_experts_to_slot_positions_rest_minus_one(self):
        m = build_wave_expert_map((3, 7, 1), num_logical_experts=8)
        assert m.dtype == torch.int32
        # slot positions follow wave order: 3->0, 7->1, 1->2; others -1.
        assert m.tolist() == [-1, 2, -1, 0, -1, -1, -1, 1]

    def test_full_map_when_all_experts_in_one_wave(self):
        m = build_wave_expert_map((0, 1, 2), num_logical_experts=3)
        assert m.tolist() == [0, 1, 2]

    def test_empty_wave_all_minus_one(self):
        m = build_wave_expert_map((), num_logical_experts=4)
        assert m.tolist() == [-1, -1, -1, -1]


class TestExpertMapDropAccumulateEquivalence:
    """Prove the dispatcher's drop mechanism (expert_map[topk_ids]==-1 zeroes the
    weight) accumulated across per-wave maps == full-map single pass. This models
    the live B2 path: each wave swaps in build_wave_expert_map(wave) and we sum."""

    def _moe_with_expert_map(self, x, topk_ids, topk_weights, expert_weights, expert_map):
        # Mirror token_dispatcher: weight *= (expert_map[topk_ids] != -1).
        T, H = x.shape
        k = topk_ids.shape[1]
        out = torch.zeros(T, H, dtype=x.dtype)
        for t in range(T):
            for j in range(k):
                e = int(topk_ids[t, j])
                kept = int(expert_map[e]) != -1
                if kept:
                    out[t] += float(topk_weights[t, j]) * (x[t] @ expert_weights[e])
        return out

    def test_per_wave_maps_accumulate_to_full(self):
        torch.manual_seed(11)
        T, H, E, k = 6, 4, 10, 3
        x = torch.randn(T, H)
        W = torch.randn(E, H, H)
        topk_ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
        topk_weights = torch.rand(T, k)

        # Full reference (no drops).
        full_map = torch.arange(E, dtype=torch.int32)
        ref = self._moe_with_expert_map(x, topk_ids, topk_weights, W, full_map)

        # Wave-streamed: partition active experts into waves of <=num_slots=4,
        # build a per-wave expert_map, accumulate.
        active = sorted({int(e) for e in topk_ids.flatten().tolist()})
        num_slots = 4
        acc = torch.zeros(T, H)
        for s in range(0, len(active), num_slots):
            wave = tuple(active[s : s + num_slots])
            wmap = build_wave_expert_map(wave, num_logical_experts=E)
            acc += self._moe_with_expert_map(x, topk_ids, topk_weights, W, wmap)
        assert torch.allclose(ref, acc, atol=1e-6)


# ---------------------------------------------------------------------------
# B2: build_b2_wave_routing (offload-path per-wave remap + weight mask) and the
# log2phy-style accumulate equivalence that matches the LIVE mechanism (offload
# dispatch pre-remaps topk_ids via log2phy; no auto-zero -> we zero explicitly).
# ---------------------------------------------------------------------------


class TestBuildB2WaveRouting:
    def test_minus_one_becomes_slot_zero_and_weight_zeroed(self):
        # physical ids: token0 -> [slot2, -1], token1 -> [-1, slot0]
        phys = torch.tensor([[2, -1], [-1, 0]])
        w = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        safe_ids, masked_w = build_b2_wave_routing(phys, w)
        assert safe_ids.tolist() == [[2, 0], [0, 0]]
        # weights zeroed exactly where phys == -1.
        assert torch.allclose(masked_w, torch.tensor([[0.7, 0.0], [0.0, 0.6]]))

    def test_all_in_wave_unchanged(self):
        phys = torch.tensor([[0, 1], [1, 0]])
        w = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        safe_ids, masked_w = build_b2_wave_routing(phys, w)
        assert safe_ids.tolist() == phys.tolist()
        assert torch.allclose(masked_w, w)

    def test_all_dropped_zero_weights(self):
        phys = torch.tensor([[-1, -1]])
        w = torch.tensor([[0.9, 0.1]])
        safe_ids, masked_w = build_b2_wave_routing(phys, w)
        assert safe_ids.tolist() == [[0, 0]]
        assert masked_w.tolist() == [[0.0, 0.0]]


class TestB2Log2phyAccumulateEquivalence:
    """Mirror the LIVE offload mechanism: per wave, remap logical topk_ids through
    that wave's log2phy (non-wave -> -1), apply build_b2_wave_routing, run experts
    on PHYSICAL slot weights, accumulate. Must equal the full single pass."""

    def _wave_log2phy(self, wave, num_logical):
        # logical expert -> slot position 0..k-1 for wave members, else -1.
        import torch
        m = torch.full((num_logical,), -1, dtype=torch.long)
        for slot, e in enumerate(wave):
            m[e] = slot
        return m

    def test_live_style_waves_equal_full(self):
        torch.manual_seed(5)
        T, H, E, k, num_slots = 7, 4, 11, 3, 4
        x = torch.randn(T, H)
        W = torch.randn(E, H, H)  # logical expert weights
        topk_ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
        topk_weights = torch.rand(T, k)

        # Full reference: out[t] = sum_j w[t,j] * x[t] @ W[ids[t,j]]
        ref = torch.zeros(T, H)
        for t in range(T):
            for j in range(k):
                ref[t] += float(topk_weights[t, j]) * (x[t] @ W[int(topk_ids[t, j])])

        active = sorted({int(e) for e in topk_ids.flatten().tolist()})
        acc = torch.zeros(T, H)
        for s in range(0, len(active), num_slots):
            wave = tuple(active[s : s + num_slots])
            log2phy = self._wave_log2phy(wave, E)
            # Physical slot weights for this wave: slot p holds logical wave[p].
            slot_W = torch.stack([W[e] for e in wave]) if wave else torch.zeros(0, H, H)
            phys_ids = log2phy[topk_ids]  # non-wave -> -1
            safe_ids, masked_w = build_b2_wave_routing(phys_ids, topk_weights)
            # Run experts on PHYSICAL slot weights (what the slot bank holds).
            for t in range(T):
                for j in range(k):
                    p = int(safe_ids[t, j])
                    acc[t] += float(masked_w[t, j]) * (x[t] @ slot_W[p])
        assert torch.allclose(ref, acc, atol=1e-6)


# ---------------------------------------------------------------------------
# B2 overlap-ready executor: WaveStager two-phase contract + bounded prefetch.
# Proves the software-pipelined loop (a) produces identical output to serial for
# any prefetch_depth, and (b) never issues a wave into a buffer whose prior
# occupant is still in flight (the correctness guard that makes async safe).
# ---------------------------------------------------------------------------


class _MockOverlapStager(WaveStager):
    """Models an async stager with N rotating buffers. Records the issue/wait
    interleaving so tests can assert pipeline ordering + buffer safety.

    A buffer is "occupied" from issue until the matching ``wait`` (the executor
    waits a wave right before computing + freeing it). We map wave_index -> buffer
    via round-robin (wave_index % buffer_count); if an issue targets a buffer
    still marked occupied, that's a write-before-consume hazard -> assertion fails.
    Freeing on ``wait`` is exact for this loop: the executor's order is
    wait(k) -> compute(k) -> issue(k+max_in_flight), and issue happens only after
    the prior occupant of that slot has been waited.
    """

    def __init__(self, buffer_count, events):
        self.buffer_count = buffer_count
        self.events = events
        self._occupied: dict[int, int] = {}  # buffer_slot -> wave_index in flight

    def issue(self, wave_index, expert_indices):
        slot = wave_index % self.buffer_count
        assert slot not in self._occupied, (
            f"buffer {slot} still holds wave {self._occupied.get(slot)} "
            f"when issuing wave {wave_index} -> write-before-consume hazard"
        )
        self._occupied[slot] = wave_index
        self.events.append(("issue", wave_index))

    def wait(self, wave_index):
        # The executor computes + frees this wave's buffer right after wait.
        self._occupied.pop(wave_index % self.buffer_count, None)
        self.events.append(("wait", wave_index))


class TestWaveStagerPipeline:
    def _make_mlp_input(self, num_experts, tokens_per_expert, hidden_size=4):
        total_tokens = sum(tokens_per_expert)
        torch.manual_seed(7)
        hidden_states = torch.randn(total_tokens, hidden_size)
        group_list = torch.tensor(tokens_per_expert, dtype=torch.int32)
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

    def _run(self, num_slots, prefetch_depth, buffer_count):
        mlp_input = self._make_mlp_input(6, [1, 2, 1, 3, 1, 2])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        plan = plan_capacity_bounded_phases(slices, tuple(range(6)), num_slots=num_slots)
        events: list[tuple[str, int]] = []
        stager = _MockOverlapStager(buffer_count, events)
        out = execute_phased_mlp(
            mlp_compute_input=mlp_input,
            phase_plan=plan,
            _apply_mlp_fn=_dummy_mlp_fn,
            wave_stager=stager,
            prefetch_depth=prefetch_depth,
        )
        return out, events, stager

    def test_serial_matches_direct(self):
        mlp_input = self._make_mlp_input(6, [1, 2, 1, 3, 1, 2])
        direct = _dummy_mlp_fn(mlp_input)
        out, _, _ = self._run(num_slots=2, prefetch_depth=0, buffer_count=1)
        assert torch.allclose(direct, out)

    def test_overlap_matches_serial_output(self):
        # depth=0 single buffer vs depth=1 double buffer -> identical output.
        serial, _, _ = self._run(num_slots=2, prefetch_depth=0, buffer_count=1)
        overlap, _, _ = self._run(num_slots=2, prefetch_depth=1, buffer_count=2)
        assert torch.allclose(serial, overlap)

    def test_prefetch_issues_ahead(self):
        # num_slots=2 over 6 experts -> 3 waves. depth=1,buffers=2: wave 1 must be
        # issued BEFORE wave 0 is waited (issue,issue,wait pattern at the start).
        _, events, _ = self._run(num_slots=2, prefetch_depth=1, buffer_count=2)
        # First three events: issue 0, issue 1, wait 0.
        assert events[0] == ("issue", 0)
        assert events[1] == ("issue", 1)
        assert events[2] == ("wait", 0)

    def test_serial_issues_one_at_a_time(self):
        # depth=0: strict issue,wait,(compute),issue,wait,... no look-ahead.
        _, events, _ = self._run(num_slots=2, prefetch_depth=0, buffer_count=1)
        assert events[0] == ("issue", 0)
        assert events[1] == ("wait", 0)
        # Next issue only after wave 0 consumed (i.e. its wait happened first).
        assert events[2] == ("issue", 1)

    def test_buffer_count_caps_inflight(self):
        # depth=5 (greedy) but buffer_count=2 -> never more than 2 issued ahead.
        # _MockOverlapStager.issue asserts no buffer reuse before consume; the run
        # completing without assertion is the proof. Also output must be correct.
        mlp_input = self._make_mlp_input(6, [1, 2, 1, 3, 1, 2])
        direct = _dummy_mlp_fn(mlp_input)
        out, events, _ = self._run(num_slots=1, prefetch_depth=5, buffer_count=2)
        assert torch.allclose(direct, out)
        # At most 2 outstanding issues before the first wait.
        first_wait = next(i for i, e in enumerate(events) if e[0] == "wait")
        issues_before_first_wait = sum(1 for e in events[:first_wait] if e[0] == "issue")
        assert issues_before_first_wait <= 2

    def test_invalid_both_stager_and_callback(self):
        mlp_input = self._make_mlp_input(2, [1, 1])
        slices = compute_expert_token_slices(mlp_input.group_list, mlp_input.group_list_type)
        plan = plan_capacity_bounded_phases(slices, (0, 1), num_slots=1)
        with pytest.raises(ValueError, match="at most one"):
            execute_phased_mlp(
                mlp_compute_input=mlp_input,
                phase_plan=plan,
                _apply_mlp_fn=_dummy_mlp_fn,
                stage_wave_fn=lambda ids: None,
                wave_stager=_MockOverlapStager(1, []),
            )
