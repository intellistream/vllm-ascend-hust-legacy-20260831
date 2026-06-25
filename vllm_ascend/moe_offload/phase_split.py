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

"""MVP-D.11: Post-dispatch phase split semantic prototype.

This module provides the contracts, slicing, phase planning, partial MLP
execution, and scatter/gather logic for splitting the post-dispatch MoE MLP
compute into multiple phases (hit-first, then miss).  D.11 is a **semantic
prototype**: it proves that slicing + per-phase grouped matmul + gather
produces element-wise identical results to a single-phase run.  It does NOT
introduce async transfer, performance optimisation, or changes to router /
top-k / token count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from vllm_ascend import envs

if TYPE_CHECKING:
    import torch

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEMlpComputeInput,
        MoEWeights,
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEPhase:
    """A single phase of expert MLP execution.

    Each phase covers a contiguous (after extraction) range of tokens for a
    subset of the active experts.
    """

    phase_index: int
    # Logical expert ids covered by this phase (order as they appear in the
    # original sorted-hidden-states layout).
    expert_indices: tuple[int, ...]
    # Start / end (exclusive) offsets in the *original* sorted hidden states
    # that this phase will extract.  Because experts may be interleaved with
    # experts from other phases the extracted region is not necessarily
    # contiguous in the original buffer – the executor is responsible for
    # gathering the individual expert slices.
    token_slices: tuple[tuple[int, int], ...]
    # True when all experts in this phase are resident / slot-ready (hit).
    is_hit: bool

    @property
    def total_tokens(self) -> int:
        return sum(end - start for start, end in self.token_slices)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "expert_indices": list(self.expert_indices),
            "token_slices": [[int(s), int(e)] for s, e in self.token_slices],
            "total_tokens": self.total_tokens,
            "is_hit": self.is_hit,
        }


@dataclass(frozen=True)
class MoEPhasePlan:
    """Complete phase-split plan for one MoE layer invocation."""

    phases: tuple[MoEPhase, ...]
    total_phases: int
    hit_phases: int
    miss_phases: int
    total_tokens: int
    # Optional human-readable reason for diagnostics.
    reason: str = ""

    def to_jsonable(self) -> dict[str, object]:
        return {
            "total_phases": self.total_phases,
            "hit_phases": self.hit_phases,
            "miss_phases": self.miss_phases,
            "total_tokens": self.total_tokens,
            "reason": self.reason,
            "phases": [p.to_jsonable() for p in self.phases],
        }


# ---------------------------------------------------------------------------
# Expert token slicing from group_list
# ---------------------------------------------------------------------------


def compute_expert_token_slices(
    group_list: "torch.Tensor",
    group_list_type: int,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for each expert in *group_list* order.

    *group_list_type*:
        - 1 (count):  ``group_list[i]`` = number of tokens for expert *i*.
        - 0 (cumsum): ``group_list[i]`` = cumulative end offset for expert *i*.
    """
    if group_list_type == 1:
        counts = [int(c) for c in group_list.cpu().tolist()]
    elif group_list_type == 0:
        cumsum = [int(c) for c in group_list.cpu().tolist()]
        counts = [cumsum[0]]
        for i in range(1, len(cumsum)):
            counts.append(cumsum[i] - cumsum[i - 1])
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}; D.11 supports 0 or 1 only.")

    slices: list[tuple[int, int]] = []
    offset = 0
    for count in counts:
        slices.append((offset, offset + count))
        offset += count
    return slices


# ---------------------------------------------------------------------------
# Phase planner
# ---------------------------------------------------------------------------


def _build_single_phase_plan(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    reason: str = "",
) -> MoEPhasePlan:
    """Fallback: every active expert in one phase."""
    total = sum(end - start for start, end in expert_slices)
    phase = MoEPhase(
        phase_index=0,
        expert_indices=active_expert_ids,
        token_slices=tuple(expert_slices),
        is_hit=True,
    )
    return MoEPhasePlan(
        phases=(phase,),
        total_phases=1,
        hit_phases=1,
        miss_phases=0,
        total_tokens=total,
        reason=reason,
    )


def plan_hit_miss_phases(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    slot_readiness: dict[int, bool] | None = None,
    max_phases: int = 2,
) -> MoEPhasePlan:
    """Split active experts into hit (ready) / miss phases.

    Parameters
    ----------
    expert_slices:
        Per-expert ``(start, end)`` offsets in the original sorted hidden
        states, aligned with *active_expert_ids*.
    active_expert_ids:
        Logical expert ids in the order they appear in ``group_list`` (which
        is also the order of *expert_slices*).
    slot_readiness:
        ``expert_id -> bool`` mapping.  Experts not in the map are treated as
        *ready* (hit).  Pass ``None`` to force single-phase.
    max_phases:
        Upper bound on the number of phases (default 2 = hit + miss).
    """
    if slot_readiness is None or max_phases <= 1:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="single_phase")

    if len(active_expert_ids) != len(expert_slices):
        raise ValueError(
            f"Mismatched lengths: active_expert_ids={len(active_expert_ids)}, "
            f"expert_slices={len(expert_slices)}"
        )

    hit_pairs: list[tuple[int, int]] = []  # (expert_id, slice_index)
    miss_pairs: list[tuple[int, int]] = []

    for slice_idx, expert_id in enumerate(active_expert_ids):
        if slot_readiness.get(int(expert_id), True):
            hit_pairs.append((int(expert_id), slice_idx))
        else:
            miss_pairs.append((int(expert_id), slice_idx))

    phases: list[MoEPhase] = []

    def _make_phase(idx: int, pairs: list[tuple[int, int]], is_hit: bool) -> MoEPhase | None:
        if not pairs:
            return None
        return MoEPhase(
            phase_index=idx,
            expert_indices=tuple(eid for eid, _ in pairs),
            token_slices=tuple(expert_slices[si] for _, si in pairs),
            is_hit=is_hit,
        )

    hit_phase = _make_phase(0, hit_pairs, True)
    if hit_phase is not None:
        phases.append(hit_phase)

    miss_phase = _make_phase(len(phases), miss_pairs, False)
    if miss_phase is not None:
        phases.append(miss_phase)

    if not phases:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="empty_phases_fallback")

    total_tokens = sum(end - start for start, end in expert_slices)
    return MoEPhasePlan(
        phases=tuple(phases),
        total_phases=len(phases),
        hit_phases=sum(1 for p in phases if p.is_hit),
        miss_phases=sum(1 for p in phases if not p.is_hit),
        total_tokens=total_tokens,
        reason="hit_miss_split" if miss_pairs else "all_hit",
    )


def build_b2_wave_routing(
    physical_topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
):
    """B2 per-wave routing on the OFFLOAD path (log2phy-remapped topk_ids).

    The offload dispatch path remaps ``topk_ids`` through the wave's ``log2phy``
    buffer BEFORE this point: experts staged into this wave's slots get a valid
    physical slot id (>=0); every other (non-wave) logical expert maps to -1.
    Unlike the EP path, offload dispatch does NOT auto-zero dropped experts, so
    here we make the wave self-contained:

      * ``safe_ids`` = physical_topk_ids with -1 replaced by 0 (an in-range slot)
        so ``npu_moe_init_routing`` never indexes out of bounds. Those tokens are
        routed into slot 0 but contribute nothing because...
      * ``masked_weights`` = topk_weights zeroed wherever physical id was -1, so
        the combine (unpermute by probs) adds 0 for non-wave (token,expert) pairs.

    Summing each wave's combined output reproduces the full MoE output (addition
    over disjoint expert subsets), per the wave-accumulate keystone.

    Returns ``(safe_ids, masked_weights)`` -- both same shape as the inputs.
    """
    import torch

    kept = physical_topk_ids != -1
    safe_ids = torch.where(kept, physical_topk_ids, torch.zeros_like(physical_topk_ids))
    masked_weights = topk_weights * kept.to(topk_weights.dtype)
    return safe_ids, masked_weights


def build_wave_expert_map(
    wave_logical_experts: tuple[int, ...],
    num_logical_experts: int,
) -> "torch.Tensor":
    """Build a per-wave ``expert_map`` for B2 wave-streamed prefill.

    The AllGather token dispatcher drops experts whose ``expert_map`` entry is -1
    (``mask = expert_map[topk_ids] != -1; topk_weights = topk_weights * mask``).
    For one wave we therefore map ONLY that wave's logical experts to physical slot
    positions ``0..k-1`` (their order within the wave) and map every other logical
    expert to -1. Running dispatch->matmul->combine with this map computes exactly
    that wave's ``(token, expert)`` contributions (others contribute 0); summing
    across waves reproduces the full MoE output (see the wave-accumulate keystone).

    Returns an int32 tensor of shape ``[num_logical_experts]``.
    """
    import torch

    expert_map = torch.full((int(num_logical_experts),), -1, dtype=torch.int32)
    for slot_position, logical_expert in enumerate(wave_logical_experts):
        expert_map[int(logical_expert)] = slot_position
    return expert_map


def plan_capacity_bounded_phases(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    num_slots: int,
) -> MoEPhasePlan:
    """B2: split active experts into capacity-bounded waves of <= num_slots each.

    Unlike :func:`plan_hit_miss_phases` (which splits on slot *readiness*), this
    planner splits on slot *capacity*: when an offloaded layer's active expert
    set exceeds ``num_slots`` (the fixed HBM slot budget), a single grouped
    matmul cannot run because not all experts can be resident at once. We instead
    emit ``ceil(N / num_slots)`` waves, each covering a contiguous chunk of at
    most ``num_slots`` experts. The executor stages each wave's experts into the
    slot bank (eager prefill only), runs a partial grouped matmul over just that
    wave's tokens, and scatters the result back. Because every token belongs to
    exactly one expert and every expert to exactly one wave, the per-wave scatters
    are disjoint and cover all tokens -> the concatenated result is element-wise
    identical to a single-phase run.

    Parameters mirror :func:`plan_hit_miss_phases`: ``expert_slices[i]`` is the
    ``(start, end)`` token range for ``active_expert_ids[i]`` in the sorted hidden
    states. ``num_slots`` is the per-layer fixed slot count.

    Waves are marked ``is_hit=False`` because every wave requires staging (no wave
    is resident up front under B2's capacity pressure).
    """
    if num_slots <= 0:
        raise ValueError(f"num_slots must be greater than 0, got {num_slots}")
    if len(active_expert_ids) != len(expert_slices):
        raise ValueError(
            f"Mismatched lengths: active_expert_ids={len(active_expert_ids)}, "
            f"expert_slices={len(expert_slices)}"
        )

    total_tokens = sum(end - start for start, end in expert_slices)

    # Fits in one wave -> degenerate to a single phase (same as B1's slot path).
    if len(active_expert_ids) <= num_slots:
        return _build_single_phase_plan(
            expert_slices, active_expert_ids, reason="capacity_single_wave"
        )

    phases: list[MoEPhase] = []
    for wave_index, start in enumerate(range(0, len(active_expert_ids), num_slots)):
        chunk_ids = active_expert_ids[start : start + num_slots]
        chunk_slices = tuple(expert_slices[start : start + num_slots])
        phases.append(
            MoEPhase(
                phase_index=wave_index,
                expert_indices=tuple(int(e) for e in chunk_ids),
                token_slices=chunk_slices,
                is_hit=False,
            )
        )

    return MoEPhasePlan(
        phases=tuple(phases),
        total_phases=len(phases),
        hit_phases=0,
        miss_phases=len(phases),
        total_tokens=total_tokens,
        reason="capacity_bounded_waves",
    )


# ---------------------------------------------------------------------------
# Partial MLP execution helpers
# ---------------------------------------------------------------------------


def _extract_phase_tokens(
    hidden_states: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Extract and concatenate tokens for a phase from sorted hidden states."""
    import torch

    if not token_slices:
        return torch.empty(0, hidden_states.size(1), dtype=hidden_states.dtype, device=hidden_states.device)

    chunks = [hidden_states[start:end] for start, end in token_slices]
    if len(chunks) == 1:
        return chunks[0].contiguous()
    return torch.cat(chunks, dim=0).contiguous()


def _build_phase_group_list(
    group_list: "torch.Tensor",
    group_list_type: int,
    expert_indices: tuple[int, ...],
) -> "torch.Tensor":
    """Build a new group_list tensor covering only *expert_indices*."""
    import torch

    if group_list_type == 1:
        selected = [group_list[i] for i in expert_indices]
        return torch.stack(selected) if selected else torch.empty(0, dtype=group_list.dtype, device=group_list.device)
    elif group_list_type == 0:
        # cumulative mode: need to reconstruct per-expert counts
        cumsum = group_list.cpu().tolist()
        prev = 0
        counts = []
        for i, end in enumerate(cumsum):
            counts.append(end - prev)
            prev = end
        selected = [counts[i] for i in expert_indices]
        result = torch.tensor(selected, dtype=group_list.dtype, device=group_list.device)
        # Convert back to cumulative
        return torch.cumsum(result, dim=0)
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}")


def _slice_expert_weights(
    weights: "MoEWeights",
    expert_indices: tuple[int, ...],
) -> "MoEWeights":
    """Return a MoEWeights view sliced to *expert_indices*."""
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEWeights as _MoEWeights

    def _index(w):
        if w is None:
            return None
        if isinstance(w, list):
            return [t[list(expert_indices)] for t in w]
        return w[list(expert_indices)]

    return _MoEWeights(
        w1=_index(weights.w1),
        w2=_index(weights.w2),
        w1_bias=_index(weights.w1_bias),
        w2_bias=_index(weights.w2_bias),
        w1_scale=_index(weights.w1_scale),
        w2_scale=_index(weights.w2_scale),
        w1_scale_bias=_index(weights.w1_scale_bias),
        w2_scale_bias=_index(weights.w2_scale_bias),
        w1_offset=_index(weights.w1_offset),
        w2_offset=_index(weights.w2_offset),
    )


# ---------------------------------------------------------------------------
# Scatter / gather
# ---------------------------------------------------------------------------


def _scatter_phase_output(
    full_output: "torch.Tensor",
    phase_output: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Write *phase_output* back into *full_output* at the given slices.

    Returns *full_output* (modified in-place).
    """
    offset = 0
    for start, end in token_slices:
        length = end - start
        if length > 0:
            full_output[start:end] = phase_output[offset : offset + length]
            offset += length
    return full_output


# ---------------------------------------------------------------------------
# Top-level phased MLP orchestrator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wave staging interface (overlap-ready: separates transfer from compute)
# ---------------------------------------------------------------------------


class WaveStager:
    """Two-phase staging contract for capacity-bounded waves.

    The contract deliberately splits "move this wave's experts into HBM" into two
    calls so the executor can pipeline transfer (MTE) against compute (Cube):

      * ``issue(wave_index, expert_indices)`` -- start staging the wave's experts
        into a fixed slot buffer. May be asynchronous (return before the H2D copy
        finishes); the serial implementation does it synchronously.
      * ``wait(wave_index)`` -- block until that wave's slots are READY to be read
        by the grouped matmul. The serial implementation is a no-op (issue already
        finished the copy).

    A serial (``prefetch_depth=0``) run calls ``issue`` then ``wait`` then compute
    for each wave in turn -> identical to the original single-buffer staging. An
    overlapped run (``prefetch_depth>=1``, double/N-buffered) issues wave k+1
    BEFORE computing wave k, so the next wave's H2D rides under the current wave's
    matmul. ``buffer_count`` declares how many waves may be in flight; the executor
    guarantees it never issues a wave into a buffer whose prior occupant has not
    yet been consumed (so a single-buffer stager is never asked to overlap).

    This base class is the serial reference. NPU async (separate transfer stream +
    SetFlag/WaitFlag) is a drop-in subclass that overrides issue/wait -- no change
    to the executor or the planner.
    """

    #: How many waves may be resident concurrently. 1 == single buffer (serial).
    buffer_count: int = 1

    def issue(self, wave_index: int, expert_indices: tuple[int, ...]) -> None:
        raise NotImplementedError

    def wait(self, wave_index: int) -> None:
        raise NotImplementedError


class _CallbackWaveStager(WaveStager):
    """Serial stager adapting the simple ``stage_wave_fn(expert_indices)`` callback.

    ``issue`` runs the callback synchronously (the H2D completes before it
    returns); ``wait`` is a no-op. ``buffer_count=1`` so the executor keeps strict
    serial order -- preserving the original ``stage_wave_fn`` semantics exactly.
    """

    buffer_count = 1

    def __init__(self, stage_wave_fn):
        self._stage_wave_fn = stage_wave_fn

    def issue(self, wave_index: int, expert_indices: tuple[int, ...]) -> None:
        self._stage_wave_fn(expert_indices)

    def wait(self, wave_index: int) -> None:
        return None


def execute_phased_mlp(
    *,
    mlp_compute_input: "MoEMlpComputeInput",
    phase_plan: MoEPhasePlan,
    _apply_mlp_fn=None,
    stage_wave_fn=None,
    wave_stager=None,
    prefetch_depth: int = 0,
) -> "torch.Tensor":
    """Execute MoE MLP in phases according to *phase_plan*.

    This replaces a single ``_apply_mlp(mlp_compute_input)`` call with one
    ``_apply_mlp`` call per phase, each operating on a contiguous subset of
    tokens / experts, then scatters the results back into a full output buffer.

    Parameters
    ----------
    mlp_compute_input:
        The full (single-phase) MLP compute input.
    phase_plan:
        Pre-computed phase split plan.
    _apply_mlp_fn:
        Callable ``(MoEMlpComputeInput) -> Tensor``.  Defaults to
        ``unified_apply_mlp``.
    stage_wave_fn:
        Optional callable ``(expert_indices: tuple[int, ...]) -> None`` invoked
        before each non-empty phase's MLP, used by B2 capacity-bounded waves to
        stage that wave's experts into the fixed slot bank (eager prefill only).
        ``None`` (default) preserves the original behavior: weights are assumed
        already resident. Wrapped in a serial ``_CallbackWaveStager``. Mutually
        exclusive with ``wave_stager``.
    wave_stager:
        Optional :class:`WaveStager` giving the overlap-ready two-phase
        (issue/wait) staging contract. Lets a subclass run transfer on a separate
        stream so wave k+1's H2D overlaps wave k's matmul. ``buffer_count`` caps
        in-flight waves.
    prefetch_depth:
        How many waves to issue ahead of the one being computed. ``0`` (default)
        == serial (issue, wait, compute per wave). ``>=1`` enables software
        pipelining; clamped so issued-ahead waves never exceed the stager's
        ``buffer_count`` (no buffer is reused before its wave is consumed).

    Returns
    -------
    Tensor with the same shape / layout as a single-phase ``_apply_mlp`` call.
    """
    import torch

    if stage_wave_fn is not None and wave_stager is not None:
        raise ValueError("pass at most one of stage_wave_fn / wave_stager")
    if wave_stager is None and stage_wave_fn is not None:
        wave_stager = _CallbackWaveStager(stage_wave_fn)

    if _apply_mlp_fn is None:
        from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp as _default

        _apply_mlp_fn = _default

    # Single phase → fast-path (no slicing overhead). Skipped when a stager is
    # present: B2 needs the per-wave stage call to run even for a lone wave.
    if phase_plan.total_phases == 1 and wave_stager is None:
        return _apply_mlp_fn(mlp_compute_input=mlp_compute_input)

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput as _MoEMlpComputeInput

    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    hidden_size = hidden_states.size(-1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    full_output = torch.empty(
        phase_plan.total_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )

    # Only non-empty waves participate (0-token waves are pure no-ops).
    waves = [p for p in phase_plan.phases if p.total_tokens > 0]

    def _compute_wave(phase) -> None:
        phase_hidden = _extract_phase_tokens(hidden_states, phase.token_slices)
        phase_group_list = _build_phase_group_list(group_list, group_list_type, phase.expert_indices)
        phase_weights = _slice_expert_weights(mlp_compute_input.weights, phase.expert_indices)
        phase_input = _MoEMlpComputeInput(
            hidden_states=phase_hidden,
            group_list=phase_group_list,
            group_list_type=group_list_type,
            dynamic_scale=mlp_compute_input.dynamic_scale,
            topk_scales=mlp_compute_input.topk_scales,
            weights=phase_weights,
            quant=mlp_compute_input.quant,
            fusion=mlp_compute_input.fusion,
            activation=mlp_compute_input.activation,
            need_trans=mlp_compute_input.need_trans,
            dynamic_eplb=mlp_compute_input.dynamic_eplb,
        )
        phase_output = _apply_mlp_fn(mlp_compute_input=phase_input)
        _scatter_phase_output(full_output, phase_output, phase.token_slices)

    if wave_stager is None:
        # No staging contract: plain per-wave compute (weights already resident).
        for phase in waves:
            _compute_wave(phase)
        return full_output

    # Software-pipelined staging. ``ahead`` = how many waves are issued but not yet
    # computed; capped by both prefetch_depth and the stager's buffer_count so a
    # buffer is never reused while its wave is still in flight (correctness guard
    # that lets a single-buffer/serial stager stay strictly serial).
    max_in_flight = min(max(prefetch_depth, 0) + 1, max(wave_stager.buffer_count, 1))
    issued = 0
    # Prime: issue the first ``max_in_flight`` waves.
    while issued < min(max_in_flight, len(waves)):
        wave_stager.issue(issued, waves[issued].expert_indices)
        issued += 1
    for compute_idx in range(len(waves)):
        wave_stager.wait(compute_idx)
        _compute_wave(waves[compute_idx])
        # After consuming this wave's buffer, issue the next not-yet-issued wave.
        if issued < len(waves):
            wave_stager.issue(issued, waves[issued].expert_indices)
            issued += 1

    return full_output


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSplitProfileEvent:
    name: str
    layer_id: int
    seconds: float
    phase_plan_jsonable: dict[str, object] | None = None
    fail_reason: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {
            "event": "phase_split",
            "name": self.name,
            "layer_id": self.layer_id,
            "seconds": round(self.seconds, 6),
        }
        if self.phase_plan_jsonable is not None:
            data["phase_plan"] = self.phase_plan_jsonable
        if self.fail_reason is not None:
            data["fail_reason"] = self.fail_reason
        return data


def _write_phase_split_profile_jsonl(event: PhaseSplitProfileEvent) -> None:
    profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
    if not profile_path:
        return
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_jsonable(), sort_keys=True) + "\n")
