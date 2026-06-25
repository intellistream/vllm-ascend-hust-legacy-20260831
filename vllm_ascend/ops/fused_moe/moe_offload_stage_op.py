#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Graph-compatible MoE offload staging op (SEW-Offload, Regime B path ①).

This custom op is the *control-plane seam* that makes data-dependent expert
staging compatible with ACLGraph PIECEWISE capture. Registered into
``compilation_config.splitting_ops`` (like ``vllm::mla_forward``), it forces the
FX graph splitter (see vllm/compilation/backends.py ``split_graph`` /
``should_split``) to cut the captured region right here, between the router
(``select_experts``) and the grouped MLP. Splitting-op subgraphs are EXCLUDED
from compilation/capture (backends.py: ``submod_names_to_compile = [... if not
item.is_splitting_graph]``), so this op runs EAGER between two captured pieces.

Running eager is what makes the otherwise-forbidden work legal:
  - D2H read of the active expert set (``torch.unique(topk_ids).cpu()``),
  - host decision (which logical expert occupies which fixed slot),
  - synchronous H2D staging of miss experts into the fixed slot tensors,
  - in-place write of the persistent ``log2phy`` buffer (fixed address).

The op RETURNS a clone of ``topk_ids`` (rather than mutating in place / being a
void op) so the downstream grouped MLP takes a real data dependency on it. That
dependency (a) prevents torch.compile from reordering or eliminating the
side-effectful staging, and (b) guarantees piece B (grouped MLP) does not replay
until staging has finished updating the slots + log2phy buffer.

Phase 1 (this file): op registration + fake impl + eager pass-through + UT.
NOT yet wired into fused_moe.apply and NOT yet added to splitting_ops — that is
phase 2, which touches the compilation path and requires NPU capture validation.
"""

from __future__ import annotations

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op


# Env-gated diagnostics (SEW_SEAM_PROBE): count how many times the seam op runs,
# and in which mode (capturing vs eager). The decisive R3 question is whether the
# op runs EAGER at replay (=> staging reaches the persistent buffer) or got
# captured inside a piece as a no-op (=> buffer stays at -1 => mis-route).
_PROBE_CALLS = {"capturing": 0, "eager_staged": 0, "eager_passthrough": 0}


def _moe_offload_stage_impl(
    topk_ids: torch.Tensor,
    layer_id: int,
    num_logical_experts: int,
) -> None:
    """Eager staging seam. Side-effect-only: stages miss experts into fixed slots
    and writes the persistent log2phy buffer in place.

    ADDRESS-STABILITY CONTRACT (why this returns None + declares
    ``mutates_args=["topk_ids"]`` instead of returning a clone):
    this op is a *splitting boundary* between two captured ACLGraph pieces
    (``moe_router_indirect`` upstream, ``moe_mlp`` downstream). The downstream
    captured piece reads ``topk_ids`` from the FIXED address it recorded at
    capture (acl_graph.py asserts identical input addresses at replay). If this
    op returned a fresh ``topk_ids.clone()``, that clone would land at a
    DIFFERENT address on every eager replay, so the captured ``moe_mlp`` would
    read a stale/garbage capture-time buffer -> out-of-bounds expert index ->
    MTE DDR out-of-range. So we MUST thread the SAME ``topk_ids`` tensor (the
    router piece's pool-resident output) straight through to ``moe_mlp``.

    The downstream data dependency that prevents torch.compile from DCE-ing or
    reordering the (side-effectful) staging is provided by declaring
    ``mutates_args=["topk_ids"]`` -- the same pattern the canonical splitting op
    ``unified_attention_with_output`` uses (mutates ``output``, returns None).
    The staging itself does not change ``topk_ids`` *values* (it writes the
    separate log2phy buffer); the declared mutation only encodes ordering.

    Lazily imported runtime accessor keeps this op importable without a live
    offload runtime (e.g. during unit tests that only check registration).
    """
    from vllm_ascend.moe_offload.runtime import (
        _is_current_graph_capturing,
        get_moe_offload_runtime,
    )

    # During capture we must not perform host sync / conditional H2D. In the
    # canonical flow the captured graph only reads the fixed slot tensors + the
    # fixed log2phy buffer; staging is a no-op here.
    if _is_current_graph_capturing():
        if os.environ.get("SEW_SEAM_PROBE"):
            _PROBE_CALLS["capturing"] += 1
            print(
                f"SEW_SEAM branch=CAPTURING layer={int(layer_id)} "
                f"count={_PROBE_CALLS['capturing']}",
                flush=True,
            )
        return

    runtime = get_moe_offload_runtime()
    layer_id = int(layer_id)

    # Regime A (num_slots >= num_logical_experts): the log2phy mapping is STATIC
    # and was already filled for all experts before capture
    # (stage_full_residency_slot_plan, wired in fused_moe load/lazy-register).
    # Per-step staging here would overwrite that static buffer with only the
    # current active subset, resetting every inactive expert back to -1 -- so a
    # later step's expert that maps to -1 makes the captured gather read slot[-1]
    # (MTE DDR out-of-range). The seam must therefore be a transparent no-op in
    # Regime A; it owns staging only in Regime B (num_slots < n).
    if runtime.is_static_residency_regime(int(num_logical_experts)):
        if os.environ.get("SEW_SEAM_PROBE"):
            _PROBE_CALLS["eager_passthrough"] += 1
            print(
                f"SEW_SEAM branch=EAGER_PASSTHROUGH reason=regime_a layer={layer_id} "
                f"count={_PROBE_CALLS['eager_passthrough']}",
                flush=True,
            )
        return

    # Only stage for layers that are actually offloaded under fixed slots and
    # that have been registered. Everything else is a transparent pass-through.
    if not runtime.should_use_fixed_slot_plan_for_layer(layer_id):
        if os.environ.get("SEW_SEAM_PROBE"):
            _PROBE_CALLS["eager_passthrough"] += 1
            print(
                f"SEW_SEAM branch=EAGER_PASSTHROUGH reason=not_fixed_slot "
                f"layer={layer_id} count={_PROBE_CALLS['eager_passthrough']}",
                flush=True,
            )
        return
    if not runtime.is_layer_registered(layer_id):
        if os.environ.get("SEW_SEAM_PROBE"):
            _PROBE_CALLS["eager_passthrough"] += 1
            print(
                f"SEW_SEAM branch=EAGER_PASSTHROUGH reason=not_registered "
                f"layer={layer_id} count={_PROBE_CALLS['eager_passthrough']}",
                flush=True,
            )
        return

    # D2H read of the active logical expert set. This is the host decision that
    # ACLGraph cannot record — legal here only because this op runs eager.
    active = torch.unique(topk_ids).to("cpu", non_blocking=False)
    active_experts = tuple(
        int(e) for e in active.tolist() if 0 <= int(e) < int(num_logical_experts)
    )

    # B2 wave-streamed prefill deferral. When B2 is enabled and this is an eager
    # prefill call whose active union exceeds num_slots, the single-shot staging
    # below would fail-close (working-set guard). Instead the wave loop in
    # fused_experts (_run_b2_wave_prefill) owns staging+compute per wave. So the
    # seam must be a NO-OP here and NOT write log2phy: the downstream moe_mlp ->
    # fused_experts B2 early-branch re-stages each wave and consumes the router's
    # topk_ids directly (its own per-wave log2phy), never reading this buffer.
    # prefill is detected by topk_ids row count > 1 (decode == 1). Decode keeps
    # the single-wave staging below (fanout <= num_slots), so capture is unaffected.
    is_prefill = int(topk_ids.shape[0]) > 1
    if runtime.should_use_b2_wave_prefill(
        layer_id=layer_id,
        active_expert_count=len(set(active_experts)),
        is_prefill=is_prefill,
    ):
        if os.environ.get("SEW_SEAM_PROBE") or os.environ.get("SEW_B2_PROBE"):
            _PROBE_CALLS["eager_passthrough"] += 1
            print(
                f"SEW_SEAM branch=EAGER_PASSTHROUGH reason=b2_wave_defer "
                f"layer={layer_id} n_active={len(set(active_experts))} "
                f"num_slots={runtime.config.num_slots}",
                flush=True,
            )
        return

    # Regime B staging. The offloaded layer has NO NPU full-weight copy: its
    # original w13/w2 were moved to CPU at load (fused_moe.py
    # _stage_processed_weight_to_cpu_if_needed, gated on non-residency, NOT on the
    # release flag). The only NPU-resident weights are the num_slots slot buffers,
    # so EVERY call on this layer -- prefill or decode -- must run through the slot
    # bank, computing at most num_slots experts at once. There is no FULL_WEIGHT
    # fallback for an offloaded layer (decide_layered_path's FULL_WEIGHT_PATH
    # assumes NPU-resident weights, which an offloaded layer never has).
    #
    # Therefore the only valid regimes for staging here are:
    #   * this call's distinct active set <= num_slots -> stage it (works for
    #     decode always, and for prefill iff num_slots >= the prompt's per-layer
    #     active union -- "B1": num_slots >= max-call-fanout and < n).
    #   * this call's active set > num_slots -> the fixed-slot guard fail-closes
    #     ("exceeds num_slots"); that working set needs wave-streamed prefill
    #     ("B2", a separate feature). We let stage_fixed_slot_plan raise so the
    #     failure is a clear, actionable guard message rather than a downstream
    #     device/MTE error.
    # P0 latency probe (env-gated SEW_SEAM_PROBE): bracket the staging call with a
    # device sync so STAGE_MS reflects the true on-critical-path H2D cost of this
    # seam (host decision + synchronous miss-expert copy_). The synchronize is
    # diagnostic-only and never runs in the un-probed production path, so it does
    # not perturb real serving latency. n_active vs n_mapped on the same line lets
    # us attribute STAGE_MS to "how many experts had to be moved this call".
    _probe = bool(os.environ.get("SEW_SEAM_PROBE"))
    if _probe:
        import time as _time

        torch.npu.synchronize()
        _t0 = _time.perf_counter()

    runtime.stage_fixed_slot_plan(
        layer_id=layer_id,
        active_experts=active_experts,
        num_logical_experts=int(num_logical_experts),
    )
    if _probe:
        torch.npu.synchronize()
        _stage_ms = (_time.perf_counter() - _t0) * 1000.0
        _PROBE_CALLS["eager_staged"] += 1
        buf = runtime.log2phy_buffer(layer_id)
        n_mapped = None if buf is None else int((buf >= 0).sum().item())
        print(
            f"SEW_SEAM branch=EAGER_STAGED layer={layer_id} "
            f"count={_PROBE_CALLS['eager_staged']} n_active={len(active_experts)} "
            f"n_mapped={n_mapped} STAGE_MS={_stage_ms:.3f}",
            flush=True,
        )
    return


def _moe_offload_stage_fake(
    topk_ids: torch.Tensor,
    layer_id: int,
    num_logical_experts: int,
) -> None:
    # Side-effect-only op: no return value (mutates the persistent log2phy buffer
    # + slots in place; topk_ids is threaded through unchanged for address
    # stability across the captured router/mlp pieces).
    return


direct_register_custom_op(
    op_name="moe_offload_stage",
    op_func=_moe_offload_stage_impl,
    fake_impl=_moe_offload_stage_fake,
    mutates_args=["topk_ids"],
    dispatch_key="PrivateUse1",
)
