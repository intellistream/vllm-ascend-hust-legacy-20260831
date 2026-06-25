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


def test_is_static_residency_regime_predicate():
    # Regime A: num_slots >= num_logical_experts (static mapping).
    runtime_a, _ = _make_runtime(num_slots=8, num_experts=4)
    assert runtime_a.is_static_residency_regime(4) is True
    assert runtime_a.is_static_residency_regime(8) is True
    # Boundary: equal counts is still Regime A (every expert owns a slot).
    runtime_eq, _ = _make_runtime(num_slots=4, num_experts=4)
    assert runtime_eq.is_static_residency_regime(4) is True
    # Regime B: num_slots < num_logical_experts (data-dependent mapping).
    runtime_b, _ = _make_runtime(num_slots=4, num_experts=8)
    assert runtime_b.is_static_residency_regime(8) is False


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


# --- Regime A staging hook (stage_full_residency_slot_plan) -----------------
# This is the missing wire that makes the captured graph token-correct: with
# num_slots >= num_logical_experts, all experts get a fixed slot and the log2phy
# mapping is static, so we can stage once at load time before capture.


def test_full_residency_hook_fills_all_experts_regime_a():
    # num_slots == num_experts -> Regime A (everything fits, no eviction).
    runtime, num_experts = _make_runtime(num_slots=4, num_experts=4)
    buf = runtime.log2phy_buffer(0)
    addr_before = buf.data_ptr()
    # before staging: pure -1 sentinel (would mis-route the captured graph).
    assert torch.equal(buf, torch.full((num_experts,), -1, dtype=torch.int32))

    staged = runtime.stage_full_residency_slot_plan(layer_id=0)

    assert staged is True
    log2phy = runtime.log2phy_buffer(0)
    # in-place: persistent address unchanged -> still capture-replayable.
    assert log2phy.data_ptr() == addr_before
    # every logical expert now maps to a real slot (no -1 left).
    assert int(log2phy.min()) >= 0
    assert log2phy.numel() == num_experts
    # capture-safe weights expose the now-valid persistent buffer.
    capture = runtime.capture_safe_slot_weights(layer_id=0)
    assert capture is not None
    assert capture.log2phy.data_ptr() == addr_before
    assert int(capture.log2phy.min()) >= 0


def test_full_residency_hook_noop_when_graph_compat_off():
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(enabled=True, trace_only=False, num_slots=4, graph_compatible_offload=False)
    )
    runtime.register_layer_for_fixed_slots(_mock_layer(layer_id=0, num_experts=4), slot_device=torch.device("cpu"))
    staged = runtime.stage_full_residency_slot_plan(layer_id=0)
    assert staged is False
    # buffer stays at the -1 init (no staging performed).
    assert int(runtime.log2phy_buffer(0).max()) == -1


def test_full_residency_hook_noop_when_capturing(monkeypatch):
    runtime, _ = _make_runtime(num_slots=4, num_experts=4)
    import vllm_ascend.moe_offload.runtime as rt

    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: True)
    # safe no-op during capture (must have staged eager earlier); does NOT raise.
    staged = runtime.stage_full_residency_slot_plan(layer_id=0)
    assert staged is False
    assert int(runtime.log2phy_buffer(0).max()) == -1


def test_full_residency_hook_noop_for_unregistered_layer():
    runtime, _ = _make_runtime(num_slots=4, num_experts=4)
    assert runtime.stage_full_residency_slot_plan(layer_id=99) is False


def test_full_residency_hook_fail_closed_when_slots_insufficient():
    # num_slots < num_experts is NOT Regime A; the underlying working-set guard
    # must reject it rather than silently mis-stage.
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    with pytest.raises(RuntimeError, match="exceeds num_slots"):
        runtime.stage_full_residency_slot_plan(layer_id=0)


# ---------------------------------------------------------------------------
# Regime B path ①: vllm::moe_offload_stage splitting-op seam.
# These prove the op's eager behavior CPU-side (registration, capture no-op,
# pass-through, data-dependency clone, active-set staging). Wiring into
# fused_moe.apply + splitting_ops and NPU capture validation are phase 2.
# ---------------------------------------------------------------------------
def test_moe_offload_stage_op_registered():
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op  # noqa: F401

    assert hasattr(torch.ops.vllm, "moe_offload_stage")


def test_moe_offload_stage_op_pass_through_unregistered_layer(monkeypatch):
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    # Regime B (num_slots < num_experts) so we reach the registration check rather
    # than the Regime-A no-op. Layer 99 is not registered -> transparent no-op.
    runtime, _ = _make_runtime(num_slots=4, num_experts=8)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    topk = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_before = topk.clone()
    # Side-effect-only op: returns None and leaves topk_ids untouched (threaded
    # through to moe_mlp at a stable address).
    out = op_mod._moe_offload_stage_impl(topk, layer_id=99, num_logical_experts=8)
    assert out is None
    assert torch.equal(topk, topk_before)  # values unchanged (no staging ran)


def test_moe_offload_stage_op_noop_during_capture(monkeypatch):
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    runtime, _ = _make_runtime(num_slots=4, num_experts=4)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: True)

    topk = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    out = op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=4)
    # capture path performs NO host sync / no staging; buffer stays at sentinel.
    assert out is None
    assert int(runtime.log2phy_buffer(0).max()) == -1


def test_moe_offload_stage_op_stages_active_set(monkeypatch):
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    # Regime B (num_slots < num_experts): the per-step seam owns staging, so the
    # op DOES derive log2phy from the current active set. (In Regime A the seam is
    # a no-op -- see test_moe_offload_stage_op_noop_in_regime_a.)
    runtime, _ = _make_runtime(num_slots=4, num_experts=8)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    buf_before = runtime.log2phy_buffer(0)
    addr_before = buf_before.data_ptr()
    assert int(buf_before.max()) == -1  # sentinel before staging

    topk = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    topk_before = topk.clone()
    out = op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=8)

    buf_after = runtime.log2phy_buffer(0)
    # In-place stable address (graph can capture against it).
    assert buf_after.data_ptr() == addr_before
    # Active experts {1,2,3} now mapped to real slots (>= 0).
    for e in (1, 2, 3):
        assert int(buf_after[e]) >= 0
    # Side-effect-only op: returns None, topk_ids threaded through unchanged.
    assert out is None
    assert torch.equal(topk, topk_before)


def test_moe_offload_stage_op_noop_in_regime_a(monkeypatch):
    """Regime A (num_slots >= num_experts): the static log2phy mapping is filled
    once before capture by stage_full_residency_slot_plan; the per-step seam op
    MUST NOT restage (doing so would reset inactive experts to -1, making the
    captured gather read slot[-1] -> MTE out-of-range). The op is a transparent
    no-op that leaves the pre-staged buffer untouched."""
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    runtime, _ = _make_runtime(num_slots=4, num_experts=4)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    # Emulate the load-time static fill: all experts mapped.
    assert runtime.stage_full_residency_slot_plan(layer_id=0) is True
    buf_before = runtime.log2phy_buffer(0).clone()
    assert int(buf_before.min()) >= 0  # every expert mapped (no -1)

    # A single-step active subset that, if (wrongly) staged, would reset the
    # other experts to -1.
    topk = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_before = topk.clone()
    out = op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=4)

    buf_after = runtime.log2phy_buffer(0)
    # Static mapping preserved verbatim -- the seam did not touch it.
    assert torch.equal(buf_after, buf_before)
    assert int(buf_after.min()) >= 0
    # Side-effect-only op: returns None, topk_ids threaded through unchanged.
    assert out is None
    assert torch.equal(topk, topk_before)


def test_moe_offload_stage_op_fail_closed_when_active_exceeds_slots(monkeypatch):
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    # num_slots=2 < distinct active experts (3) -> underlying guard must reject
    # rather than silently mis-stage.
    runtime, _ = _make_runtime(num_slots=2, num_experts=4)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    topk = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="exceeds num_slots"):
        op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=4)


def _make_layered_runtime(num_slots: int = 4, num_experts: int = 8):
    """Regime B runtime with layered_runtime enabled (decode-fanout staging)."""
    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            trace_only=False,
            num_slots=num_slots,
            graph_compatible_offload=True,
            layered_runtime=True,
        )
    )
    runtime.register_layer_for_fixed_slots(
        _mock_layer(layer_id=0, num_experts=num_experts), slot_device=torch.device("cpu")
    )
    return runtime, num_experts


def test_moe_offload_stage_op_b1_stages_when_call_fanout_fits_slots(monkeypatch):
    """Regime B1: a single call whose distinct active set fits the slot budget is
    staged into the fixed slots (works for both prefill-union and decode calls as
    long as that one call's fanout <= num_slots). There is NO full-weight bypass
    for an offloaded layer -- its original w13/w2 live on CPU, so every call must
    go through the slot bank."""
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    # num_slots=6 >= this call's 4 distinct experts {1,2,3,4} < n(8): B1, fits.
    runtime, _ = _make_layered_runtime(num_slots=6, num_experts=8)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    addr_before = runtime.log2phy_buffer(0).data_ptr()
    topk = torch.tensor([[1, 2, 3], [4, 1, 2]], dtype=torch.int32)
    out = op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=8)

    buf_after = runtime.log2phy_buffer(0)
    assert buf_after.data_ptr() == addr_before  # stable address
    for e in (1, 2, 3, 4):
        assert int(buf_after[e]) >= 0  # active experts mapped to real slots
    assert out is None


def test_moe_offload_stage_op_b2_fail_closed_when_call_fanout_exceeds_slots(monkeypatch):
    """Regime B2: a single call whose distinct active set exceeds num_slots must
    fail closed with the actionable working-set guard message (not a downstream
    device/MTE error). This is the working set that needs wave-streamed prefill
    (a separate feature); the seam must surface it clearly rather than route it to
    a non-existent full-weight path."""
    import vllm_ascend.ops.fused_moe.moe_offload_stage_op as op_mod
    import vllm_ascend.moe_offload.runtime as rt

    runtime, _ = _make_layered_runtime(num_slots=2, num_experts=8)
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(rt, "_is_current_graph_capturing", lambda: False)

    # 3 distinct active experts {1,2,3} > num_slots(2): fail-closed.
    topk = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="exceeds num_slots"):
        op_mod._moe_offload_stage_impl(topk, layer_id=0, num_logical_experts=8)


# ---------------------------------------------------------------------------
# Option B piece 1: vllm::moe_router op (decompose the opaque moe_forward).
# These prove the router op is a FAITHFUL wrapper of the apply-path
# select_experts call: every arg forwarded 1:1, custom_routing_function pinned
# None, and bit-equivalent (topk_weights, topk_ids) on the native path. Wiring
# into AscendMoERunner.forward + NPU capture validation are phase 2 (V-B/V-C).
# ---------------------------------------------------------------------------
def test_moe_router_op_registered():
    import vllm_ascend.ops.fused_moe.moe_router_op  # noqa: F401

    assert hasattr(torch.ops.vllm, "moe_router")


def test_moe_router_op_forwards_every_arg_1to1(monkeypatch):
    """The op must forward each argument to select_experts under the SAME
    keyword the apply-path call uses, with custom_routing_function pinned None.
    A drift here would silently change router semantics."""
    import vllm_ascend.ops.fused_moe.experts_selector as es
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        num_tokens = kwargs["hidden_states"].shape[0]
        k = kwargs["top_k"]
        return (
            torch.zeros((num_tokens, k), dtype=torch.float32),
            torch.zeros((num_tokens, k), dtype=torch.int32),
        )

    monkeypatch.setattr(es, "select_experts", _spy)

    hidden = torch.randn(3, 8)
    logits = torch.randn(3, 16)
    bias = torch.randn(16)
    op_mod._moe_router_impl(
        hidden_states=hidden,
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=4,
        num_expert_group=8,
        scoring_func="softmax",
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
        num_experts=16,
    )

    # Exactly the keyword set the apply-path select_experts call uses.
    assert set(captured) == {
        "hidden_states",
        "router_logits",
        "top_k",
        "use_grouped_topk",
        "renormalize",
        "topk_group",
        "num_expert_group",
        "custom_routing_function",
        "scoring_func",
        "routed_scaling_factor",
        "e_score_correction_bias",
        "num_experts",
    }
    assert captured["custom_routing_function"] is None  # first-version constraint
    assert captured["top_k"] == 2
    assert captured["use_grouped_topk"] is False
    assert captured["renormalize"] is True
    assert captured["topk_group"] == 4
    assert captured["num_expert_group"] == 8
    assert captured["scoring_func"] == "softmax"
    assert captured["routed_scaling_factor"] == 1.5
    assert captured["num_experts"] == 16
    assert torch.equal(captured["hidden_states"], hidden)
    assert torch.equal(captured["router_logits"], logits)
    assert torch.equal(captured["e_score_correction_bias"], bias)


def test_moe_router_op_bit_equivalent_to_direct_select_native(monkeypatch):
    """On the native (non-fusion) path the op output must be bit-identical to a
    direct select_experts call with the same args -- proving the only change is
    *where* the call happens, not *what* it computes."""
    import vllm_ascend.ops.fused_moe.experts_selector as es
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod

    # Force the native torch path (no torch_npu fusion op on CPU) and disable
    # any weight-prefetch side effect.
    monkeypatch.setattr(es, "check_npu_moe_gating_top_k", lambda **kw: False)
    monkeypatch.setattr(es, "get_weight_prefetch_method", lambda: None)

    torch.manual_seed(0)
    hidden = torch.randn(5, 8)
    logits = torch.randn(5, 16)

    common = dict(
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=16,
    )

    ref_w, ref_ids = es.select_experts(
        hidden_states=hidden,
        router_logits=logits,
        custom_routing_function=None,
        **common,
    )
    op_w, op_ids = op_mod._moe_router_impl(
        hidden_states=hidden,
        router_logits=logits,
        **common,
    )

    assert torch.equal(op_ids, ref_ids)
    assert torch.equal(op_w, ref_w)


def test_moe_router_fake_shapes_and_dtypes():
    """Fake impl (trace-time proxy) must match the real output shapes/dtypes so
    the captured piece downstream sees consistent metadata."""
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod

    hidden = torch.randn(7, 8)
    logits = torch.randn(7, 16)
    w, ids = op_mod._moe_router_fake(
        hidden_states=hidden,
        router_logits=logits,
        top_k=3,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=16,
    )
    assert w.shape == (7, 3)
    assert ids.shape == (7, 3)
    assert ids.dtype == torch.int32
    assert w.dtype == logits.dtype


# ---------------------------------------------------------------------------
# Option B piece 1 (P2a): vllm::moe_router_indirect -- layer-name indirection.
# Resolves the layer at runtime, reads the SAME routing scalars the apply-path
# reads from the layer, computes num_logical_experts the SAME way, and delegates
# to the tested explicit-scalar core. Proves source-faithfulness + index safety.
# ---------------------------------------------------------------------------
def _mock_routing_layer(num_experts: int = 16, top_k: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=top_k,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        custom_routing_function=None,
        n_shared_experts=0,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=num_experts, num_logical_experts=None),
    )


def test_moe_router_indirect_op_registered():
    import vllm_ascend.ops.fused_moe.moe_router_op  # noqa: F401

    assert hasattr(torch.ops.vllm, "moe_router_indirect")


def test_moe_router_indirect_reads_layer_scalars_and_matches_core(monkeypatch):
    """The indirect op must resolve the layer, read the apply-path scalars, and
    produce output bit-identical to the explicit-scalar core called with those
    same scalars + num_logical_experts computed the apply-path way."""
    import vllm_ascend.ops.fused_moe.experts_selector as es
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    monkeypatch.setattr(es, "check_npu_moe_gating_top_k", lambda **kw: False)
    monkeypatch.setattr(es, "get_weight_prefetch_method", lambda: None)

    layer = _mock_routing_layer(num_experts=16, top_k=2)
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    torch.manual_seed(1)
    hidden = torch.randn(5, 8)
    logits = torch.randn(5, 16)

    ind_w, ind_ids = op_mod._moe_router_indirect_impl(
        hidden_states=hidden, router_logits=logits, layer_name="layer.0"
    )
    # num_logical_experts: moe_config.num_logical_experts is None -> falls back to
    # num_experts - 0 - 0 = 16. Core must be called with exactly that.
    core_w, core_ids = op_mod._moe_router_impl(
        hidden_states=hidden,
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=16,
    )
    assert torch.equal(ind_ids, core_ids)
    assert torch.equal(ind_w, core_w)


def test_moe_router_indirect_rejects_custom_routing_function(monkeypatch):
    """Seam requires custom_routing_function=None; a non-None one must fail
    closed rather than silently dropping it (which would change routing)."""
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    layer = _mock_routing_layer()
    layer.custom_routing_function = lambda *a, **k: None
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    with pytest.raises(AssertionError, match="custom_routing_function"):
        op_mod._moe_router_indirect_impl(
            hidden_states=torch.randn(3, 8),
            router_logits=torch.randn(3, 16),
            layer_name="layer.0",
        )


def test_moe_router_indirect_applies_runner_gate(monkeypatch):
    """Internal-router models (Qwen3-MoE) hold the gate on the runner and pass
    router_logits == hidden_states as a placeholder. The op MUST apply the gate
    to hidden_states and route on gate(h), NOT on the placeholder. Faithful to
    _forward_impl:710. Verified by: routing on the placeholder differs from
    routing on gate(h), and the op matches the latter."""
    import vllm_ascend.ops.fused_moe.experts_selector as es
    import vllm_ascend.ops.fused_moe.moe_router_op as op_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    monkeypatch.setattr(es, "check_npu_moe_gating_top_k", lambda **kw: False)
    monkeypatch.setattr(es, "get_weight_prefetch_method", lambda: None)

    torch.manual_seed(7)
    num_experts, hidden_dim, top_k = 16, 8, 2
    gate_w = torch.randn(num_experts, hidden_dim)

    def gate(h):  # mimics ReplicatedLinear: returns (logits, bias)
        return h @ gate_w.t(), None

    layer = _mock_routing_layer(num_experts=num_experts, top_k=top_k)
    layer.runner = SimpleNamespace(gate=gate)
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    hidden = torch.randn(5, hidden_dim)
    placeholder = hidden  # qwen3 passes router_logits = hidden_states

    # Op output: should route on gate(hidden), not on the placeholder.
    op_w, op_ids = op_mod._moe_router_indirect_impl(
        hidden_states=hidden, router_logits=placeholder, layer_name="layer.0"
    )
    # Reference: explicit core called with the gated logits.
    gated_logits, _ = gate(hidden)
    ref_w, ref_ids = op_mod._moe_router_impl(
        hidden_states=hidden,
        router_logits=gated_logits,
        top_k=top_k,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=num_experts,
    )
    assert torch.equal(op_ids, ref_ids)
    assert torch.equal(op_w, ref_w)


# ---------------------------------------------------------------------------
# Option B piece 3 (P2b): vllm::moe_mlp + B1 topk-injection registry.
# Prove: (a) registry is empty/no-op by default; (b) moe_mlp sets the injection
# around _forward_impl and clears it in finally (even on error); (c) the injected
# topk is exactly what a downstream apply-path read would consume.
# ---------------------------------------------------------------------------
def test_moe_mlp_op_registered():
    import vllm_ascend.ops.fused_moe.moe_mlp_op  # noqa: F401

    assert hasattr(torch.ops.vllm, "moe_mlp")


def test_injection_registry_empty_by_default():
    from vllm_ascend.ops.fused_moe import moe_seam_inject

    # A layer id that was never injected must report absent -> apply-path takes
    # its normal select_experts branch (byte-unchanged when seam is off).
    assert not moe_seam_inject.has_injected_topk(123456)
    assert moe_seam_inject.peek_injected_topk(123456) is None


def test_injection_set_peek_clear_roundtrip():
    from vllm_ascend.ops.fused_moe import moe_seam_inject

    w = torch.randn(4, 2)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
    moe_seam_inject.set_injected_topk(7, w, ids)
    try:
        assert moe_seam_inject.has_injected_topk(7)
        got_w, got_ids = moe_seam_inject.peek_injected_topk(7)
        # Same objects (no copy) -> apply consumes the router's exact tensors.
        assert got_w is w
        assert got_ids is ids
    finally:
        moe_seam_inject.clear_injected_topk(7)
    assert not moe_seam_inject.has_injected_topk(7)


def test_moe_mlp_sets_and_clears_injection_around_forward(monkeypatch):
    """moe_mlp must stash the router topk before _forward_impl and clear it
    after, so the apply-path sees exactly the injected pair during the call and
    nothing leaks afterwards."""
    import vllm_ascend.ops.fused_moe.moe_mlp_op as mlp_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core
    from vllm_ascend.ops.fused_moe import moe_seam_inject

    w = torch.randn(2, 2)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seen = {}

    def _fake_forward_impl(layer, h, rl, sei, iid):
        # Observed mid-call: the injection must be present and equal to inputs.
        seen["present"] = moe_seam_inject.has_injected_topk(5)
        pw, pids = moe_seam_inject.peek_injected_topk(5)
        seen["w_is"] = pw is w
        seen["ids_is"] = pids is ids
        return torch.zeros_like(h)

    layer = SimpleNamespace(
        layer_id=5,
        runner=SimpleNamespace(_forward_impl=_fake_forward_impl),
    )
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    hidden = torch.randn(2, 8)
    out = mlp_mod._moe_mlp_impl(
        hidden_states=hidden,
        router_logits=torch.randn(2, 16),
        topk_weights=w,
        topk_ids=ids,
        shared_experts_input=None,
        input_ids=None,
        layer_name="layer.5",
    )
    assert seen == {"present": True, "w_is": True, "ids_is": True}
    assert out.shape == hidden.shape
    # Cleared after the call -> no leak into the next layer/step.
    assert not moe_seam_inject.has_injected_topk(5)


def test_moe_mlp_clears_injection_on_error(monkeypatch):
    """The finally must clear the injection even if _forward_impl raises,
    otherwise a stale topk would corrupt a later layer's apply-path."""
    import vllm_ascend.ops.fused_moe.moe_mlp_op as mlp_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core
    from vllm_ascend.ops.fused_moe import moe_seam_inject

    def _boom(layer, h, rl, sei, iid):
        raise RuntimeError("kernel blew up")

    layer = SimpleNamespace(
        layer_id=9, runner=SimpleNamespace(_forward_impl=_boom)
    )
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    with pytest.raises(RuntimeError, match="kernel blew up"):
        mlp_mod._moe_mlp_impl(
            hidden_states=torch.randn(2, 8),
            router_logits=torch.randn(2, 16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            shared_experts_input=None,
            input_ids=None,
            layer_name="layer.9",
        )
    assert not moe_seam_inject.has_injected_topk(9)


def test_moe_mlp_rejects_shared_experts_tuple(monkeypatch):
    """First version supports only the single-tensor (_shared_experts is None)
    path; a tuple return must fail closed rather than be silently mishandled."""
    import vllm_ascend.ops.fused_moe.moe_mlp_op as mlp_mod
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    def _tuple_forward(layer, h, rl, sei, iid):
        return (torch.zeros_like(h), torch.zeros_like(h))

    layer = SimpleNamespace(
        layer_id=3, runner=SimpleNamespace(_forward_impl=_tuple_forward)
    )
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)

    with pytest.raises(AssertionError, match="_shared_experts is None"):
        mlp_mod._moe_mlp_impl(
            hidden_states=torch.randn(2, 8),
            router_logits=torch.randn(2, 16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            shared_experts_input=None,
            input_ids=None,
            layer_name="layer.3",
        )


# ---------------------------------------------------------------------------
# Option B piece (P2c): AscendMoERunner._select_forward seam selection + guards.
# Built via __new__ (full construction needs many deps); we exercise the guard /
# selection logic directly with mocked attrs. DEFAULT-OFF: every guard failure
# must route to the base monolithic path (NOT _seam_forward_entry).
# ---------------------------------------------------------------------------
def _seam_runner(monkeypatch, *, seam_on=True, shared=None, gate=None,
                 dp=1, ep=1, tp=1, pcp=1):
    """Minimal AscendMoERunner exposing only what the config guards read."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    import vllm_ascend.moe_offload.runtime as rt

    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner._shared_experts = shared
    runner.gate = gate
    runner.layer_name = "model.layers.0.mlp.experts"
    runner.moe_config = SimpleNamespace(
        dp_size=dp, ep_size=ep, tp_size=tp, pcp_size=pcp
    )
    fake_runtime = SimpleNamespace(
        config=SimpleNamespace(offload_stage_seam=seam_on)
    )
    monkeypatch.setattr(rt, "get_moe_offload_runtime", lambda: fake_runtime)
    return runner


def test_select_forward_seam_off_uses_base(monkeypatch):
    runner = _seam_runner(monkeypatch, seam_on=False)
    entry = runner._select_forward()
    assert entry is not runner._seam_forward_entry


def test_select_forward_seam_on_single_card_selects_seam(monkeypatch):
    runner = _seam_runner(monkeypatch, seam_on=True)
    entry = runner._select_forward()
    # Bound method identity: same underlying function.
    assert entry.__func__ is runner._seam_forward_entry.__func__


def test_select_forward_falls_back_with_shared_experts(monkeypatch):
    runner = _seam_runner(monkeypatch, seam_on=True, shared=object())
    assert runner._select_forward() is not runner._seam_forward_entry


def test_select_forward_supports_runner_gate(monkeypatch):
    # Internal-router models (Qwen3-MoE always sets a runner gate) ARE supported:
    # moe_router_indirect applies the gate before select_experts. Guard must NOT
    # bail on a non-None gate (single-card, no shared experts).
    runner = _seam_runner(monkeypatch, seam_on=True, gate=object())
    assert runner._select_forward().__func__ is runner._seam_forward_entry.__func__


@pytest.mark.parametrize("dp,ep,tp,pcp", [(2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)])
def test_select_forward_falls_back_multicard(monkeypatch, dp, ep, tp, pcp):
    runner = _seam_runner(monkeypatch, seam_on=True, dp=dp, ep=ep, tp=tp, pcp=pcp)
    assert runner._select_forward() is not runner._seam_forward_entry


def test_resolve_per_layer_guards_ok_caches_layer_id_and_n(monkeypatch):
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.layer_name = "model.layers.7.mlp.experts"
    layer = SimpleNamespace(
        layer_id=7,
        custom_routing_function=None,
        multistream_overlap_gate=False,
        enable_npugraph_ex_static_kernel=False,
        zero_expert_num=0,
        zero_expert_type=None,
        n_shared_experts=0,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=128, num_logical_experts=None),
    )
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: layer)
    assert runner._resolve_seam_per_layer_guards() is True
    assert runner._seam_layer_id == 7
    assert runner._seam_num_logical_experts == 128


@pytest.mark.parametrize(
    "attr,val",
    [
        ("custom_routing_function", lambda *a, **k: None),
        ("multistream_overlap_gate", True),
        ("enable_npugraph_ex_static_kernel", True),
    ],
)
def test_resolve_per_layer_guards_fail_closed(monkeypatch, attr, val):
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as core

    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.layer_name = "model.layers.0.mlp.experts"
    base = dict(
        layer_id=0,
        custom_routing_function=None,
        multistream_overlap_gate=False,
        enable_npugraph_ex_static_kernel=False,
        zero_expert_num=0,
        zero_expert_type=None,
        n_shared_experts=0,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=128, num_logical_experts=None),
    )
    base[attr] = val
    monkeypatch.setattr(core, "get_layer_from_name", lambda name: SimpleNamespace(**base))
    assert runner._resolve_seam_per_layer_guards() is False


def test_seam_entry_caches_fallback_and_calls_moe_forward(monkeypatch):
    """When per-layer guards fail, the entry must cache the decision and route
    every call to the monolithic moe_forward op (never the three seam ops)."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.layer_name = "model.layers.0.mlp.experts"
    # Force the per-layer guard to fail -> permanent fallback.
    monkeypatch.setattr(runner, "_resolve_seam_per_layer_guards", lambda: False)

    calls = {"moe_forward": 0}

    def _fake_moe_forward(h, rl, sei, iid, ln):
        calls["moe_forward"] += 1
        return torch.zeros_like(h)

    monkeypatch.setattr(torch.ops.vllm, "moe_forward", _fake_moe_forward)

    h = torch.randn(2, 8)
    for _ in range(3):
        out = runner._seam_forward_entry(h, torch.randn(2, 16), None, None, "enc_name")
        assert out.shape == h.shape
    assert calls["moe_forward"] == 3
    assert runner._seam_active is False
