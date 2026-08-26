# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Host-only unit tests for plan-0826d C1 (FIA shared seq_lens list).

C1 derives the templated FIA ``seq_lens`` / ``actual_seq_lengths_q`` ONCE per
step and reuses the same host list for every layer, guarded by a per-layer
uniformity probe that falls back to per-layer derivation on any disagreement.

These tests exercise the pure host-side helpers only (no NPU). The on-device
guards and the env-gated off/on behavior are covered by the offline smoke + mode
internal consistency runs described in plan-0826d step Step4.
"""

import torch

from vllm_ascend.compilation.acl_graph_split_batch import (
    _fia_layer_seq_lens,
    _fia_shared_seq_lens_probe,
    _get_fia_key_t,
    maybe_template_fia_seq_lens,
    should_template_fia_seq_lens,
)


class _RuntimeMetadata:
    def __init__(self, metadata_mode="template", backend_tag="fia"):
        self.metadata_mode = metadata_mode
        self.backend_tag = backend_tag


class _BatchDescriptor:
    def __init__(self, metadata_mode="template", backend_tag="fia"):
        self.runtime_metadata = _RuntimeMetadata(metadata_mode, backend_tag)


class _ForwardContext:
    def __init__(self, *, metadata_mode="template", backend_tag="fia",
                 actual=8, graph=16):
        self.batch_descriptor = _BatchDescriptor(metadata_mode, backend_tag)
        self.split_actual_num_tokens = actual
        self.split_graph_num_tokens = graph


class _AttnMeta:
    def __init__(self, seq_lens_list, actual_seq_lengths_q):
        self.seq_lens_list = seq_lens_list
        self.actual_seq_lengths_q = actual_seq_lengths_q


def _template_ctx(actual=8, graph=16, **kwargs):
    return _ForwardContext(actual=actual, graph=graph, **kwargs)


# --------------------------------------------------------------------------- #
# should_template / maybe_template / _get_fia_key_t                            #
# --------------------------------------------------------------------------- #


def test_should_template_true_for_fia_template_actual_lt_graph():
    ctx = _template_ctx()
    assert should_template_fia_seq_lens(ctx) is True


def test_should_template_false_for_non_fia_backend():
    assert should_template_fia_seq_lens(_template_ctx(backend_tag="pa")) is False


def test_should_template_false_for_non_template_mode():
    assert should_template_fia_seq_lens(_template_ctx(metadata_mode="eager")) is False


def test_should_template_false_when_actual_ge_graph():
    # Zero-padding split: the last seq_lens entry belongs to a real request.
    assert should_template_fia_seq_lens(_template_ctx(actual=16, graph=16)) is False


def test_maybe_template_replaces_last_entry_only():
    ctx = _template_ctx()
    out = maybe_template_fia_seq_lens(ctx, [1, 1, 1], 64)
    assert out == [1, 1, 64]


def test_maybe_template_non_template_returns_input_identity():
    ctx = _template_ctx(backend_tag="pa")
    seq = [1, 1, 1]
    assert maybe_template_fia_seq_lens(ctx, seq, 64) is seq


def test_maybe_template_empty_list_returns_as_is():
    ctx = _template_ctx()
    assert maybe_template_fia_seq_lens(ctx, [], 64) == []


def test_get_fia_key_t_uses_ndim2_tensor_col():
    kc = torch.zeros(4, 64)
    assert _get_fia_key_t(kc, 64) == 64


def test_get_fia_key_t_fallback_for_non_tensor():
    assert _get_fia_key_t(123, 64) == 64


# --------------------------------------------------------------------------- #
# _fia_shared_seq_lens_probe: one derivation == per-layer derivation           #
# --------------------------------------------------------------------------- #


def test_probe_uniform_returns_shared_value_equal_to_per_layer():
    ctx = _template_ctx()
    kc = torch.zeros(4, 64)
    metas = [_AttnMeta([8, 8, 8], [16, 16, 16]) for _ in range(3)]
    layer_items = [(m, kc, 64) for m in metas]

    seq0, act0, uniform = _fia_shared_seq_lens_probe(ctx, layer_items)

    assert uniform is True
    assert seq0 == [8, 8, 64]
    assert act0 == [16, 16, 16]
    # One derivation == per-layer derivation for every layer.
    for m in metas:
        assert _fia_layer_seq_lens(ctx, m, kc, 64, source="acl_graph_update:x") == seq0
        assert m.actual_seq_lengths_q == act0


def test_probe_non_uniform_actual_seq_lengths_q_falls_back():
    ctx = _template_ctx()
    kc = torch.zeros(4, 64)
    metas = [
        _AttnMeta([8, 8, 8], [16, 16, 16]),
        _AttnMeta([8, 8, 8], [17, 17, 17]),
    ]
    seq0, act0, uniform = _fia_shared_seq_lens_probe(
        ctx, [(m, kc, 64) for m in metas])
    assert uniform is False


def test_probe_non_uniform_seq_lens_list_falls_back():
    ctx = _template_ctx()
    kc = torch.zeros(4, 64)
    metas = [
        _AttnMeta([8, 8, 8], [16, 16, 16]),
        _AttnMeta([8, 9, 8], [16, 16, 16]),  # non-last element differs
    ]
    seq0, act0, uniform = _fia_shared_seq_lens_probe(
        ctx, [(m, kc, 64) for m in metas])
    assert uniform is False


def test_probe_empty_returns_none_none_false():
    seq0, act0, uniform = _fia_shared_seq_lens_probe(_template_ctx(), [])
    assert seq0 is None
    assert act0 is None
    assert uniform is False


def test_probe_non_template_keeps_original_list():
    ctx = _template_ctx(backend_tag="pa")
    kc = torch.zeros(4, 64)
    metas = [_AttnMeta([8, 8, 8], [16, 16, 16]) for _ in range(2)]
    seq0, act0, uniform = _fia_shared_seq_lens_probe(
        ctx, [(m, kc, 64) for m in metas])
    assert uniform is True
    assert seq0 == [8, 8, 8]


# --------------------------------------------------------------------------- #
# env-gated off switch regression (host-level contract)                        #
# --------------------------------------------------------------------------- #


def test_shared_value_matches_off_path_per_layer_derivation():
    """When the env switch is off the code uses the original per-layer path;
    the shared value must be byte-for-byte equivalent to the per-layer value in
    the uniform case, so toggling the gate is a behavior no-op."""
    ctx = _template_ctx()
    kc = torch.zeros(4, 64)
    metas = [_AttnMeta([8, 8, 8], [16, 16, 16]) for _ in range(4)]
    seq0, _act0, uniform = _fia_shared_seq_lens_probe(
        ctx, [(m, kc, 64) for m in metas])
    assert uniform is True
    for m in metas:
        per_layer = maybe_template_fia_seq_lens(
            ctx, m.seq_lens_list, _get_fia_key_t(kc, 64),
            source="acl_graph_update:x")
        assert per_layer == seq0
