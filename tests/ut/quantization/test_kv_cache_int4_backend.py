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
# SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""NPU tests for the INT4 KV cache attention backend execution contract.

These tests exercise the real ``AscendInt4AttentionBackendImpl`` data path on
NPU hardware and assert the exact paired-head contract requested by review:

  * kv-cache shape is ``head_size // 2 + 4`` (packed 2x int4/byte + inline
    per-token-head fp32 scale) and stays fp32-aligned;
  * prefill write -> packed-cache read roundtrip reconstructs K/V within INT4
    error and dense-FIA attention matches a bf16 reference;
  * decode append writes new K/V tokens, mutating the cache without corrupting
    previously cached prefill data, and the resulting decode attention matches
    a bf16 reference over the extended sequence.

Tests are skipped when no NPU is available.
"""

import itertools

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.attention.attention_v1 import (  # noqa: E402
    AscendInt4AttentionBackendImpl,
)

_NUM_BLOCKS = 128
_BLOCK_SIZE = 16
_NUM_KV_HEADS = 4
_NUM_HEADS = 8
_HEAD_SIZE = 128
_SCALE = 1.0 / (_HEAD_SIZE**0.5)
_DTYPE = torch.bfloat16


@pytest.fixture(scope="module")
def device():
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() == 0:
        pytest.skip("NPU is not available")
    torch_npu.npu.set_device(0)
    return "npu"


class _Harness:
    def __init__(
        self,
        slot_mapping,
        num_actual_tokens,
        block_tables,
        seq_lens_list,
        attn_mask,
        attn_state,
        actual_seq_lengths_q=None,
        model_runner_type="generate",
    ):
        self.slot_mapping = slot_mapping
        self.num_actual_tokens = num_actual_tokens
        self.block_tables = block_tables
        self.seq_lens_list = seq_lens_list
        self.attn_mask = attn_mask
        self.attn_state = attn_state
        self.actual_seq_lengths_q = actual_seq_lengths_q
        self.model_runner_type = model_runner_type


def _make_backend(device):
    b = object.__new__(AscendInt4AttentionBackendImpl)
    b.head_size = _HEAD_SIZE
    b.num_heads = _NUM_HEADS
    b.num_kv_heads = _NUM_KV_HEADS
    b.scale = _SCALE
    alloc = AscendInt4AttentionBackendImpl.get_kv_cache_shape(
        _NUM_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_SIZE, "int4"
    )
    pair = torch.zeros(*alloc, dtype=torch.uint8, device=device)
    b.key_cache = pair[0]
    b.value_cache = pair[1]
    return b


def _make_mask(device):
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)


def _fia_dense(query, key, value, mask, cu_q, cu_kv, device):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=mask,
        block_table=None,
        input_layout="TND",
        block_size=_BLOCK_SIZE,
        actual_seq_lengths=cu_q,
        actual_seq_lengths_kv=cu_kv,
        num_key_value_heads=_NUM_KV_HEADS,
        num_heads=_NUM_HEADS,
        scale=_SCALE,
        sparse_mode=3,
    )
    return out.view(query.shape[0], _NUM_HEADS, _HEAD_SIZE)


def _prefill_layout(seq_lens, total_tokens, device):
    """vLLM-style block-aligned slot mapping and padded block table."""
    slot_mapping = torch.empty(total_tokens, dtype=torch.int64, device=device)
    block_tables = []
    phys_start = 0
    tok_off = 0
    for sl in seq_lens:
        nblk = (sl + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        block_tables.append(torch.arange(nblk, device=device) + phys_start)
        slot_mapping[tok_off : tok_off + sl] = phys_start * _BLOCK_SIZE + torch.arange(sl, device=device)
        tok_off += sl
        phys_start += nblk
    max_blocks = max(bt.numel() for bt in block_tables)
    block_table_pad = torch.stack(
        [torch.cat([bt, torch.zeros(max_blocks - bt.numel(), dtype=bt.dtype, device=device)]) for bt in block_tables]
    )
    return slot_mapping, block_table_pad


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def test_int4_kv_cache_shape_contract(device):
    """Packed INT4 head == head_size//2 + 4 (inline fp32 scale), fp32-aligned."""
    shape = AscendInt4AttentionBackendImpl.get_kv_cache_shape(
        _NUM_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_SIZE, "int4"
    )
    assert shape == (2, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_SIZE // 2 + 4), shape
    assert shape[-1] % 4 == 0, f"padded head must stay fp32-aligned: {shape[-1]}"


def test_prefill_write_read_roundtrip(device):
    """Prefill write -> packed read reconstructs K/V within INT4 error."""
    backend = _make_backend(device)
    mask = _make_mask(device)
    seq_lens = [37, 23, 51]
    total_tokens = sum(seq_lens)

    key = (torch.randn(total_tokens, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    value = (torch.randn(total_tokens, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    query = (torch.randn(total_tokens, _NUM_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)

    slot_mapping, block_table_pad = _prefill_layout(seq_lens, total_tokens, device)

    k_packed, v_packed, k_scale, v_scale = backend._quantize_kv_to_int4(key, value, total_tokens)
    harness = _Harness(
        slot_mapping=slot_mapping,
        num_actual_tokens=total_tokens,
        block_tables=block_table_pad,
        seq_lens_list=seq_lens,
        attn_mask=mask,
        attn_state=None,
    )
    backend._reshape_and_cache_int4(k_packed, v_packed, k_scale, v_scale, harness)

    dk, dv = backend._dequant_paged_int4_to_dense(block_table_pad, seq_lens, _DTYPE)
    assert dk.shape[0] == total_tokens, f"dequant tokens {dk.shape[0]} != {total_tokens}"
    k_cos = _cosine(dk, key)
    v_cos = _cosine(dv, value)
    assert k_cos > 0.99, f"INT4 K reconstruction cosine too low: {k_cos:.4f}"
    assert v_cos > 0.99, f"INT4 V reconstruction cosine too low: {v_cos:.4f}"

    cu = torch.tensor(list(itertools.accumulate(seq_lens)), dtype=torch.int32, device=device)
    ref = _fia_dense(query, key, value, mask, cu, cu, device)
    out = _fia_dense(query, dk.to(_DTYPE), dv.to(_DTYPE), mask, cu, cu, device)
    cos = _cosine(out, ref)
    assert cos > 0.99, f"read-path attention cosine too low: {cos:.4f}"


def test_decode_append_mutates_cache_and_keeps_prefill(device):
    """Decode writes new K/V tokens; prefill data is not corrupted."""
    backend = _make_backend(device)
    mask = _make_mask(device)
    seq_lens = [37, 23, 51]
    total_tokens = sum(seq_lens)
    bsz = len(seq_lens)

    key = (torch.randn(total_tokens, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    value = (torch.randn(total_tokens, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)

    slot_mapping, block_table_pad = _prefill_layout(seq_lens, total_tokens, device)
    k_packed, v_packed, k_scale, v_scale = backend._quantize_kv_to_int4(key, value, total_tokens)
    harness = _Harness(
        slot_mapping=slot_mapping,
        num_actual_tokens=total_tokens,
        block_tables=block_table_pad,
        seq_lens_list=seq_lens,
        attn_mask=mask,
        attn_state=None,
    )
    backend._reshape_and_cache_int4(k_packed, v_packed, k_scale, v_scale, harness)
    dk_before, _ = backend._dequant_paged_int4_to_dense(block_table_pad, seq_lens, _DTYPE)
    prefill_err = (dk_before.float() - key.float()).abs().amax().item()

    # One decoded token per sequence, appended at the next free slot.
    k_dec = (torch.randn(bsz, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    v_dec = (torch.randn(bsz, _NUM_KV_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    seq_lens_dec = [s + 1 for s in seq_lens]
    slot_mapping_dec = torch.empty(bsz, dtype=torch.int64, device=device)
    phys_start = 0
    for i, sl in enumerate(seq_lens):
        nblk = (sl + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        slot_mapping_dec[i] = phys_start * _BLOCK_SIZE + sl
        phys_start += nblk

    kp, vp, ks, vs = backend._quantize_kv_to_int4(k_dec, v_dec, bsz)
    harness_dec = _Harness(
        slot_mapping=slot_mapping_dec,
        num_actual_tokens=bsz,
        block_tables=block_table_pad,
        seq_lens_list=seq_lens_dec,
        attn_mask=mask,
        attn_state=None,
    )
    backend._reshape_and_cache_int4(kp, vp, ks, vs, harness_dec)

    # Prefill region must be unchanged by the decode write.
    dk_probe, _ = backend._dequant_paged_int4_to_dense(block_table_pad, seq_lens, _DTYPE)
    probe_err = (dk_probe.float() - key.float()).abs().amax().item()
    assert probe_err == prefill_err, "Prefill KV cache corrupted by decode write"

    # Decode attention over the extended sequence matches the bf16 reference.
    dk2, dv2 = backend._dequant_paged_int4_to_dense(block_table_pad, seq_lens_dec, _DTYPE)
    total2 = sum(seq_lens_dec)
    assert dk2.shape[0] == total2, dk2.shape
    key2 = torch.cat([key, k_dec], dim=0)
    value2 = torch.cat([value, v_dec], dim=0)
    q_dec = (torch.randn(bsz, _NUM_HEADS, _HEAD_SIZE, device=device) * 0.5).to(_DTYPE)
    cu2 = torch.tensor(list(itertools.accumulate(seq_lens_dec)), dtype=torch.int32, device=device)
    q_cu = torch.arange(1, bsz + 1, dtype=torch.int32, device=device)
    ref2 = _fia_dense(q_dec, key2, value2, mask, q_cu, cu2, device)
    out2 = _fia_dense(q_dec, dk2.to(_DTYPE), dv2.to(_DTYPE), mask, q_cu, cu2, device)
    cos2 = _cosine(out2, ref2)
    assert cos2 > 0.9, f"decode-path attention cosine too low: {cos2:.4f}"
