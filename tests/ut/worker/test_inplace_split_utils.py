# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np

from vllm_ascend.worker.inplace_split_utils import (
    INPLACE_SPLIT_DRY_RUN,
    NO_SPLIT_EXACT_GRAPH_HIT,
    NO_SPLIT_INVALID_QUERY_LEN,
    NO_SPLIT_OFFSET_PADDING_TOO_LARGE,
    create_inplace_split_batch_slices,
    inplace_split_preserves_attention_backend,
    select_inplace_attention_backend,
)


def test_create_inplace_split_batch_slices_largest_lower():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(80, dtype=np.int32),
        total_num_tokens=80,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[8, 16, 32, 64, 128],
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 64
    assert plan.second_tokens == 16
    assert plan.padded_num_tokens_without_split == 128
    assert plan.split_slices[0].request_slice == slice(0, 64)
    assert plan.split_slices[1].request_slice == slice(64, 80)


def test_create_inplace_split_batch_slices_buckets_offset_graph():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(80, dtype=np.int32),
        total_num_tokens=80,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[64, 128],
        offset_match_policy="bucket",
        offset_capture_sizes=[32, 64],
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.second_actual_tokens == 16
    assert plan.second_graph_tokens == 32
    assert plan.second_padding_tokens == 16
    assert plan.offset_capture_sizes_considered == [32, 64]


def test_create_inplace_split_batch_slices_rejects_excess_offset_padding():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(80, dtype=np.int32),
        total_num_tokens=80,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[64, 128],
        offset_match_policy="bucket",
        offset_capture_sizes=[32],
        offset_max_padding_tokens=8,
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_PADDING_TOO_LARGE


def test_create_inplace_split_batch_slices_rejects_invalid_query_len():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(4, dtype=np.int32),
        total_num_tokens=4,
        uniform_decode_query_len=0,
        cudagraph_capture_sizes=[4, 8],
    )

    assert plan is None
    assert reason == NO_SPLIT_INVALID_QUERY_LEN


def test_create_inplace_split_batch_slices_skips_exact_graph_hit():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(64, dtype=np.int32),
        total_num_tokens=64,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[32, 64, 128],
    )

    assert plan is None
    assert reason == NO_SPLIT_EXACT_GRAPH_HIT


def test_inplace_split_attention_backend_guard():
    plan, _ = create_inplace_split_batch_slices(
        np.ones(80, dtype=np.int32),
        total_num_tokens=80,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[64, 128],
    )
    assert plan is not None

    uses_paged_attention = lambda shape: shape >= 128
    assert not inplace_split_preserves_attention_backend(plan, uses_paged_attention)
    assert select_inplace_attention_backend(plan, uses_paged_attention) == "pa"
