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
"""Pure-logic unit tests for the DUAL_PAD split planner.

These tests run on CPU (no NPU required): they only exercise
``dual_pad_utils.create_dual_pad_split_batch_slices`` and
``dual_pad_utils.dual_pad_precheck_reason``.
"""

import numpy as np
import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.dual_pad_utils import (
    NO_SPLIT_DP_BATCH_TOO_SMALL,
    NO_SPLIT_DP_CUDAGRAPH_MODE_NOT_FULL,
    NO_SPLIT_DP_EXACT_GRAPH_HIT,
    NO_SPLIT_DP_EXCEEDS_MAX_SIZE,
    NO_SPLIT_DP_FORCE_SPLIT_SMALLEST,
    NO_SPLIT_DP_LORA_CONFLICT,
    NO_SPLIT_DP_MODE_DISABLED,
    NO_SPLIT_DP_MLA_CONFLICT,
    NO_SPLIT_DP_MROPE_CONFLICT,
    NO_SPLIT_DP_NON_UNIFORM_DECODE,
    NO_SPLIT_DP_NO_CAPTURE_SIZES,
    NO_SPLIT_DP_NO_MAIN_CAPTURE,
    NO_SPLIT_DP_NO_PARALLEL_CAPTURE,
    NO_SPLIT_DP_SPEC_DECODE_CONFLICT,
    NO_SPLIT_DP_THRESHOLD,
    create_dual_pad_split_batch_slices,
    dual_pad_precheck_reason,
)
from vllm_ascend.worker.inplace_split_utils import INPLACE_SPLIT_DRY_RUN

SIZES = [4, 8]


def _tokens(num_reqs: int, query_len: int = 1) -> np.ndarray:
    return np.full(num_reqs, query_len, dtype=np.int32)


class _FakeSplitConfig:

    def __init__(self, *, enabled=True, mode="dual_pad", min_batch_size_for_split=4):
        self.enabled = enabled
        self.mode = mode
        self.min_batch_size_for_split = min_batch_size_for_split


def test_exact_graph_hit_no_split():
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(8), 8, SIZES)
    assert plan is None
    assert reason == NO_SPLIT_DP_EXACT_GRAPH_HIT


def test_force_split_exact_hit_splits_second_largest():
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(8), 8, SIZES, force_split=True)
    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 4
    assert plan.second_tokens == 4
    assert plan.second_graph_tokens == 4
    assert [s.num_tokens for s in plan.split_slices] == [4, 4]
    assert [s.graph_num_tokens for s in plan.split_slices] == [4, 4]
    assert plan.split_slices[1].start_num_tokens == 4


def test_threshold_not_met_no_split():
    # total=7, sizes=[4,8]: no-split pads to 8 (1 pad), split pads remainder
    # 3->4 (1 pad): padding_saved == 0 <= threshold 0 -> no split.
    plan, reason = create_dual_pad_split_batch_slices(_tokens(7), 7, SIZES)
    assert plan is None
    assert reason == NO_SPLIT_DP_THRESHOLD


def test_threshold_met_splits():
    # Same batch, but negative threshold makes 0 saved > -1 -> split.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(7), 7, SIZES, cudagraph_split_pad_threshold=-1)
    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 4
    assert plan.second_tokens == 3
    assert plan.second_graph_tokens == 4
    assert plan.second_padding_tokens == 1
    assert plan.split_slices[1].padded_num_tokens == 4
    assert plan.split_slices[0].padded_num_tokens == 4


def test_threshold_keeps_no_split_when_saving_below():
    # threshold=1 >= saved 0 -> still no split.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(7), 7, SIZES, cudagraph_split_pad_threshold=1)
    assert plan is None
    assert reason == NO_SPLIT_DP_THRESHOLD


def test_force_split_bypasses_threshold():
    # saved == 0, but force_split splits anyway.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(7), 7, SIZES, force_split=True)
    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert [s.num_tokens for s in plan.split_slices] == [4, 3]
    assert plan.split_slices[1].graph_num_tokens == 4


def test_exceeds_max_size_no_split():
    # total=10 > max main size 8: nothing to pad to without split.
    plan, reason = create_dual_pad_split_batch_slices(_tokens(10), 10, SIZES)
    assert plan is None
    assert reason == NO_SPLIT_DP_EXCEEDS_MAX_SIZE


def test_no_main_capture():
    # total=2 smaller than every capture size.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(2), 2, SIZES, force_split=True)
    assert plan is None
    assert reason == NO_SPLIT_DP_NO_MAIN_CAPTURE


def test_no_capture_sizes():
    plan, reason = create_dual_pad_split_batch_slices(_tokens(8), 8, [])
    assert plan is None
    assert reason == NO_SPLIT_DP_NO_CAPTURE_SIZES


def test_no_parallel_capture():
    # parallel pool too small for the remainder -> unsafe to split.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(7), 7, SIZES, parallel_capture_sizes=[2],
        cudagraph_split_pad_threshold=-1)
    assert plan is None
    assert reason == NO_SPLIT_DP_NO_PARALLEL_CAPTURE


def test_force_split_smallest_graph():
    # total == smallest graph size; force_split has no smaller main graph.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(4), 4, SIZES, force_split=True)
    assert plan is None
    assert reason == NO_SPLIT_DP_FORCE_SPLIT_SMALLEST


def test_parallel_capture_sizes_used_for_remainder():
    # total=7, main=[4,8], parallel=[2,4,8]: remainder 3 pads to 4.
    plan, reason = create_dual_pad_split_batch_slices(
        _tokens(7), 7, SIZES, parallel_capture_sizes=[2, 4, 8],
        cudagraph_split_pad_threshold=-1)
    assert plan is not None
    assert plan.second_graph_tokens == 4
    assert plan.offset_capture_sizes_considered == [4, 8]


def test_precheck_ok():
    cfg = _FakeSplitConfig()
    assert dual_pad_precheck_reason(
        split_batch_config=cfg,
        cudagraph_mode=CUDAGraphMode.FULL,
        num_reqs=8,
        has_lora=False,
        is_mla=False,
        is_mrope=False,
        spec_decode_enabled=False,
    ) is None


def test_precheck_mode_disabled():
    cfg = _FakeSplitConfig(enabled=False)
    assert dual_pad_precheck_reason(
        split_batch_config=cfg,
        cudagraph_mode=CUDAGraphMode.FULL,
        num_reqs=8,
        has_lora=False,
        is_mla=False,
        is_mrope=False,
        spec_decode_enabled=False,
    ) == NO_SPLIT_DP_MODE_DISABLED


def test_precheck_non_uniform():
    cfg = _FakeSplitConfig()
    assert dual_pad_precheck_reason(
        split_batch_config=cfg,
        cudagraph_mode=CUDAGraphMode.FULL,
        num_reqs=8,
        has_lora=False,
        is_mla=False,
        is_mrope=False,
        spec_decode_enabled=False,
        uniform_decode=False,
    ) == NO_SPLIT_DP_NON_UNIFORM_DECODE


@pytest.mark.parametrize(
    "cudagraph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_precheck_mode_requires_full(cudagraph_mode):
    cfg = _FakeSplitConfig()
    reason = dual_pad_precheck_reason(
        split_batch_config=cfg,
        cudagraph_mode=cudagraph_mode,
        num_reqs=8,
        has_lora=False,
        is_mla=False,
        is_mrope=False,
        spec_decode_enabled=False,
    )
    if cudagraph_mode == CUDAGraphMode.NONE:
        assert reason == NO_SPLIT_DP_CUDAGRAPH_MODE_NOT_FULL
    else:
        assert reason is None


def test_precheck_conflicts():
    cfg = _FakeSplitConfig()
    assert dual_pad_precheck_reason(
        split_batch_config=cfg, cudagraph_mode=CUDAGraphMode.FULL, num_reqs=8,
        has_lora=True, is_mla=False, is_mrope=False,
        spec_decode_enabled=False) == NO_SPLIT_DP_LORA_CONFLICT
    assert dual_pad_precheck_reason(
        split_batch_config=cfg, cudagraph_mode=CUDAGraphMode.FULL, num_reqs=8,
        has_lora=False, is_mla=True, is_mrope=False,
        spec_decode_enabled=False) == NO_SPLIT_DP_MLA_CONFLICT
    assert dual_pad_precheck_reason(
        split_batch_config=cfg, cudagraph_mode=CUDAGraphMode.FULL, num_reqs=8,
        has_lora=False, is_mla=False, is_mrope=True,
        spec_decode_enabled=False) == NO_SPLIT_DP_MROPE_CONFLICT
    assert dual_pad_precheck_reason(
        split_batch_config=cfg, cudagraph_mode=CUDAGraphMode.FULL, num_reqs=8,
        has_lora=False, is_mla=False, is_mrope=False,
        spec_decode_enabled=True) == NO_SPLIT_DP_SPEC_DECODE_CONFLICT


def test_precheck_batch_too_small():
    cfg = _FakeSplitConfig()
    assert dual_pad_precheck_reason(
        split_batch_config=cfg, cudagraph_mode=CUDAGraphMode.FULL, num_reqs=2,
        has_lora=False, is_mla=False, is_mrope=False,
        spec_decode_enabled=False) == NO_SPLIT_DP_BATCH_TOO_SMALL
