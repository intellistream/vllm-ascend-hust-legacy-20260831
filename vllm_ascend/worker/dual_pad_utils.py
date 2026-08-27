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

"""DUAL_PAD (dual-pad) split planning.

Dual-pad is the reference implementation's split-batch flavor: the two splits
run on separate padded input buffers (the second split uses the dedicated
``*_parallel_streams`` buffers) and each split dispatches to an already
captured graph size through the standard (``start_num_tokens=0``) dispatcher
path.  Unlike dual-inplace, no offset graphs or lazy captures are involved.

This module ports the reference decision logic
(``min-zbw-parallel/vllm_ascend/worker/model_runner_v3.py`` L932-1052):

- main slice = largest main-stream capture size <= total tokens
- remainder goes to the parallel stream, padded to its own nearest graph
- split only when ``padding_saved > cudagraph_split_pad_threshold``
  (unless ``force_split``); ``force_split`` also splits on an exact graph hit
  using the second-largest capture size.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.worker.inplace_split_utils import (
    INPLACE_SPLIT_DRY_RUN,
    UNIFIED_SINGLE_GRAPH_REASON,
    InplaceSplitPlan,
    SplitBatchSlice,
)


def make_unified_whole_batch_plan(plan: InplaceSplitPlan) -> InplaceSplitPlan:
    """Convert a would-be dual_pad plan into a unified whole-batch plan.

    The unified execution keeps ONE main-pool graph captured at the exact
    batch token count (the joint continuous row buffer): the union of the two
    contiguous dual_pad request slices is simply ``[0, num_reqs)``, and the
    single slice's padded size equals the exact token count (no padding tail,
    no parallel-pool graphs, no dual-stream replay).

    Args:
        plan: the committed dual_pad plan (non-None).

    Returns:
        A single-slice ``InplaceSplitPlan`` with ``use_unified=True``.
    """
    total = int(plan.total_num_tokens)
    num_reqs = int(plan.first_reqs) + int(plan.second_reqs)
    single = SplitBatchSlice(
        request_slice=slice(0, num_reqs),
        token_slice=slice(0, total),
        padded_num_tokens=total,
        start_num_tokens=0,
    )
    return InplaceSplitPlan(
        split_slices=[single],
        reason=UNIFIED_SINGLE_GRAPH_REASON,
        total_num_tokens=total,
        padded_num_tokens_without_split=total,
        first_tokens=total,
        second_tokens=0,
        first_reqs=num_reqs,
        second_reqs=0,
        lower_capture_size=total,
        remainder_tokens=0,
        capture_sizes_considered=list(plan.capture_sizes_considered),
        first_tokens_policy="unified_exact",
        offset_match_policy="none",
        second_actual_tokens=0,
        second_graph_tokens=0,
        second_padding_tokens=0,
        offset_capture_sizes_considered=[],
        offset_min_graph_tokens=1,
        offset_max_graph_tokens_by_start=None,
        offset_allowed_graph_tokens_by_start=None,
        use_unified=True,
    )


def make_dual_pad_parallel_batch_descriptor(
    num_tokens: int,
    *,
    has_lora: bool = False,
    num_active_loras: int = 0,
    uniform_decode_query_len: int = 1,
    max_num_seqs: int | None = None,
) -> BatchDescriptor:
    """Exact-size BatchDescriptor for dual-pad parallel-pool graphs.

    Parallel-pool graphs are keyed by the padded parallel size itself.  They
    must not go through the main-pool cudagraph padding: the main dispatcher
    would round the size up to the nearest main capture size, which mis-keys
    the captured graph (runtime lookup misses) and breaks dummy metadata
    construction at capture time (num_reqs vs padded num_reqs mismatch).
    Mirrors CudagraphDispatcher._create_padded_batch_descriptor's uniform
    FULL-decode num_reqs formula, minus the padding lookup.
    """
    if num_tokens <= 0:
        raise ValueError(
            "num_tokens must be positive for dual-pad parallel-pool graphs")
    if uniform_decode_query_len <= 0:
        raise ValueError("uniform_decode_query_len must be positive")
    if num_tokens % uniform_decode_query_len != 0:
        raise ValueError(
            "num_tokens must be divisible by uniform_decode_query_len for "
            "dual-pad parallel-pool graphs")
    num_reqs = num_tokens // uniform_decode_query_len
    if max_num_seqs is not None:
        num_reqs = min(num_reqs, max_num_seqs)
    return BatchDescriptor(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform=True,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )

# --- dual-pad no-split reasons ----------------------------------------------
NO_SPLIT_DP_MODE_DISABLED = "no_split_dual_pad_mode_disabled"
NO_SPLIT_DP_NON_UNIFORM_DECODE = "no_split_dual_pad_non_uniform_decode"
NO_SPLIT_DP_CUDAGRAPH_MODE_NOT_FULL = "no_split_dual_pad_cudagraph_mode_not_full"
NO_SPLIT_DP_SPEC_DECODE_CONFLICT = "no_split_dual_pad_spec_decode_conflict"
NO_SPLIT_DP_LORA_CONFLICT = "no_split_dual_pad_lora_conflict"
NO_SPLIT_DP_MLA_CONFLICT = "no_split_dual_pad_mla_conflict"
NO_SPLIT_DP_MROPE_CONFLICT = "no_split_dual_pad_mrope_conflict"
NO_SPLIT_DP_BATCH_TOO_SMALL = "no_split_dual_pad_batch_too_small"
NO_SPLIT_DP_NO_CAPTURE_SIZES = "no_split_dual_pad_no_capture_sizes"
NO_SPLIT_DP_EXACT_GRAPH_HIT = "no_split_dual_pad_exact_graph_hit"
NO_SPLIT_DP_FORCE_SPLIT_SMALLEST = "no_split_dual_pad_force_split_smallest"
NO_SPLIT_DP_EXCEEDS_MAX_SIZE = "no_split_dual_pad_exceeds_max_size"
NO_SPLIT_DP_THRESHOLD = "no_split_dual_pad_threshold_not_met"
NO_SPLIT_DP_NO_MAIN_CAPTURE = "no_split_dual_pad_no_main_capture"
NO_SPLIT_DP_NO_PARALLEL_CAPTURE = "no_split_dual_pad_no_parallel_capture"


def _ceil_to_graph(n: int, sizes: list[int]) -> int:
    """Smallest capture size >= n, or -1 when n exceeds all sizes."""
    for s in sizes:
        if s >= n:
            return s
    return -1


def dual_pad_precheck_reason(
    *,
    split_batch_config: Any,
    cudagraph_mode: CUDAGraphMode,
    num_reqs: int,
    has_lora: bool,
    is_mla: bool,
    is_mrope: bool,
    spec_decode_enabled: bool,
    uniform_decode: bool = True,
) -> Optional[str]:
    """Return the first NO_SPLIT_* reason that blocks dual-pad, or None."""
    if getattr(split_batch_config, "mode", "") != "dual_pad":
        return NO_SPLIT_DP_MODE_DISABLED
    if not split_batch_config.enabled:
        return NO_SPLIT_DP_MODE_DISABLED
    if not uniform_decode:
        return NO_SPLIT_DP_NON_UNIFORM_DECODE
    if cudagraph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
        return NO_SPLIT_DP_CUDAGRAPH_MODE_NOT_FULL
    if spec_decode_enabled:
        return NO_SPLIT_DP_SPEC_DECODE_CONFLICT
    if has_lora:
        return NO_SPLIT_DP_LORA_CONFLICT
    if is_mla:
        return NO_SPLIT_DP_MLA_CONFLICT
    if is_mrope:
        return NO_SPLIT_DP_MROPE_CONFLICT
    if num_reqs < split_batch_config.min_batch_size_for_split:
        return NO_SPLIT_DP_BATCH_TOO_SMALL
    return None


def create_dual_pad_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    cudagraph_capture_sizes: list[int] | tuple[int, ...],
    parallel_capture_sizes: list[int] | tuple[int, ...] | None = None,
    cudagraph_split_pad_threshold: int = 0,
    force_split: bool = False,
) -> tuple[Optional[InplaceSplitPlan], str]:
    """Plan a dual-pad split (largest main graph hit + padded remainder).

    Args:
        num_scheduled_tokens_per_request: token count per request (uniform
            decode only; used for request-slice bookkeeping).
        total_num_tokens: total token count to split.
        cudagraph_capture_sizes: main-stream graph sizes.
        parallel_capture_sizes: parallel-stream graph sizes; None reuses the
            main sizes (reference behaviour).
        cudagraph_split_pad_threshold: minimum padding saved (without split
            vs with split) required to actually split.
        force_split: skip the threshold and split whenever possible; on an
            exact graph hit, split into the second-largest graph + remainder.

    Returns:
        (plan, reason).  plan is None when the batch should not be split.
    """
    if not cudagraph_capture_sizes:
        return None, NO_SPLIT_DP_NO_CAPTURE_SIZES

    sorted_main = sorted(set(int(s) for s in cudagraph_capture_sizes))
    max_main_size = sorted_main[-1]

    if parallel_capture_sizes is not None:
        sorted_parallel = sorted(set(int(s) for s in parallel_capture_sizes))
    else:
        sorted_parallel = sorted_main

    num_reqs = int(len(num_scheduled_tokens_per_request))
    total = int(total_num_tokens)
    if num_reqs == 0 or total == 0:
        return None, NO_SPLIT_DP_NO_CAPTURE_SIZES

    # largest main-stream graph size that fits within total without padding
    main_tokens = max(
        (s for s in sorted_main if s <= total), default=0)
    second_tokens = total - main_tokens

    if main_tokens == total:
        # Exact graph hit: normally no split.
        if force_split and len(sorted_main) >= 2:
            candidates = [s for s in sorted_main if s < total]
            if not candidates:
                # total == smallest graph size; cannot split.
                return None, NO_SPLIT_DP_FORCE_SPLIT_SMALLEST
            main_tokens = max(candidates)
            second_tokens = total - main_tokens
        else:
            return None, NO_SPLIT_DP_EXACT_GRAPH_HIT
    elif main_tokens <= 0:
        # total smaller than every captured graph size.
        return None, NO_SPLIT_DP_NO_MAIN_CAPTURE
    elif second_tokens > 0:
        if force_split:
            pass
        elif total > max_main_size:
            # No graph to pad to without split; no split benefit.
            return None, NO_SPLIT_DP_EXCEEDS_MAX_SIZE
        else:
            original_padded = _ceil_to_graph(total, sorted_main)
            original_padding = original_padded - total
            remainder_padded = _ceil_to_graph(second_tokens, sorted_parallel)
            if remainder_padded < 0:
                # Parallel pool has no size >= remainder; unsafe to split.
                return None, NO_SPLIT_DP_NO_PARALLEL_CAPTURE
            remainder_padding = remainder_padded - second_tokens
            padding_saved = original_padding - remainder_padding
            if padding_saved <= cudagraph_split_pad_threshold:
                return None, NO_SPLIT_DP_THRESHOLD
    else:
        return None, NO_SPLIT_DP_NO_MAIN_CAPTURE

    # At this point the split is committed: main_tokens in [1, total),
    # second_tokens == total - main_tokens > 0.
    main_reqs = min(num_reqs, main_tokens)
    second_reqs = num_reqs - main_reqs
    if second_reqs <= 0:
        # Request slice would be empty (non-uniform token geometry); the
        # reference implementation only supports uniform decode (1 token/req).
        return None, NO_SPLIT_DP_NO_MAIN_CAPTURE
    second_padded = _ceil_to_graph(second_tokens, sorted_parallel)
    if second_padded < 0:
        return None, NO_SPLIT_DP_NO_PARALLEL_CAPTURE

    # main_tokens is itself a capture size -> zero padding on the main slice.
    main_padded = main_tokens

    split_slices = [
        SplitBatchSlice(
            request_slice=slice(0, main_reqs),
            token_slice=slice(0, main_tokens),
            padded_num_tokens=main_padded,
            start_num_tokens=0,
        ),
        SplitBatchSlice(
            request_slice=slice(main_reqs, num_reqs),
            token_slice=slice(main_tokens, total),
            padded_num_tokens=second_padded,
            start_num_tokens=main_tokens,
        ),
    ]

    padded_without_split = _ceil_to_graph(total, sorted_main)
    if padded_without_split < 0:
        padded_without_split = total

    offset_sizes_considered = [
        s for s in sorted_parallel if s >= second_tokens
    ]

    plan = InplaceSplitPlan(
        split_slices=split_slices,
        reason=INPLACE_SPLIT_DRY_RUN,
        total_num_tokens=total,
        padded_num_tokens_without_split=padded_without_split,
        first_tokens=main_tokens,
        second_tokens=second_tokens,
        first_reqs=main_reqs,
        second_reqs=second_reqs,
        lower_capture_size=main_padded,
        remainder_tokens=padded_without_split - total,
        capture_sizes_considered=sorted_main,
        first_tokens_policy="largest_lower",
        offset_match_policy="bucket",
        second_actual_tokens=second_tokens,
        second_graph_tokens=second_padded,
        second_padding_tokens=second_padded - second_tokens,
        offset_capture_sizes_considered=offset_sizes_considered,
        offset_min_graph_tokens=1,
        offset_max_graph_tokens_by_start=None,
        offset_allowed_graph_tokens_by_start=None,
    )
    return plan, INPLACE_SPLIT_DRY_RUN
