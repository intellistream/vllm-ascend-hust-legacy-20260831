from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

INPLACE_SPLIT_DRY_RUN = "inplace_split_dry_run"
NO_SPLIT_EXACT_GRAPH_HIT = "no_split_exact_graph_hit"
NO_SPLIT_NO_LOWER_CAPTURE_SIZE = "no_split_no_lower_capture_size"
NO_SPLIT_SECOND_EMPTY = "no_split_second_empty"
NO_SPLIT_REMAINDER_TOO_LARGE = "no_split_remainder_too_large"
NO_SPLIT_NON_UNIFORM_DECODE = "no_split_non_uniform_decode"
NO_SPLIT_INVALID_QUERY_LEN = "no_split_invalid_query_len"
NO_SPLIT_NO_CAPTURE_SIZES = "no_split_no_capture_sizes"
NO_SPLIT_ATTENTION_BACKEND_MISMATCH = "no_split_attention_backend_mismatch"
NO_SPLIT_NO_OFFSET_CAPTURE_SIZE = "no_split_no_offset_capture_size"

NO_SPLIT_OFFSET_PADDING_TOO_LARGE = "no_split_offset_padding_too_large"

NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP = "no_split_offset_graph_exceeds_start_cap"
NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE = "no_split_offset_graph_below_min_size"
NO_SPLIT_INVALID_OFFSET_MATCH_POLICY = "no_split_invalid_offset_match_policy"
NO_SPLIT_INVALID_FIRST_TOKENS_POLICY = "no_split_invalid_first_tokens_policy"

NO_SPLIT_MODE_NOT_INPLACE = "no_split_mode_not_inplace"
NO_SPLIT_PARALLEL_STREAMS_DISABLED = "no_split_parallel_streams_disabled"
NO_SPLIT_CUDAGRAPH_MODE_NOT_FULL = "no_split_cudagraph_mode_not_full"
NO_SPLIT_SPEC_DECODE_CONFLICT = "no_split_spec_decode_conflict"
NO_SPLIT_LORA_CONFLICT = "no_split_lora_conflict"
NO_SPLIT_MLA_CONFLICT = "no_split_mla_conflict"
NO_SPLIT_MROPE_CONFLICT = "no_split_mrope_conflict"
NO_SPLIT_BATCH_TOO_SMALL = "no_split_batch_too_small"

_INPLACE_SPLIT_MODES = ("inplace_serial", "inplace_parallel")


@dataclass
class SplitBatchSlice:
    request_slice: slice
    token_slice: slice
    padded_num_tokens: int = 0
    start_num_tokens: int = 0

    def __post_init__(self):
        if self.padded_num_tokens == 0:
            self.padded_num_tokens = self.num_tokens

    @property
    def num_requests(self) -> int:
        return self.request_slice.stop - self.request_slice.start

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

    @property
    def graph_num_tokens(self) -> int:
        return self.padded_num_tokens

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )



@dataclass(frozen=True)
class InplaceSplitPlan:
    split_slices: list[SplitBatchSlice]
    reason: str
    total_num_tokens: int
    padded_num_tokens_without_split: int
    first_tokens: int
    second_tokens: int
    first_reqs: int
    second_reqs: int
    lower_capture_size: int
    remainder_tokens: int
    capture_sizes_considered: list[int]
    first_tokens_policy: str
    offset_match_policy: str
    second_actual_tokens: int
    second_graph_tokens: int
    second_padding_tokens: int
    offset_capture_sizes_considered: list[int]
    offset_min_graph_tokens: int
    offset_max_graph_tokens_by_start: dict[int, int] | None
    offset_allowed_graph_tokens_by_start: dict[int, list[int]] | None



def _padded_graph_size(total_tokens: int, capture_sizes: list[int]) -> int:
    for cs in capture_sizes:
        if cs >= total_tokens:
            return cs
    return total_tokens


def _balanced_inplace_split_score(
    first_tokens: int,
    second_tokens: int,
    first_padding: int,
    second_padding: int,
) -> float:
    total_waste = first_padding + second_padding
    balance_penalty = abs(first_tokens - second_tokens) / max(first_tokens + second_tokens, 1)
    return total_waste + balance_penalty * 0.5


def create_inplace_split_batch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    total_num_tokens: int,
    uniform_decode_query_len: int,
    cudagraph_capture_sizes: list[int] | tuple[int, ...],
    inplace_max_remainder_tokens: int | None = None,
    *,
    offset_match_policy: str = "exact",
    offset_capture_sizes: list[int] | tuple[int, ...] | None = None,
    offset_min_graph_tokens: int = 1,
    offset_max_padding_tokens: int | None = None,
    offset_max_padding_ratio: float | None = None,
    offset_max_graph_tokens_by_start: dict[int, int] | None = None,
    offset_allowed_graph_tokens_by_start: dict[int, list[int]] | None = None,
    first_tokens_policy: str = "largest_lower",
) -> tuple[InplaceSplitPlan | None, str]:
    if uniform_decode_query_len < 1:
        return None, NO_SPLIT_INVALID_QUERY_LEN

    if not cudagraph_capture_sizes:
        return None, NO_SPLIT_NO_CAPTURE_SIZES

    sorted_capture_sizes = sorted(set(cudagraph_capture_sizes))
    padded_without_split = _padded_graph_size(total_num_tokens, sorted_capture_sizes)

    if padded_without_split == total_num_tokens:
        return None, NO_SPLIT_EXACT_GRAPH_HIT

    num_reqs = len(num_scheduled_tokens_per_request)
    if num_reqs == 0:
        return None, NO_SPLIT_NO_CAPTURE_SIZES

    valid_lower_sizes = [
        cs for cs in sorted_capture_sizes
        if cs < total_num_tokens and cs % uniform_decode_query_len == 0
    ]

    if not valid_lower_sizes:
        return None, NO_SPLIT_NO_LOWER_CAPTURE_SIZE

    if first_tokens_policy not in ("largest_lower", "balanced"):
        return None, NO_SPLIT_INVALID_FIRST_TOKENS_POLICY

    if offset_match_policy not in ("exact", "bucket"):
        return None, NO_SPLIT_INVALID_OFFSET_MATCH_POLICY

    normalized_offset_capture_sizes: list[int] | None = None
    if offset_capture_sizes is not None:
        normalized_offset_capture_sizes = sorted(set(offset_capture_sizes))

    best_plan: InplaceSplitPlan | None = None
    best_score: float = float("inf")

    if first_tokens_policy == "largest_lower":
        candidate_sizes = [valid_lower_sizes[-1]]
    else:
        candidate_sizes = valid_lower_sizes

    for lower_size in candidate_sizes:
        first_tokens = lower_size
        if first_tokens % uniform_decode_query_len != 0:
            continue

        first_reqs = first_tokens // uniform_decode_query_len
        if first_reqs > num_reqs:
            continue

        second_tokens = total_num_tokens - first_tokens
        if second_tokens <= 0:
            if first_tokens_policy == "largest_lower":
                return None, NO_SPLIT_SECOND_EMPTY
            continue

        second_reqs = num_reqs - first_reqs
        remainder_tokens = padded_without_split - total_num_tokens

        if inplace_max_remainder_tokens is not None and remainder_tokens > inplace_max_remainder_tokens:
            if first_tokens_policy == "largest_lower":
                return None, NO_SPLIT_REMAINDER_TOO_LARGE
            continue

        second_graph_tokens = second_tokens
        offset_sizes_considered: list[int] = []

        if offset_match_policy == "bucket":
            if normalized_offset_capture_sizes is None:
                return None, NO_SPLIT_NO_OFFSET_CAPTURE_SIZE
            for ocs in normalized_offset_capture_sizes:
                if ocs >= second_tokens:
                    second_graph_tokens = ocs
                    offset_sizes_considered = [
                        ocs2 for ocs2 in normalized_offset_capture_sizes if ocs2 >= second_tokens
                    ]
                    break
            else:
                if first_tokens_policy == "largest_lower":
                    return None, NO_SPLIT_NO_OFFSET_CAPTURE_SIZE
                continue

        if second_graph_tokens < offset_min_graph_tokens:
            if first_tokens_policy == "largest_lower":
                return None, NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE
            continue

        second_padding_tokens = second_graph_tokens - second_tokens

        if offset_max_padding_tokens is not None and second_padding_tokens > offset_max_padding_tokens:
            if first_tokens_policy == "largest_lower":
                return None, NO_SPLIT_OFFSET_PADDING_TOO_LARGE
            continue

        if offset_max_padding_ratio is not None and second_graph_tokens > 0:
            actual_ratio = second_padding_tokens / second_graph_tokens
            if actual_ratio > offset_max_padding_ratio:
                if first_tokens_policy == "largest_lower":
                    return None, NO_SPLIT_OFFSET_PADDING_TOO_LARGE
                continue

        if offset_max_graph_tokens_by_start is not None:
            max_allowed = offset_max_graph_tokens_by_start.get(first_tokens)
            if max_allowed is not None and second_graph_tokens > max_allowed:
                if first_tokens_policy == "largest_lower":
                    return None, NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP
                continue

        if offset_allowed_graph_tokens_by_start is not None:
            allowed = offset_allowed_graph_tokens_by_start.get(first_tokens)
            if allowed is not None and second_graph_tokens not in allowed:
                if first_tokens_policy == "largest_lower":
                    return None, NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP
                continue

        split_slices = [
            SplitBatchSlice(
                request_slice=slice(0, first_reqs),
                token_slice=slice(0, first_tokens),
                padded_num_tokens=first_tokens,
                start_num_tokens=0,
            ),
            SplitBatchSlice(
                request_slice=slice(first_reqs, num_reqs),
                token_slice=slice(first_tokens, total_num_tokens),
                padded_num_tokens=second_graph_tokens,
                start_num_tokens=first_tokens,
            ),
        ]

        plan = InplaceSplitPlan(
            split_slices=split_slices,
            reason=INPLACE_SPLIT_DRY_RUN,
            total_num_tokens=total_num_tokens,
            padded_num_tokens_without_split=padded_without_split,
            first_tokens=first_tokens,
            second_tokens=second_tokens,
            first_reqs=first_reqs,
            second_reqs=second_reqs,
            lower_capture_size=lower_size,
            remainder_tokens=remainder_tokens,
            capture_sizes_considered=valid_lower_sizes,
            first_tokens_policy=first_tokens_policy,
            offset_match_policy=offset_match_policy,
            second_actual_tokens=second_tokens,
            second_graph_tokens=second_graph_tokens,
            second_padding_tokens=second_padding_tokens,
            offset_capture_sizes_considered=offset_sizes_considered,
            offset_min_graph_tokens=offset_min_graph_tokens,
            offset_max_graph_tokens_by_start=offset_max_graph_tokens_by_start,
            offset_allowed_graph_tokens_by_start=offset_allowed_graph_tokens_by_start,
        )

        if first_tokens_policy == "largest_lower":
            return plan, INPLACE_SPLIT_DRY_RUN

        first_padding = first_tokens - split_slices[0].num_tokens
        score = _balanced_inplace_split_score(
            first_tokens, second_tokens, first_padding, second_padding_tokens
        )
        if score < best_score:
            best_score = score
            best_plan = plan

    if best_plan is not None:
        return best_plan, INPLACE_SPLIT_DRY_RUN

    return None, NO_SPLIT_NO_LOWER_CAPTURE_SIZE


def inplace_split_preserves_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> bool:
    """Return whether split graphs keep the unsplit attention backend.

    Some Ascend decode shapes route to paged attention while others route to
    fused infer attention. Inplace split must not silently change that routing,
    because graph capture records backend-specific task params.
    """
    unsplit_uses_pa = uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split)
    return all(
        uses_paged_attention(split_slice.graph_num_tokens) == unsplit_uses_pa
        for split_slice in inplace_split_plan.split_slices)


def inplace_split_first_graph_matches_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> bool:
    """Return whether split-0 can safely reuse its ordinary graph backend."""
    if not inplace_split_plan.split_slices:
        return False
    unsplit_uses_pa = uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split)
    first_split = inplace_split_plan.split_slices[0]
    return uses_paged_attention(first_split.graph_num_tokens) == unsplit_uses_pa


def select_inplace_attention_backend(
    inplace_split_plan: InplaceSplitPlan,
    uses_paged_attention: Callable[[int], bool],
) -> str:
    """Select attention backend for inplace split based on unsplit behavior."""
    return ("pa" if uses_paged_attention(
        inplace_split_plan.padded_num_tokens_without_split) else "fia")