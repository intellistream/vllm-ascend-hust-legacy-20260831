# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V4
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V2
# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V1
# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_RUNTIME_V1
# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_RUNTIME_V1
# PEARL_STAGE5_NANOPEARL_BONUS_AWARE_PREFETCH_V1
# PEARL_STAGE5_NANOPEARL_EXACT_PREFIX_COMPAT_V1
# PEARL_STAGE5_NANOPEARL_FORCE_REBASE_CONTROL_V1
# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_RUNTIME_V1
#!/usr/bin/env python3
"""Small, transport-agnostic PRE/POST-VERIFY runtime for Stage-5.

This module contains the part of nano-PEARL that is independent from vLLM's
ModelRunner API.  Draft and Target keep their own KV caches.  The only object
that crosses the process boundary is a batch of token IDs; the next Draft
round is prefetched under the optimistic assumption that the current draft
will be accepted in full.

The caller supplies ``request_batch``.  It must be safe to call from one
background worker and must return one list of draft IDs for every request.
The controller deliberately serializes requests through that callback because
the current HCCL bridge has one ordered stream.  A later direct process-group
integration can replace the callback without changing the state machine.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import os
import threading
from typing import Callable, Iterable, Sequence


class PearlMode(str, Enum):
    PRE_VERIFY = "pre_verify"
    POST_VERIFY = "post_verify"


@dataclass(frozen=True)
class DraftRequest:
    request_id: str
    prefix_token_ids: tuple[int, ...]
    gamma: int

    @classmethod
    def from_mapping(cls, value: dict) -> "DraftRequest":
        request_id = str(value["request_id"])
        prefix = tuple(int(x) for x in value["prefix_token_ids"])
        gamma = int(value["gamma"])
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not prefix:
            raise ValueError(f"empty prefix for {request_id!r}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive for {request_id!r}")
        return cls(request_id, prefix, gamma)


@dataclass(frozen=True)
class DraftResult:
    request_id: str
    prefix_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class VerifyResult:
    """Compact result needed to advance the local Draft/Target state."""

    request_id: str
    accepted_len: int
    draft_len: int
    replacement_token_id: int | None = None
    finished: bool = False

    def __post_init__(self) -> None:
        if self.accepted_len < 0 or self.draft_len < 0:
            raise ValueError("accepted_len and draft_len must be non-negative")
        if self.accepted_len > self.draft_len:
            raise ValueError("accepted_len cannot exceed draft_len")

    @classmethod
    def from_mapping(cls, value: dict) -> "VerifyResult":
        return cls(
            request_id=str(value["request_id"]),
            accepted_len=int(value["accepted_len"]),
            draft_len=int(value["draft_len"]),
            replacement_token_id=(
                None
                if value.get("replacement_token_id") is None
                else int(value["replacement_token_id"])
            ),
            finished=bool(value.get("finished", False)),
        )

    @property
    def all_accepted(self) -> bool:
        return self.accepted_len == self.draft_len


BatchRequestFn = Callable[[list[dict]], list[list[int]]]
RebaseBatchFn = Callable[[list[dict]], None]
TraceFn = Callable[[str], None]
CommitBatchFn = Callable[[list[dict]], None]


@dataclass
class _PendingPrefetch:
    requests: tuple[DraftRequest, ...]
    future: Future[list[DraftResult]]


class NanoPearlPrefetchController:
    """Drive the PRE_VERIFY/POST_VERIFY transition for one target batch.

    ``get_or_request`` is called by the Target proposer with the current
    committed prefixes.  After obtaining the current draft, it immediately
    starts a background request for ``prefix + draft``.  On the next call:

    * if the new prefixes still contain those optimistic prefixes, the
      background result is consumed in POST_VERIFY;
    * otherwise a rejection occurred, the stale result is drained/discarded,
      and a fresh request is made in PRE_VERIFY.

    The prefix comparison is intentionally conservative: it only treats a
    round as post-verify when every row still has its optimistic prefix.  The
    Target remains the source of truth, so this cannot change output tokens.
    """

    def __init__(
        self,
        request_batch: BatchRequestFn,
        *,
        trace: TraceFn | None = None,
        commit_batch: CommitBatchFn | None = None,
        rebase_batch: RebaseBatchFn | None = None,
    ) -> None:
        self._request_batch = request_batch
        self._commit_batch = commit_batch
        self._length_only_enabled = (
            os.environ.get(
                "PEARL_STAGE5_NANOPEARL_LENGTH_ONLY", "0"
            ) == "1"
        )
        self._rebase_batch = rebase_batch
        self._rebase_on_discard = (
            os.environ.get(
                "PEARL_STAGE5_NANOPEARL_SAFE_DISCARD_REBASE", "1"
            )
            != "0"
        )
        # # PEARL_STAGE5_NANOPEARL_FORCE_REBASE_CONTROL_V1: diagnostic switch; default keeps existing behavior.
        self._force_rebase = os.environ.get(
            "PEARL_STAGE5_NANOPEARL_FORCE_REBASE", "0"
        ) == "1"
        self._trace_fn = trace
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pearl-post-verify",
        )
        self._lock = threading.RLock()
        self._pending: _PendingPrefetch | None = None
        self._closed = False
        self.mode = PearlMode.PRE_VERIFY
        self.round_id = 0

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def _trace(self, message: str) -> None:
        if self._trace_fn is not None:
            self._trace_fn(message)

    @staticmethod
    def _normalize(
        requests: Iterable[DraftRequest | dict],
    ) -> tuple[DraftRequest, ...]:
        normalized = tuple(
            item if isinstance(item, DraftRequest) else DraftRequest.from_mapping(item)
            for item in requests
        )
        if not normalized:
            raise ValueError("nano-pearl request batch must not be empty")
        return normalized

    @staticmethod
    @staticmethod
    def _compatible(
        expected: tuple[DraftRequest, ...],
        current: tuple[DraftRequest, ...],
    ) -> bool:
        """Return true only for the exact state used by the prefetch."""
        if len(expected) != len(current):
            return False
        for old, new in zip(expected, current):
            if old.request_id != new.request_id:
                return False
            if old.gamma != new.gamma:
                return False
            # A Target bonus token or any other prefix change invalidates the
            # Draft result computed for the optimistic prefix.
            if new.prefix_token_ids != old.prefix_token_ids:
                return False
        return True
    @staticmethod
    def _to_wire(requests: Sequence[DraftRequest]) -> list[dict]:
        return [
            {
                "request_id": request.request_id,
                "prefix_token_ids": list(request.prefix_token_ids),
                "gamma": request.gamma,
            }
            for request in requests
        ]

    def _eligible_length_only_ids(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        pending_by_id: dict[str, tuple[DraftRequest, DraftResult]],
        pending_error: Exception | None,
    ) -> set[str]:
        """Return rows whose Target prefix is already in Draft KV state."""
        if (
            not self._length_only_enabled
            or pending_error is not None
            or not pending_by_id
        ):
            return set()

        eligible: set[str] = set()
        for request in current:
            result = explicit.get(request.request_id)
            matched = pending_by_id.get(request.request_id)
            if result is None or matched is None:
                continue
            old, old_result = matched
            draft = old_result.draft_token_ids
            if result.finished or not result.all_accepted:
                continue
            if result.draft_len != len(draft):
                continue
            if request.gamma != old.gamma:
                continue
            if (
                len(request.prefix_token_ids) < len(old.prefix_token_ids)
                or request.prefix_token_ids[: len(old.prefix_token_ids)]
                != old.prefix_token_ids
            ):
                continue

            extra = request.prefix_token_ids[len(old.prefix_token_ids) :]
            if len(extra) > len(draft):
                continue
            if tuple(extra) != tuple(draft[: len(extra)]):
                continue

            # A replacement/bonus is safe only when the current Target
            # prefix already contains the same prefetched token.  If Target
            # has not appended it yet, extra is empty and no token transfer
            # is needed in this round.
            if (
                result.replacement_token_id is not None
                and extra
                and int(extra[0]) != int(result.replacement_token_id)
            ):
                continue
            eligible.add(request.request_id)

        if eligible:
            self._trace(
                "length_only_eligible "
                f"round={self.round_id} batch={len(eligible)} "
                f"rows={','.join(sorted(eligible))}"
            )
        return eligible
    def _commit_current(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        length_only_ids: set[str] | None = None,
    ) -> set[str]:
        """Commit boundaries; strict rows carry only accepted/valid lengths."""
        if self._commit_batch is None or not explicit:
            return set()

        length_only_ids = length_only_ids or set()
        updates: list[dict] = []
        committed_ids: set[str] = set()
        for request in current:
            result = explicit.get(request.request_id)
            if result is None:
                continue
            valid_len = max(0, len(request.prefix_token_ids) - 1)
            length_only = (
                self._length_only_enabled
                and request.request_id in length_only_ids
            )
            if length_only:
                # Strict length-only payload.  The receiver derives the
                # target boundary from valid_len and uses its resident Draft
                # state for all token/KV bookkeeping.
                update = {
                    "request_id": request.request_id,
                    "accepted_len": result.accepted_len,
                    "valid_len": valid_len,
                    "length_only": True,
                }
            else:
                update = {
                    "request_id": request.request_id,
                    "gamma": request.gamma,
                    "accepted_len": result.accepted_len,
                    "draft_len": result.draft_len,
                    "valid_len": valid_len,
                    "target_prefix_len": len(request.prefix_token_ids),
                    "replacement_token_id": result.replacement_token_id,
                    "finished": result.finished,
                    "length_only": False,
                    "prefix_token_ids": list(request.prefix_token_ids),
                }
            updates.append(update)
            committed_ids.add(request.request_id)

        if not updates:
            return set()

        self._trace(
            "commit_batch_start "
            f"round={self.round_id} batch={len(updates)} "
            f"accepted={sum(item['accepted_len'] for item in updates)} "
            f"valid={sum(item['valid_len'] for item in updates)} "
            f"length_only={sum(bool(item['length_only']) for item in updates)} "
            "wire_length_fields=accepted_len,valid_len"
        )
        self._commit_batch(updates)
        self._trace(
            "commit_batch_done "
            f"round={self.round_id} batch={len(updates)} "
            f"rows={','.join(sorted(committed_ids))}"
        )
        return committed_ids
    def _call_batch(self, requests: tuple[DraftRequest, ...]) -> list[DraftResult]:
        rows = self._request_batch(self._to_wire(requests))
        if len(rows) != len(requests):
            raise RuntimeError(
                f"Draft returned {len(rows)} rows for {len(requests)} requests"
            )
        results: list[DraftResult] = []
        for request, row in zip(requests, rows):
            draft = tuple(int(x) for x in row[: request.gamma])
            results.append(
                DraftResult(
                    request_id=request.request_id,
                    prefix_token_ids=request.prefix_token_ids,
                    draft_token_ids=draft,
                )
            )
        return results

    def _drain_stale(self, pending: _PendingPrefetch) -> None:
        # The transport is ordered.  Wait before sending rebase_batch so
        # response/round pairs cannot be mixed.
        try:
            pending.future.result()
        except Exception as exc:
            self._trace(f"prefetch_discard_error={exc!r}")

    def _rebase_current(self, current: tuple[DraftRequest, ...]) -> None:
        if not self._rebase_on_discard or self._rebase_batch is None:
            return
        self._trace(
            "discard_rebase_start "
            f"round={self.round_id} batch={len(current)}"
        )
        try:
            self._rebase_batch(self._to_wire(current))
        except Exception as exc:
            self._trace(f"discard_rebase_error={exc!r}")
            raise
        self._trace(
            "discard_rebase_done "
            f"round={self.round_id} batch={len(current)}"
        )

    def _start_prefetch(self, requests: tuple[DraftRequest, ...]) -> None:
        if self._closed:
            return
        if not requests:
            return
        with self._lock:
            if self._pending is not None:
                self._drain_stale(self._pending)
                self._pending = None
            self._pending = _PendingPrefetch(
                requests=requests,
                future=self._executor.submit(self._call_batch, requests),
            )
        self._trace(
            "prefetch_start "
            f"round={self.round_id} batch={len(requests)} "
            f"lookahead={sum(request.gamma for request in requests)}"
        )

    def _optimistic_requests(
        self,
        requests: tuple[DraftRequest, ...],
        results: Sequence[DraftResult],
    ) -> tuple[DraftRequest, ...]:
        next_requests: list[DraftRequest] = []
        for request, result in zip(requests, results):
            draft = result.draft_token_ids
            if not draft:
                continue
            next_requests.append(
                DraftRequest(
                    request_id=request.request_id,
                    prefix_token_ids=request.prefix_token_ids + draft,
                    gamma=request.gamma,
                )
            )
        return tuple(next_requests)

    def get_or_request(
        self,
        requests: Iterable[DraftRequest | dict],
        verify_results: Sequence[VerifyResult | dict] | None = None,
    ) -> list[DraftResult]:
        """Advance one batch using explicit Target verification when present.

        ``verify_results`` describes the draft that produced the current
        Target step.  A row is eligible for POST-VERIFY consumption only when
        the Target accepted its whole draft and the authoritative bonus token
        is already the first token of the prefetched continuation.  Partial
        acceptance or a different bonus rebases that row only.

        Older callers may omit ``verify_results``; in that case the previous
        prefix-compatible behavior is retained as a safe fallback.
        """
        current = self._normalize(requests)
        explicit: dict[str, VerifyResult] = {}
        if verify_results:
            for raw in verify_results:
                result = (
                    raw
                    if isinstance(raw, VerifyResult)
                    else VerifyResult.from_mapping(raw)
                )
                explicit[result.request_id] = result
            self._trace(
                "verify_result_received "
                f"round={self.round_id} batch={len(explicit)} "
                f"accepted={sum(x.accepted_len for x in explicit.values())} "
                f"draft={sum(x.draft_len for x in explicit.values())}"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("nano-pearl prefetch controller is closed")
            pending = self._pending
            self._pending = None

        pending_by_id: dict[str, tuple[DraftRequest, DraftResult]] = {}
        pending_error: Exception | None = None
        if pending is not None:
            try:
                pending_results = pending.future.result()
                pending_by_id = {
                    request.request_id: (request, result)
                    for request, result in zip(pending.requests, pending_results)
                }
            except Exception as exc:
                pending_error = exc
                self._trace(f"post_verify_prefetch_error={exc!r}")

        length_only_ids = self._eligible_length_only_ids(
            current,
            explicit,
            pending_by_id,
            pending_error,
        )
        committed_ids = self._commit_current(
            current, explicit, length_only_ids
        )

        selected: dict[str, DraftResult] = {}
        refresh_requests: list[DraftRequest] = []
        consumed_extra: dict[str, int] = {}
        explicit_rebase: list[str] = []

        for request in current:
            matched = pending_by_id.get(request.request_id)
            verification = explicit.get(request.request_id)
            if matched is None or pending_error is not None:
                refresh_requests.append(request)
                continue

            old_request, old_result = matched
            old_prefix = old_request.prefix_token_ids
            new_prefix = request.prefix_token_ids
            prefix_matches = (
                old_request.gamma == request.gamma
                and len(new_prefix) >= len(old_prefix)
                and new_prefix[: len(old_prefix)] == old_prefix
            )
            extra = new_prefix[len(old_prefix):] if prefix_matches else ()
            draft = old_result.draft_token_ids

            # Explicit verification is authoritative.  A partial rejection
            # cannot consume the optimistic continuation; it must rebase from
            # the Target prefix.  The all-accepted case may consume when the
            # target bonus is exactly the first prefetched token (or when no
            # bonus was appended to the current prefix yet).
            if verification is not None:
                draft_len_matches = verification.draft_len == len(draft)
                all_accepted = (
                    verification.accepted_len == verification.draft_len
                    and not verification.finished
                )
                bonus_matches = (
                    verification.replacement_token_id is None
                    or not extra
                    or int(extra[0]) == int(verification.replacement_token_id)
                )
                can_consume = (
                    prefix_matches
                    and draft_len_matches
                    and all_accepted
                    and bonus_matches
                    and len(extra) < len(draft)
                    and tuple(draft[: len(extra)]) == tuple(extra)
                    and bool(draft[len(extra):])
                )
                if can_consume:
                    selected[request.request_id] = DraftResult(
                        request_id=request.request_id,
                        prefix_token_ids=request.prefix_token_ids,
                        draft_token_ids=tuple(draft[len(extra):]),
                    )
                    consumed_extra[request.request_id] = len(extra)
                else:
                    refresh_requests.append(request)
                    explicit_rebase.append(request.request_id)
                continue

            # Compatibility path for older ModelRunner callers.
            can_consume = (
                prefix_matches
                and len(extra) < len(draft)
                and tuple(draft[: len(extra)]) == tuple(extra)
                and bool(draft[len(extra):])
            )
            if can_consume:
                selected[request.request_id] = DraftResult(
                    request_id=request.request_id,
                    prefix_token_ids=request.prefix_token_ids,
                    draft_token_ids=tuple(draft[len(extra):]),
                )
                consumed_extra[request.request_id] = len(extra)
            else:
                refresh_requests.append(request)

        if refresh_requests:
            if pending is not None:
                self._trace(
                    "pre_verify discard_prefetch "
                    f"round={self.round_id} batch={len(current)} "
                    f"refresh_rows={len(refresh_requests)} "
                    f"consume_rows={len(consumed_extra)} "
                    f"explicit_rebase_rows={len(explicit_rebase)}"
                )
                rebase_current = getattr(self, "_rebase_current", None)
                if callable(rebase_current):
                    uncommitted_refresh = tuple(
                        request
                        for request in refresh_requests
                        if request.request_id not in committed_ids
                    )
                    if uncommitted_refresh:
                        rebase_current(uncommitted_refresh)
            fresh = self._call_batch(tuple(refresh_requests))
            for request, result in zip(refresh_requests, fresh):
                selected[request.request_id] = result
            self.mode = PearlMode.PRE_VERIFY

        if not selected:
            raise RuntimeError("nano-pearl produced no Draft results")

        if consumed_extra:
            self._trace(
                "post_verify consume_prefetch "
                f"round={self.round_id} batch={len(current)} "
                f"consume_rows={len(consumed_extra)} "
                f"consumed_extra={sum(consumed_extra.values())}"
            )
            self.mode = PearlMode.POST_VERIFY
        elif not refresh_requests:
            self.mode = PearlMode.POST_VERIFY

        results = [selected[request.request_id] for request in current]
        self.round_id += 1
        next_requests = self._optimistic_requests(current, results)
        if next_requests:
            self.mode = PearlMode.POST_VERIFY
            self._start_prefetch(next_requests)
        return results

    def notify_verify(self, results: Sequence[VerifyResult]) -> None:
        """Record an explicit Target result when the caller has it.

        The proposer-only integration can infer this from the next prefix.  A
        future direct ModelRunner integration should call this method with the
        actual rejection result so traces do not depend on prefix inference.
        """

        if not results:
            return
        if all(result.all_accepted for result in results):
            self.mode = PearlMode.POST_VERIFY
            self._trace(
                "verify_result all_accepted "
                f"batch={len(results)}"
            )
        else:
            self.mode = PearlMode.PRE_VERIFY
            self._trace(
                "verify_result rollback "
                f"batch={len(results)} "
                f"accepted={sum(result.accepted_len for result in results)}"
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = self._pending
            self._pending = None
        if pending is not None:
            self._drain_stale(pending)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "NanoPearlPrefetchController":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def trace_from_env(prefix: str = "[PEARL_STAGE5_NANOPEARL_V1]") -> TraceFn | None:
    if os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") != "1":
        return None

    def emit(message: str) -> None:
        print(f"{prefix} {message}", flush=True)

    return emit

