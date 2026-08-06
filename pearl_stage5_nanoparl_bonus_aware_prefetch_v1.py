#!/usr/bin/env python3
"""Add bonus-aware, per-request nano-PEARL prefetch consumption.

The exact-prefix control restored correctness but discarded every lookahead.
This patch moves one step toward PEARL's post-verify behavior: a pending Draft
result may be consumed when the authoritative Target prefix is the pending
prefix followed by an exact prefix of the pending draft tokens.  The consumed
tokens are removed from the returned lookahead, so the remaining suffix is
conditioned on the current Target prefix.  Rows that do not match are
rebased/refreshed independently instead of invalidating the whole batch.

This is safe for the current greedy GSM8K path.  A later sampler integration
should pass explicit accepted lengths for non-greedy sampling rather than
inferring them from token-prefix alignment.

Only ``pearl_stage5_nanoparl_runtime_v1.py`` is backed up.  The operation is
idempotent and refuses ambiguous method anchors.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_nanoparl_runtime_v1.py")
MARKER = "# PEARL_STAGE5_NANOPEARL_BONUS_AWARE_PREFETCH_V1"


def _replace_method(source: str) -> str:
    start = source.find("    def get_or_request(")
    if start < 0 or source.count("    def get_or_request(") != 1:
        raise RuntimeError(
            "expected exactly one get_or_request method; no files were changed"
        )
    end = source.find("    def notify_verify(", start + 1)
    if end < 0 or end <= start:
        raise RuntimeError(
            "get_or_request method boundary was not found; no files were changed"
        )

    replacement = '''    def get_or_request(
        self,
        requests: Iterable[DraftRequest | dict],
    ) -> list[DraftResult]:
        """Consume only bonus-aligned rows; refresh mismatched rows."""
        current = self._normalize(requests)
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

        consumed: dict[str, DraftResult] = {}
        refresh_requests: list[DraftRequest] = []
        consumed_extra: dict[str, int] = {}
        force_rebase = bool(getattr(self, "_force_rebase", False))

        if pending is not None and force_rebase:
            self._trace(
                "force_rebase disable_consume_prefetch "
                f"round={self.round_id} batch={len(current)}"
            )

        for request in current:
            matched = pending_by_id.get(request.request_id)
            if matched is None or pending_error is not None or force_rebase:
                refresh_requests.append(request)
                continue

            old_request, old_result = matched
            old_prefix = old_request.prefix_token_ids
            new_prefix = request.prefix_token_ids
            extra = new_prefix[len(old_prefix):] if new_prefix[:len(old_prefix)] == old_prefix else ()
            draft = old_result.draft_token_ids
            # The current Target prefix may include one or more tokens from
            # the optimistic Draft continuation.  Consume only that exact
            # prefix and return the remaining suffix.
            if (
                old_request.gamma == request.gamma
                and len(new_prefix) >= len(old_prefix)
                and new_prefix[:len(old_prefix)] == old_prefix
                and len(extra) < len(draft)
                and tuple(draft[:len(extra)]) == tuple(extra)
                and draft[len(extra):]
            ):
                consumed[request.request_id] = DraftResult(
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
                    f"consume_rows={len(consumed)}"
                )
                rebase_current = getattr(self, "_rebase_current", None)
                if callable(rebase_current):
                    rebase_current(tuple(refresh_requests))
            fresh = self._call_batch(tuple(refresh_requests))
            for request, result in zip(refresh_requests, fresh):
                consumed[request.request_id] = result
            if consumed_extra:
                self._trace(
                    "post_verify consume_prefetch "
                    f"round={self.round_id} batch={len(current)} "
                    f"consume_rows={len(consumed_extra)} "
                    f"consumed_extra={sum(consumed_extra.values())}"
                )
            self.mode = PearlMode.PRE_VERIFY
        elif consumed:
            self.mode = PearlMode.POST_VERIFY
            self._trace(
                "post_verify consume_prefetch "
                f"round={self.round_id} batch={len(current)} "
                f"consume_rows={len(consumed)} "
                f"consumed_extra={sum(consumed_extra.values())}"
            )
        else:
            raise RuntimeError("nano-pearl produced no Draft results")

        results = [consumed[request.request_id] for request in current]
        self.round_id += 1
        next_requests = self._optimistic_requests(current, results)
        if next_requests:
            self.mode = PearlMode.POST_VERIFY
            self._start_prefetch(next_requests)
        return results
'''
    return source[:start] + replacement + source[end:]


def transform(source: str) -> str:
    if MARKER in source:
        return source
    source = MARKER + "\n" + source
    transformed = _replace_method(source)
    compile(transformed, str(TARGET), "exec")
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: bonus-aware per-request prefetch consume/rebase")

    if MARKER in original:
        print("already patched: no files changed")
        return 0

    transformed = transform(original)
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    backup_dir = args.backup_dir
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_nanoparl_bonus_aware_prefetch_v1.{stamp}"
        )
    backup_dir = backup_dir.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    shutil.copy2(target, backup_dir / target.name)
    target.write_text(transformed, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
