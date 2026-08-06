#!/usr/bin/env python3
"""Wire explicit Target verification results into nano-PEARL.

The previous bonus-aware controller inferred acceptance from the next Target
prefix.  This patch adds the missing result path:

    Ascend rejection sampler -> ModelRunner -> custom proposer -> runtime

Each row carries ``accepted_len``, ``draft_len``, ``replacement_token_id``
and ``finished``.  The runtime uses those values to decide whether a pending
look-ahead can be consumed or must be rebased.  Prefix matching remains only
as a compatibility fallback when an older caller does not provide results.

The patch is deliberately source-guarded and backs up only the four files it
touches.  It does not recursively copy a repository containing dynamic build
symlinks.  It is safe to run first with ``--dry-run`` and refuses ambiguous
anchors.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


MARKERS = {
    "pearl_stage5_nanoparl_runtime_v1.py":
        "# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_RUNTIME_V1",
    "pearl_stage5_nanoparl_proposer_v3.py":
        "# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_PROPOSER_V1",
    "vllm_ascend/sample/rejection_sampler.py":
        "# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_SAMPLER_V1",
    "vllm_ascend/worker/model_runner_v1.py":
        "# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_RUNNER_V1",
}


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(source: str, method: str, next_method: str, body: str) -> str:
    start = source.find(f"    def {method}(")
    end = source.find(f"    def {next_method}(", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method}: method boundary not found; no files were changed"
        )
    return source[:start] + body.rstrip() + "\n\n" + source[end:]


EXPLICIT_GET_OR_REQUEST = '''    def get_or_request(
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
                    rebase_current(tuple(refresh_requests))
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
'''


def transform_runtime(source: str) -> str:
    marker = MARKERS["pearl_stage5_nanoparl_runtime_v1.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    source = replace_once(
        source,
        "        if self.accepted_len > self.draft_len:\n"
        "            raise ValueError(\"accepted_len cannot exceed draft_len\")\n\n"
        "    @property\n",
        "        if self.accepted_len > self.draft_len:\n"
        "            raise ValueError(\"accepted_len cannot exceed draft_len\")\n\n"
        "    @classmethod\n"
        "    def from_mapping(cls, value: dict) -> \"VerifyResult\":\n"
        "        return cls(\n"
        "            request_id=str(value[\"request_id\"]),\n"
        "            accepted_len=int(value[\"accepted_len\"]),\n"
        "            draft_len=int(value[\"draft_len\"]),\n"
        "            replacement_token_id=(\n"
        "                None\n"
        "                if value.get(\"replacement_token_id\") is None\n"
        "                else int(value[\"replacement_token_id\"])\n"
        "            ),\n"
        "            finished=bool(value.get(\"finished\", False)),\n"
        "        )\n\n"
        "    @property\n",
        "VerifyResult.from_mapping anchor",
    )
    source = source.replace(
        "        requests: Iterable[DraftRequest | dict],\n"
        "    ) -> list[DraftResult]:\n",
        "        requests: Iterable[DraftRequest | dict],\n"
        "        verify_results: Sequence[VerifyResult | dict] | None = None,\n"
        "    ) -> list[DraftResult]:\n",
        1,
    )
    source = replace_method(
        source,
        "get_or_request",
        "notify_verify",
        EXPLICIT_GET_OR_REQUEST,
    )
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


def transform_proposer(source: str) -> str:
    marker = MARKERS["pearl_stage5_nanoparl_proposer_v3.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    source = replace_once(
        source,
        "        sampled_token_ids = kwargs.pop(\"sampled_token_ids\", None)\n",
        "        sampled_token_ids = kwargs.pop(\"sampled_token_ids\", None)\n"
        "        verify_results = kwargs.pop(\"verify_results\", None)\n",
        "proposer verify_results argument",
    )
    source = replace_once(
        source,
        "        results = self._nano_controller.get_or_request(requests)\n",
        "        normalized_verify_results = None\n"
        "        if verify_results is not None:\n"
        "            normalized_verify_results = []\n"
        "            for raw in verify_results:\n"
        "                item = dict(raw) if isinstance(raw, dict) else raw\n"
        "                if isinstance(item, dict):\n"
        "                    row_index = item.get(\"row_index\")\n"
        "                    if row_index is not None and stable_ids is not None:\n"
        "                        row_index = int(row_index)\n"
        "                        if 0 <= row_index < len(stable_ids):\n"
        "                            item[\"request_id\"] = stable_ids[row_index]\n"
        "                    normalized_verify_results.append(item)\n"
        "                else:\n"
        "                    normalized_verify_results.append(item)\n"
        "        results = self._nano_controller.get_or_request(\n"
        "            requests, verify_results=normalized_verify_results\n"
        "        )\n",
        "proposer explicit verification dispatch",
    )
    compile(source, "pearl_stage5_nanoparl_proposer_v3.py", "exec")
    return source


SAMPLER_INSERT = '''
        # PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_SAMPLER_V1
        # Preserve the authoritative per-row rejection result for the custom
        # proposer.  The ModelRunner remaps row_index to stable request_id.
        pearl_verify_results = []
        valid_counts = (output_token_ids != PLACEHOLDER_TOKEN_ID).sum(dim=1)
        for row_index, draft_len_raw in enumerate(metadata.num_draft_tokens):
            draft_len = int(draft_len_raw)
            if draft_len <= 0:
                continue
            valid_count = int(valid_counts[row_index].item())
            accepted_len = max(0, min(draft_len, valid_count - 1))
            replacement_token_id = None
            if accepted_len < output_token_ids.shape[1]:
                candidate = int(output_token_ids[row_index, accepted_len].item())
                if candidate != PLACEHOLDER_TOKEN_ID:
                    replacement_token_id = candidate
            pearl_verify_results.append(
                {
                    "request_id": str(row_index),
                    "row_index": row_index,
                    "accepted_len": accepted_len,
                    "draft_len": draft_len,
                    "replacement_token_id": replacement_token_id,
                    "finished": False,
                }
            )
        self.last_verify_results = pearl_verify_results
'''


def transform_sampler(source: str) -> str:
    marker = MARKERS["vllm_ascend/sample/rejection_sampler.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    anchor = "\n        logprobs_tensors = None\n"
    if source.count(anchor) != 1:
        raise RuntimeError(
            "Ascend rejection sampler logprobs anchor: expected one, found "
            f"{source.count(anchor)}; no files were changed"
        )
    source = source.replace(anchor, "\n" + SAMPLER_INSERT + anchor, 1)
    compile(source, "vllm_ascend/sample/rejection_sampler.py", "exec")
    return source


def transform_runner(source: str) -> str:
    marker = MARKERS["vllm_ascend/worker/model_runner_v1.py"]
    if marker in source:
        return source
    source = marker + "\n" + source

    no_spec_anchor = "            return self.sampler(\n                logits=logits,\n                sampling_metadata=sampling_metadata,\n            )\n"
    if source.count(no_spec_anchor) >= 1:
        source = source.replace(
            no_spec_anchor,
            "            self._pearl_verify_results = None\n" + no_spec_anchor,
            1,
        )

    spec_anchor = (
        "        sampler_output = self.rejection_sampler(\n"
        "            spec_decode_metadata,\n"
        "            None,  # draft_probs\n"
        "            logits,\n"
        "            sampling_metadata,\n"
        "        )\n"
        "        return sampler_output\n"
    )
    spec_replacement = (
        "        sampler_output = self.rejection_sampler(\n"
        "            spec_decode_metadata,\n"
        "            None,  # draft_probs\n"
        "            logits,\n"
        "            sampling_metadata,\n"
        "        )\n"
        "        self._pearl_verify_results = getattr(\n"
        "            self.rejection_sampler, \"last_verify_results\", None\n"
        "        )\n"
        "        return sampler_output\n"
    )
    if source.count(spec_anchor) != 1:
        raise RuntimeError(
            "ModelRunner _sample rejection-sampler anchor: expected one, found "
            f"{source.count(spec_anchor)}; no files were changed"
        )
    source = source.replace(spec_anchor, spec_replacement, 1)

    call_anchor = (
        "draft_token_ids = self.drafter.propose(\n"
        "                valid_sampled_token_ids,\n"
        "                self.input_batch.num_tokens_no_spec,\n"
        "                self.input_batch.token_ids_cpu,\n"
    )
    start = source.find(call_anchor)
    if start < 0 or source.count(call_anchor) != 1:
        raise RuntimeError(
            "ModelRunner custom proposer call anchor: expected one, found "
            f"{source.count(call_anchor)}; no files were changed"
        )
    close = source.find("\n            )", start + len(call_anchor))
    if close < 0:
        raise RuntimeError(
            "ModelRunner custom proposer call end was not found; no files were changed"
        )
    assignment_line_start = source.rfind("\n", 0, start) + 1
    assignment_indent = source[assignment_line_start:start]
    extra = (
        "                request_ids=getattr(self.input_batch, \"req_ids\", None),\n"
        "                verify_results=getattr(self, \"_pearl_verify_results\", None),\n"
    )
    source = source[:close] + "\n" + extra.rstrip("\n") + source[close:]
    closing = source.find("\n            )", close + 1)
    if closing < 0:
        raise RuntimeError(
            "ModelRunner custom proposer call closing anchor disappeared; "
            "no files were changed"
        )
    closing_end = closing + len("\n            )")
    source = (
        source[:closing_end]
        + "\n"
        + assignment_indent
        + "self._pearl_verify_results = None"
        + source[closing_end:]
    )
    compile(source, "vllm_ascend/worker/model_runner_v1.py", "exec")
    return source


TRANSFORMS = {
    Path("pearl_stage5_nanoparl_runtime_v1.py"): transform_runtime,
    Path("pearl_stage5_nanoparl_proposer_v3.py"): transform_proposer,
    Path("vllm_ascend/sample/rejection_sampler.py"): transform_sampler,
    Path("vllm_ascend/worker/model_runner_v1.py"): transform_runner,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    originals: dict[Path, str] = {}
    transformed: dict[Path, str] = {}
    changed: list[Path] = []
    for relative, transform in TRANSFORMS.items():
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        original = target.read_text(encoding="utf-8")
        originals[relative] = original
        updated = transform(original)
        transformed[relative] = updated
        state = "post" if updated == original else "pre"
        print(f"target: {target}")
        print(f"state: {state}")
        if updated != original:
            changed.append(relative)
    print("change: explicit per-request Target verification for nano-PEARL")
    if not changed:
        print("already patched: no files changed")
        return 0
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = (
        args.backup_dir.expanduser().resolve()
        if args.backup_dir is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_explicit_verify_v1.{stamp}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    for relative in changed:
        saved = backup_dir / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, saved)
    for relative in changed:
        target = repo / relative
        temporary = target.with_name(target.name + ".pearl_stage5_tmp")
        temporary.write_text(transformed[relative], encoding="utf-8")
        temporary.replace(target)
    print(f"backup: {backup_dir}")
    for relative in changed:
        print(f"patched: {repo / relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
