#!/usr/bin/env python3
"""Fix nano-PEARL rebase false resets caused by optimistic prompt state.

The Draft prefetch path intentionally places the optimistic Target prefix plus
draft tokens into ``self.prompt_token_ids``.  When Target rejects a tail token,
the old rebase guard mistakes that expected tail mismatch for a user-prompt
mismatch and creates a fresh Request.  v3 keeps a per-external-request
authoritative rebase anchor and temporarily uses that anchor while invoking
the validated same-Request KV requeue path.

Only ``pearl_stage5_draft.py`` is modified and backed up.  The patch is
idempotent and refuses to overwrite an existing backup directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path


TARGET = Path("pearl_stage5_draft.py")
BASE_MARKER = "# PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REBASE_V2"
MARKER = "# PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REBASE_V3"


def transform(source: str) -> str:
    if MARKER in source:
        compile(source, str(TARGET), "exec")
        return source

    if BASE_MARKER not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py: v2 nano-PEARL true-partial rebase marker "
            f"({BASE_MARKER}) is missing; no files were changed"
        )

    if source.count("    def rebase_batch(") != 1:
        raise RuntimeError(
            "pearl_stage5_draft.py: expected one rebase_batch() method; "
            "no files were changed"
        )
    if source.count("    def propose_batch(") != 1:
        raise RuntimeError(
            "pearl_stage5_draft.py: expected one propose_batch() method; "
            "no files were changed"
        )

    start = source.find("    def rebase_batch(")
    end = source.find("    def propose_batch(", start + 1)
    if start < 0 or end <= start:
        raise RuntimeError(
            "pearl_stage5_draft.py: could not locate rebase_batch boundaries; "
            "no files were changed"
        )

    method = r'''    def rebase_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rebase Target prefixes without mistaking optimistic tails for prompts.

        During PRE/POST-VERIFY prefetch, ``self.prompt_token_ids`` is updated
        to the optimistic prefix so vLLM's Request and ModelRunner remain
        synchronized.  That value is *not* an immutable user prompt: it can
        contain draft tokens which Target is expected to roll back.

        Keep one first-authoritative-prefix-minus-one-token anchor per external
        Target request.  The minus-one accounts for the sampled Target token
        that is already present in the custom proposer prefix on the first
        call.  Later Target prefixes must preserve this anchor; only their
        suffix is eligible for same-Request partial-KV rebase.
        """
        if not isinstance(requests, list) or not requests:
            raise ValueError("rebase_batch requires a non-empty requests list")

        join_pipeline = getattr(self, "_join_pipeline", None)
        if callable(join_pipeline):
            join_pipeline()

        normalized: list[tuple[str, list[int]]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                raise ValueError(f"rebase request {index} must be an object")
            external_id = str(item.get("request_id", f"row-{index}"))
            prefix = item.get("prefix_token_ids")
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(
                    "rebase prefix_token_ids must be non-empty for "
                    f"{external_id!r}"
                )
            normalized.append((external_id, [int(x) for x in prefix]))

        true_partial_reuse = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE", "1"
        ) != "0"
        trace = os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1"
        anchor_by_external_id = getattr(
            self, "_nano_rebase_anchor_by_external_id", None
        )
        if anchor_by_external_id is None:
            anchor_by_external_id = {}
            self._nano_rebase_anchor_by_external_id = anchor_by_external_id

        results: list[dict[str, Any]] = []

        with self._lock:
            for external_id, prefix in normalized:
                self._activate_request(external_id)
                old_internal_id = self.request_id
                if old_internal_id is not None:
                    self._pending_tokens.pop(str(old_internal_id), None)

                old_prefix = list(self.committed_token_ids)
                common_len = 0
                for old_token, new_token in zip(old_prefix, prefix):
                    if old_token != new_token:
                        break
                    common_len += 1
                reusable_tokens = max(0, common_len - 1)

                # The first rebase prefix is authoritative Target state.  The
                # proposer includes one already-sampled Target token, so keep
                # the stable portion before that token as this slot's anchor.
                anchor = anchor_by_external_id.get(external_id)
                if anchor is None:
                    anchor = tuple(prefix[:-1] if len(prefix) > 1 else prefix)
                    anchor_by_external_id[external_id] = anchor
                anchor = tuple(int(x) for x in anchor)
                anchor_len = len(anchor)
                anchor_matches = (
                    len(prefix) >= anchor_len
                    and prefix[:anchor_len] == list(anchor)
                )

                reused = False
                action = "fresh_reset"
                reason = "true_partial_reuse_disabled"

                if true_partial_reuse and old_internal_id is not None:
                    from vllm.v1.request import RequestStatus

                    request = self._request()
                    scheduler = self.core.scheduler
                    request_is_running = (
                        request.status == RequestStatus.RUNNING
                        and request in scheduler.running
                    )

                    if not anchor_matches:
                        # An external ID must not silently reuse KV from a
                        # different prompt.  Reset the anchor only after the
                        # request has been sent through the fresh path below.
                        reason = "rebase_anchor_divergence"
                    elif common_len < anchor_len:
                        reason = "common_prefix_before_rebase_anchor"
                    elif reusable_tokens <= 0:
                        reason = "no_reusable_tokens"
                    elif not request_is_running:
                        reason = f"request_not_running:{request.status!s}"
                    else:
                        # The validated helper checks self.prompt_token_ids.
                        # At this point that field may contain optimistic draft
                        # tokens, so temporarily expose only the stable anchor.
                        # The helper then rewrites the Request/runner state to
                        # the complete authoritative Target prefix and retains
                        # the existing Request-owned KV blocks.
                        self.prompt_token_ids = list(anchor)
                        self._requeue_request_preserve_kv(prefix)
                        if self.prompt_token_ids != prefix:
                            self.prompt_token_ids = list(prefix)
                        reused = True
                        action = "retain_partial_tail"
                        reason = "eligible_running_request_anchor_safe"

                if not reused:
                    # Correctness fallback for a new/ineligible slot or a
                    # request whose stable anchor no longer matches.
                    self._reset_request(prefix)
                    anchor_by_external_id[external_id] = tuple(
                        prefix[:-1] if len(prefix) > 1 else prefix
                    )
                    anchor = anchor_by_external_id[external_id]
                    anchor_len = len(anchor)

                new_internal_id = self.request_id
                if new_internal_id is not None:
                    self._pending_tokens.pop(str(new_internal_id), None)

                results.append(
                    {
                        "request_id": external_id,
                        "prefix_len": len(prefix),
                        "common_len": common_len,
                        "reusable_tokens": reusable_tokens,
                        "anchor_len": anchor_len,
                        "action": action,
                    }
                )

                if trace:
                    if reused:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REUSE_V3] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
                            f"anchor_len={anchor_len} "
                            f"prefix_len={len(prefix)} "
                            f"old_internal={old_internal_id!r} "
                            f"new_internal={new_internal_id!r} "
                            "action=retain_partial_tail "
                            f"reason={reason}",
                            flush=True,
                        )
                    else:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_REBASE_V3] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
                            f"anchor_len={anchor_len} "
                            f"prefix_len={len(prefix)} "
                            f"old_internal={old_internal_id!r} "
                            f"new_internal={new_internal_id!r} "
                            "action=fresh_reset "
                            f"reason={reason}",
                            flush=True,
                        )

        return results

'''
    transformed = source[:start] + method + source[end:]
    transformed = MARKER + "\n" + transformed
    compile(transformed, str(TARGET), "exec")
    return transformed


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.pearl_true_partial_v3.",
        dir=str(path.parent),
        text=True,
    )
    try:
        os.chmod(temp_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)

    raw = target.read_bytes()
    source = raw.decode("utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in source else 'pre'}")
    transformed = transform(source)
    print(
        "change: nano-PEARL stable-anchor true partial-block KV rebase v3"
    )

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    if backup_dir_arg is None:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_nanoparl_true_partial_rebase_v3."
            f"{timestamp()}"
        )
    else:
        backup_dir = backup_dir_arg.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )

    backup_dir.mkdir(parents=True)
    (backup_dir / TARGET).write_bytes(raw)
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "target": str(target),
                "source_sha256": sha256(raw),
                "marker": MARKER,
                "base_marker": BASE_MARKER,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_atomic(target, transformed, stat.S_IMODE(target.stat().st_mode))
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
