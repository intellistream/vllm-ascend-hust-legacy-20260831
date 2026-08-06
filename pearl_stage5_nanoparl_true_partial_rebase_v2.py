#!/usr/bin/env python3
"""Make nano-PEARL rebase eligible Draft slots with partial KV reuse.

The existing nano-PEARL discard path always called ``_reset_request`` from
``rebase_batch``.  That proved the state-machine ordering, but it also threw
away the live Draft request and all of its KV blocks on every rejection.

This patch keeps the existing, validated ``_requeue_request_preserve_kv``
implementation as the physical reuse path.  It is selected by default when
the request is still RUNNING, remains in ``scheduler.running``, has a valid
common prefix, and has at least one reusable token.  Requests that do not
meet those conditions retain the old fresh-reset correctness fallback.

Only ``pearl_stage5_draft.py`` is backed up.  This avoids recursively copying
dynamic build symlinks under ``csrc/build``.
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
BASE_MARKER = "# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_DRAFT_V1"
MARKER = "# PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REBASE_V2"


def transform(source: str) -> str:
    if MARKER in source:
        compile(source, str(TARGET), "exec")
        return source

    if BASE_MARKER not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py: the validated nano-PEARL rebase patch "
            f"({BASE_MARKER}) is missing; no files were changed"
        )

    if source.count("    def rebase_batch(") != 1:
        raise RuntimeError(
            "pearl_stage5_draft.py: expected one existing rebase_batch() "
            "method; no files were changed"
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
        """Rebase stale nano-PEARL slots, retaining eligible partial KV.

        ``_requeue_request_preserve_kv`` is the same-Request path used by the
        earlier persistent-requeue implementation.  It trims the optimistic
        tail, updates the runner bookkeeping, removes only the live batch row,
        and puts the Request back into WAITING without freeing its owned KV.
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
                prompt_len = len(self.prompt_token_ids or [])

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

                    if common_len < prompt_len:
                        reason = "prompt_prefix_divergence"
                    elif reusable_tokens <= 0:
                        reason = "no_reusable_tokens"
                    elif not request_is_running:
                        reason = f"request_not_running:{request.status!s}"
                    else:
                        # This is the actual same-Request partial-KV path.
                        # It intentionally does not call _reset_request().
                        self._requeue_request_preserve_kv(prefix)
                        reused = True
                        action = "retain_partial_tail"
                        reason = "eligible_running_request"

                if not reused:
                    # Correctness fallback for an ineligible or disabled slot.
                    self._reset_request(prefix)

                new_internal_id = self.request_id
                if new_internal_id is not None:
                    self._pending_tokens.pop(str(new_internal_id), None)

                results.append(
                    {
                        "request_id": external_id,
                        "prefix_len": len(prefix),
                        "common_len": common_len,
                        "reusable_tokens": reusable_tokens,
                        "action": action,
                    }
                )

                if trace:
                    if reused:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_TRUE_PARTIAL_REUSE_V2] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
                            f"prefix_len={len(prefix)} "
                            f"old_internal={old_internal_id!r} "
                            f"new_internal={new_internal_id!r} "
                            "action=retain_partial_tail "
                            f"reason={reason}",
                            flush=True,
                        )
                    else:
                        print(
                            "[PEARL_STAGE5_NANOPEARL_REBASE_V2] "
                            f"request_id={external_id!r} "
                            f"common_len={common_len} "
                            f"reusable_tokens={reusable_tokens} "
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
        prefix=f".{path.name}.pearl_true_partial.",
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
    print("change: nano-PEARL true partial-block KV reuse in rebase_batch")

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    if backup_dir_arg is None:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_nanoparl_true_partial_rebase_v2."
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
                "mode": "targeted-file-backup",
                "target": str(TARGET),
                "source_sha256": sha256(raw),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"backup: {backup_dir}")
    write_atomic(target, transformed, target.stat().st_mode)
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-ascend-hust"),
    )
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
