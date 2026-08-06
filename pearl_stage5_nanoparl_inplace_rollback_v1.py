#!/usr/bin/env python3
"""Add the first true nano-PEARL in-place Draft rollback path.

The existing true-partial path preserves the Request and its KV blocks, but
still removes the model-runner row and requeues the Request through the
scheduler. This patch adds an opt-in fast path that keeps the Request in
scheduler.running and updates only the authoritative token boundary and the
runner-side computed-token state.

Enable the new path for a test with:

    PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK=1

The default remains disabled until the batch>1 correctness smoke test passes.
Only pearl_stage5_draft.py is backed up; no recursive repository backup is
performed, so build symlinks cannot make the backup fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1"
ENV_NAME = "PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK"


def _method_bounds(source: str, method_name: str) -> tuple[int, int]:
    """Return the source span of one top-level class method."""
    start_marker = f"    def {method_name}"
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError(
            f"{TARGET}: cannot find {start_marker}(); "
            "apply the existing true-partial rebase patch first; "
            "no files were changed"
        )
    if source.find(start_marker, start + 1) >= 0:
        raise RuntimeError(
            f"{TARGET}: found more than one {start_marker}(); "
            "no files were changed"
        )

    next_method = source.find("\n    def ", start + len(start_marker))
    end = len(source) if next_method < 0 else next_method + 1
    return start, end


def _inplace_branch() -> str:
    return f'''        # {MARKER}
        # The normal path below performs batch-row removal plus
        # RUNNING -> WAITING -> re-add. True nano-PEARL keeps the live row
        # and only moves the authoritative boundary backward. The current
        # vLLM V1 convention leaves the last committed token uncomputed, so
        # valid_len is represented by num_computed_tokens + 1.
        if os.environ.get("{ENV_NAME}", "0") == "1":
            accepted_len = common_len
            valid_len = max(0, accepted_len - 1)
            request.num_computed_tokens = valid_len
            request.is_prefill_chunk = False
            request.spec_token_ids = []
            request.num_output_placeholders = 0

            # Keep the existing Request, model-runner row, and request-owned
            # block IDs. _sync_model_runner_state updates the CPU scheduling
            # view; the next normal engine step consumes only the suffix
            # beyond valid_len. Do not free blocks or rebuild the row.
            self._sync_model_runner_state(prefix_token_ids, request)
            inflight_prefills = getattr(scheduler, "_inflight_prefills", None)
            if inflight_prefills is not None:
                inflight_prefills.discard(request)

            print(
                "[PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1] "
                f"request={{request.request_id!r}} "
                f"accepted_len={{accepted_len}} "
                f"valid_len={{valid_len}} "
                f"num_computed_tokens={{request.num_computed_tokens}} "
                "action=retain_running_row",
                flush=True,
            )
            return

'''


def transform(source: str) -> str:
    if MARKER in source:
        compile(source, str(TARGET), "exec")
        return source

    start, end = _method_bounds(source, "_requeue_request_preserve_kv")
    method = source[start:end]
    sync_line = "        self._sync_model_runner_state(prefix_token_ids, request)\n"
    sync_count = method.count(sync_line)
    if sync_count != 1:
        raise RuntimeError(
            f"{TARGET}: expected one runner-state sync inside "
            f"_requeue_request_preserve_kv(), found {sync_count}; "
            "no files were changed"
        )

    # Insert the fast path immediately before the existing sync/remove/requeue
    # tail. Returning here leaves the old correctness path untouched when the
    # environment switch is absent or set to 0.
    method = method.replace(sync_line, _inplace_branch() + sync_line, 1)
    transformed = source[:start] + method + source[end:]
    transformed = MARKER + "\n" + transformed
    compile(transformed, str(TARGET), "exec")
    return transformed


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.pearl_inplace_rollback.",
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

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    already_patched = MARKER in original
    transformed = transform(original)

    print(f"target: {target}")
    print(f"state: {'post' if already_patched else 'pre'}")
    print("change: opt-in in-place accepted_len/valid_len rollback")

    if transformed == original:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_nanoparl_inplace_rollback_v1.{_timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_file = backup_dir / TARGET
    backup_file.write_bytes(original_bytes)
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "patch_script": Path(__file__).name,
                "target": str(target),
                "backup_file": str(backup_file),
                "original_sha256": _sha256(original_bytes),
                "marker": MARKER,
                "mode": "in-place-running-row-rollback",
                "enable_env": f"{ENV_NAME}=1",
                "async_scheduling": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _write_atomic(target, transformed, target.stat().st_mode)
    except BaseException:
        try:
            target.write_bytes(original_bytes)
        except BaseException:
            pass
        raise

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

