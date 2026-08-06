#!/usr/bin/env python3
"""Fix the literal marker reference in the Stage-5 full-block trace.

The diagnostic method was inserted into ``pearl_stage5_draft.py`` but its
print statement accidentally referenced ``MARKER``, which exists only in the
patch script.  This patch replaces that reference with the literal marker.

Every real modification creates a new full-file backup before the atomic
replacement.  ``--dry-run`` does not create a backup or change files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1"
OLD = '        print(f"[{MARKER}] " + repr(payload), flush=True)\n'
NEW = (
    '        print("[PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1] " '
    "+ repr(payload), flush=True)\n"
)
REQUIRED = (
    "    def _trace_full_block_state(\n",
    "full_requeue.before_replace",
    "full_requeue.after_sync",
    "full_requeue.after_enqueue",
    "full_requeue.before_get_output",
    "full_requeue.after_get_output",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if NEW in source and all(item in source for item in REQUIRED):
        return "post"
    missing = [item for item in REQUIRED if item not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required trace anchors missing: {missing}; "
            "no files were changed"
        )
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: bad MARKER reference count={source.count(OLD)}; "
            "no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    backup_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_persistent_requeue_full_block_trace_fix_v1."
        f"{timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def write_atomic(target: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        dir=str(target.parent),
        text=True,
    )
    try:
        os.chmod(temp_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: replace trace-local MARKER reference with literal marker")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    patched = original.replace(OLD, NEW, 1)
    compile(patched, str(target), "exec")
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "target": str(target),
        "backup_file": str(backup_file),
        "original_sha256": sha256_bytes(original_bytes),
        "original_size": len(original_bytes),
        "marker": MARKER,
        "mode": "trace-marker-reference-fix",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        write_atomic(target, patched, target.stat().st_mode)
    except Exception:
        shutil.copy2(backup_file, target)
        raise
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
