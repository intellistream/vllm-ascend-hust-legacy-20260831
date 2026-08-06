#!/usr/bin/env python3
"""Make the validated Stage-5 partial-block fallback enabled by default.

The GSM8K control experiment showed that same-Request requeue with a
partial/un-aligned KV tail drops AC from about 74% to about 11%, while the
existing ``PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK=1`` path restores
the reference AC.  This patch changes only that environment-variable default
from ``0`` to ``1``.  Setting the variable explicitly to ``0`` still exposes
the old path for later debugging.

Every real patch creates a new complete target-file backup. ``--dry-run``
creates neither a backup nor a repository change.
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
BASE_MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1"
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_DEFAULT_V1"

OLD = (
    '            "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK", "0"\n'
)
NEW = (
    '            "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK", "1"\n'
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if NEW in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the expected default line "
            "is missing; no files were changed"
        )
    if BASE_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: expected {BASE_MARKER}; no files were changed"
        )
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one fallback-default anchor, found "
            f"{source.count(OLD)}; no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; choose a new "
                "path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_persistent_requeue_partial_fallback_default_v1."
            f"{timestamp()}"
        )
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


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(f"target file not found: {target}")

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print(
        "change: enable partial-block correctness fallback by default; "
        "explicit '=0' still disables it"
    )

    if state == "patched":
        print("already patched: no files were changed and no backup was created")
        return

    patched = original.replace(OLD, NEW, 1)
    if patched == original:
        raise RuntimeError("internal replacement failed; no files were changed")
    patched = patched.replace(
        f"# {BASE_MARKER}\n",
        f"# {BASE_MARKER}\n        # {MARKER}\n",
        1,
    )
    if MARKER not in patched:
        raise RuntimeError("marker insertion failed; no files were changed")
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
        "mode": "default-partial-block-correctness-fallback",
        "default_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK=1",
        "disable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK=0",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_atomic(target, patched, target.stat().st_mode)
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
