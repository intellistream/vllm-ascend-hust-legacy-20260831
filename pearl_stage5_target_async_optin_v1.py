#!/usr/bin/env python3
"""Make Stage-5 Target async scheduling opt-in.

The existing correctness patch hard-codes ``async_scheduling=False`` because
the external proposer currently consumes CPU-side sampled token IDs.  This
patch preserves that default and adds an explicit environment switch for an
async trial:

    PEARL_STAGE5_TARGET_ASYNC_SCHEDULING=1

Every real modification creates a new full-file backup before the atomic
replacement.  Dry-run never changes files or creates a backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGET_REL = Path("pearl_stage5_worker.py")
CONTROL_MARKER = "# PEARL_STAGE5_TARGET_SYNC_CONTROL_V1"
OPTIN_MARKER = "# PEARL_STAGE5_TARGET_ASYNC_OPTIN_V1"

OLD = (
    f"        {CONTROL_MARKER}\n"
    "        async_scheduling=False,\n"
)
NEW = (
    f"        {CONTROL_MARKER}\n"
    f"        {OPTIN_MARKER}\n"
    "        async_scheduling=(\n"
    '            os.environ.get("PEARL_STAGE5_TARGET_ASYNC_SCHEDULING", "0") == "1"\n'
    "        ),\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if source.count(NEW) == 1:
        return "post"
    if source.count(NEW) > 1:
        raise RuntimeError(
            f"{TARGET_REL}: opt-in block appears multiple times; no files changed"
        )

    if OPTIN_MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: opt-in marker exists without the expected block; "
            "no files changed"
        )
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected exactly one async=False block, found "
            f"{source.count(OLD)}; no files changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit is not None:
        backup_dir = Path(explicit).expanduser().resolve()
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_target_async_optin_v1.{timestamp()}"
        )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def write_atomic(target: Path, data: bytes, mode: int) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        os.chmod(temporary_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print(
        "change: Target async scheduling becomes opt-in via "
        "PEARL_STAGE5_TARGET_ASYNC_SCHEDULING=1"
    )

    if state == "post":
        py_compile.compile(str(target), doraise=True)
        print("already patched: py_compile PASS; no files changed")
        return

    patched = original.replace(OLD, NEW, 1)
    if patched.count(NEW) != 1:
        raise RuntimeError("internal replacement failed; no files changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "patch_script": Path(__file__).name,
                "target": str(target),
                "backup_file": str(backup_file),
                "original_sha256": sha256_bytes(original_bytes),
                "original_size": len(original_bytes),
                "marker": OPTIN_MARKER,
                "enable_env": "PEARL_STAGE5_TARGET_ASYNC_SCHEDULING=1",
                "default": "async_scheduling=False",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        write_atomic(target, patched.encode("utf-8"), target.stat().st_mode)
        py_compile.compile(str(target), doraise=True)
    except Exception:
        shutil.copy2(backup_file, target)
        raise

    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    print("py_compile: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
