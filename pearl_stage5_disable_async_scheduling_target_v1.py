#!/usr/bin/env python3
"""Disable async scheduling only for the Stage-5 Target worker.

This is a controlled correctness experiment.  It changes only the Target
``LLM(...)`` construction in ``pearl_stage5_worker.py``; the Draft worker and
the vLLM core are untouched.

The script is idempotent.  Before an actual change it saves a complete copy of
the target file in a new timestamped backup directory, writes atomically, and
compiles the patched file.  On compilation failure it restores the backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


TARGET_REL = Path("pearl_stage5_worker.py")
MARKER = "# PEARL_STAGE5_TARGET_SYNC_CONTROL_V1"
PRE_BLOCK = "        enforce_eager=True,\n        trust_remote_code=True,"
POST_BLOCK = (
    "        enforce_eager=True,\n"
    f"        {MARKER}\n"
    "        async_scheduling=False,\n"
    "        trust_remote_code=True,"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(text: str) -> str:
    post_count = text.count(POST_BLOCK)
    if post_count == 1:
        return "post"
    if post_count > 1:
        raise RuntimeError(
            f"{TARGET_REL} contains the post-patch block {post_count} times"
        )

    if MARKER in text or "async_scheduling=False" in text:
        raise RuntimeError(
            f"{TARGET_REL} contains a partial or different async-scheduling patch"
        )

    pre_count = text.count(PRE_BLOCK)
    if pre_count != 1:
        raise RuntimeError(
            f"{TARGET_REL} expected exactly one Target LLM anchor, found {pre_count}"
        )
    return "pre"


def make_backup(target: Path, backup_dir: Path, original: bytes) -> None:
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    manifest = {
        "target": str(target),
        "backup_file": str(backup_file),
        "sha256_before": sha256_bytes(original),
        "size_before": len(original),
        "created_at": datetime.now().isoformat(timespec="microseconds"),
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_atomic(target: Path, data: bytes) -> None:
    mode = target.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def compile_target(target: Path) -> None:
    py_compile.compile(str(target), doraise=True)


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(f"target file does not exist: {target}")

    original = target.read_bytes()
    original_text = original.decode("utf-8")
    state = inspect_state(original_text)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: Target LLM async_scheduling=False")

    if state == "post":
        compile_target(target)
        print("already patched; py_compile passed; no files changed")
        return

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    if backup_dir_arg:
        backup_dir = Path(backup_dir_arg)
        if not backup_dir.is_absolute():
            backup_dir = repo.parent / backup_dir
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_disable_async_scheduling_target_v1."
            f"{timestamp()}"
        )

    make_backup(target, backup_dir, original)
    patched_text = original_text.replace(PRE_BLOCK, POST_BLOCK, 1)
    patched = patched_text.encode("utf-8")
    try:
        write_atomic(target, patched)
        compile_target(target)
    except Exception:
        shutil.copy2(backup_dir / target.name, target)
        raise

    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    print("py_compile: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-ascend-hust"),
    )
    parser.add_argument("--backup-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
