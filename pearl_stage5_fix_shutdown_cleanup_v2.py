#!/usr/bin/env python3
"""Retry the Stage-5 shutdown patch with a symlink-safe repository backup.

Version 1 did not modify the source: its full-repository backup failed while
copying a volatile build artifact.  This version preserves symlinks so
generated or dangling shared-library links do not abort the backup before the
source patch is applied.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import pearl_stage5_fix_shutdown_cleanup_v1 as base


def _default_backup_dir(repo: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return repo.parent / (
        f"{repo.name}.pearl_stage5_fix_shutdown_cleanup_v2.{stamp}"
    )


def _copy_repository_backup(repo: Path, backup_dir: Path) -> None:
    # Keep symlinks instead of following generated .so links under csrc/build.
    # This also makes the snapshot stable when a build process removes a
    # dangling generated link during the copy.
    shutil.copytree(
        repo,
        backup_dir,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / base.TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    state = base.inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: Target shutdown compatibility fallback (symlink-safe backup)")

    if dry_run or state == "post":
        if dry_run:
            print("dry-run: no files were changed and no backup was created")
        else:
            print("already applied: no files were changed and no backup was created")
        return

    if backup_dir is None:
        backup_dir = _default_backup_dir(repo)
    backup_dir = backup_dir.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup path already exists; refusing to overwrite: {backup_dir}"
        )

    patched = original.replace(
        base.RUN_TARGET_ANCHOR,
        base.HELPER + base.RUN_TARGET_ANCHOR,
        1,
    )
    patched = patched.replace(base.SHUTDOWN_OLD, base.SHUTDOWN_NEW, 1)
    if patched == original:
        raise RuntimeError("patch produced no change; no files were changed")
    if base.inspect_state(patched) != "post":
        raise RuntimeError(
            "patched source failed post-state validation; no files were changed"
        )

    # The source is written only after the complete new backup succeeds.
    _copy_repository_backup(repo, backup_dir)
    target.write_text(patched, encoding="utf-8")
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

