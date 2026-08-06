#!/usr/bin/env python3
"""Honor the Stage-5 fresh-block diagnostic flag.

The current Draft requeue path skips KV block freeing whenever
PEARL_STAGE5_PERSISTENT_REQUEUE=1, even when
PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK=1 is also set. That leaves stale
blocks attached to a request whose computed-token count was reset to zero.

This patch only makes the existing fresh-block flag effective. It does not
implement persistent KV reuse; it is a correctness-control patch to verify
the stale-KV hypothesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


TARGET = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_FIX_V1"

OLD = '''        if os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE", "0"
        ) != "1":
'''

NEW = '''        # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_FIX_V1
        # A correctness diagnostic may explicitly request fresh KV blocks even
        # while the persistent-requeue path remains enabled.
        persistent_requeue = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE", "0"
        ) == "1"
        force_fresh_block = os.environ.get(
            "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK", "0"
        ) == "1"
        if not persistent_requeue or force_fresh_block:
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(text: str) -> tuple[str, str]:
    if MARKER in text:
        return text, "post"
    if text.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET}: expected exactly one requeue condition; no files were changed"
        )
    return text.replace(OLD, NEW, 1), "pre"


def make_backup(target: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_file = backup_dir / TARGET.name
    shutil.copy2(target, backup_file)
    manifest = {
        "target": str(target),
        "backup_file": str(backup_file),
        "sha256_before": sha256(target),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    updated, state = transform(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: honor PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK=1")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return 0
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = (
        args.backup_dir.resolve()
        if args.backup_dir is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_persistent_requeue_fresh_block_fix_v1.{timestamp}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup already exists: {backup_dir}")
    make_backup(target, backup_dir)
    print(f"backup: {backup_dir}")

    temporary = target.with_name(target.name + ".pearl_stage5_fresh_block_tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(backup_dir / TARGET.name, target)
        raise

    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
