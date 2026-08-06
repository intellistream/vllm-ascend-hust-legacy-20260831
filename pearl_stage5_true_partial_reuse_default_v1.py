#!/usr/bin/env python3
"""Make validated true partial-block KV reuse the default Stage-5 path.

The existing true partial-tail implementation remains unchanged.  This patch
only changes its environment-variable fallback from ``0`` to ``1``.  The
explicit override is preserved: setting
``PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE=0`` disables it.
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
BASE_MARKER = "# PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_BLOCK_REUSE_V1"
MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V1"
ENV_NAME = "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE"


OLD_BLOCKS = (
    "            'PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE', \"0\"\n",
    '            "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE", "0"\n',
)
NEW_BLOCK = (
    f"        {MARKER}\n"
    "        # Default-on; set the environment variable to 0 for a diagnostic fallback.\n"
    "        true_partial_reuse = os.environ.get(\n"
    f"            {ENV_NAME!r}, \"1\"\n"
    "        ) == \"1\" and reusable_tokens > 0\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_true_partial_reuse_default_v1.{stamp}"
        )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
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


def transform(source: str) -> str:
    if MARKER in source:
        return source
    if BASE_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: true partial-block reuse is not installed; "
            "apply pearl_stage5_persistent_requeue_true_partial_reuse_v1.py "
            "first; no files were changed"
        )

    matches = [block for block in OLD_BLOCKS if source.count(block)]
    if len(matches) != 1 or source.count(matches[0]) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one true-partial default anchor, found "
            f"{len(matches)}; no files were changed"
        )
    patched = source.replace(matches[0], NEW_BLOCK, 1)
    if patched == source or MARKER not in patched or '"1"' not in NEW_BLOCK:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(TARGET_REL), "exec")
    return patched


def apply_patch(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)
    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    if MARKER in original:
        print(f"target: {target}")
        print("state: post")
        print("already patched: no files were changed and no backup was created")
        return

    patched = transform(original)
    print(f"target: {target}")
    print("state: pre")
    print(
        "change: true partial-block KV reuse default=1; explicit env=0 still disables"
    )
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "target": str(target),
        "backup_file": str(backup_file),
        "marker": MARKER,
        "original_sha256": sha256_bytes(original_bytes),
        "original_size": len(original_bytes),
        "default_env": f"{ENV_NAME}=1",
        "disable_env": f"{ENV_NAME}=0",
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
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
