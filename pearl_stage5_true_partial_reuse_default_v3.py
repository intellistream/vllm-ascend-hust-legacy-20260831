#!/usr/bin/env python3
"""Repair the malformed default-on true partial-block expression.

v1 replaced only the inner environment-key line and left the original
``true_partial_reuse = os.environ.get(`` prefix in the accumulated source.
v2 then replaced only the inner assignment.  This v3 repair replaces the
whole malformed expression atomically and keeps explicit ``=0`` support.
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
V1_MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V1"
V2_MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V2"
V3_MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V3"
ENV_NAME = "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_true_partial_reuse_default_v3.{stamp}"
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
    if V3_MARKER in source:
        return source
    if V1_MARKER not in source or V2_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: expected default-on v1/v2 markers are missing; "
            "no files were changed"
        )

    marker_pos = source.index(V1_MARKER)
    outer_expr = "true_partial_reuse = os.environ.get("
    outer_pos = source.rfind(outer_expr, 0, marker_pos)
    if outer_pos < 0:
        raise RuntimeError(
            f"{TARGET_REL}: malformed outer true_partial_reuse expression was not found; "
            "no files were changed"
        )
    start = source.rfind("\n", 0, outer_pos) + 1

    block_end_anchor = source.find("partial_recompute_block_size", marker_pos)
    if block_end_anchor < 0:
        raise RuntimeError(
            f"{TARGET_REL}: following partial-recompute block was not found; "
            "no files were changed"
        )
    end_expr = ") == \"1\" and reusable_tokens > 0"
    end_pos = source.rfind(end_expr, marker_pos, block_end_anchor)
    if end_pos < 0:
        raise RuntimeError(
            f"{TARGET_REL}: malformed true_partial_reuse expression end was not found; "
            "no files were changed"
        )
    end = source.find("\n", end_pos)
    if end < 0:
        end = len(source)

    indent = source[start:outer_pos]
    replacement = (
        f"{indent}{V1_MARKER}\n"
        f"{indent}{V2_MARKER}\n"
        f"{indent}{V3_MARKER}\n"
        f"{indent}# Clean default-on expression; explicit env=0 still disables reuse.\n"
        f"{indent}true_partial_reuse = (\n"
        f"{indent}    os.environ[{ENV_NAME!r}]\n"
        f"{indent}    if {ENV_NAME!r} in os.environ\n"
        f"{indent}    else \"1\"\n"
        f"{indent}) == \"1\" and reusable_tokens > 0"
    )
    patched = source[:start] + replacement + source[end:]
    compile(patched, str(TARGET_REL), "exec")
    return patched


def apply_patch(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)
    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    if V3_MARKER in original:
        print(f"target: {target}")
        print("state: post")
        print("already repaired: no files were changed and no backup was created")
        return

    patched = transform(original)
    print(f"target: {target}")
    print("state: pre")
    print("change: clean malformed default-on true partial-block expression")
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
        "marker": V3_MARKER,
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
