#!/usr/bin/env python3
"""Repair the Stage-5 true partial-block default without calling environ.get.

The first default-on patch exposed a compatibility wrapper around
``os.environ.get`` in the accumulated diagnostic source.  This repair replaces
only the ``true_partial_reuse`` assignment with a membership/index lookup, so
the default remains enabled while an explicit ``...=0`` still disables it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V1"
REPAIR_MARKER = "# PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_DEFAULT_V2"
ENV_NAME = "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_true_partial_reuse_default_v2.{stamp}"
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
    if REPAIR_MARKER in source:
        return source
    if MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: default-on v1 marker is missing; no files were changed"
        )

    marker_pos = source.index(MARKER)
    assignment_pos = source.find("true_partial_reuse", marker_pos)
    if assignment_pos < 0:
        raise RuntimeError(
            f"{TARGET_REL}: true_partial_reuse assignment is missing; "
            "no files were changed"
        )

    line_start = source.rfind("\n", 0, assignment_pos) + 1
    line_end_match = re.search(r"\n[ \t]*(?:if|elif|else|request\.|self\.|#)", source[assignment_pos:])
    reusable_end = source.find("reusable_tokens > 0", assignment_pos)
    if reusable_end < 0:
        raise RuntimeError(
            f"{TARGET_REL}: true_partial_reuse expression end is missing; "
            "no files were changed"
        )
    expression_end = source.find("\n", reusable_end)
    if expression_end < 0:
        expression_end = len(source)
    indent = source[line_start:assignment_pos]
    replacement = (
        f"{indent}{REPAIR_MARKER}\n"
        f"{indent}# Avoid the accumulated environ.get compatibility wrapper.\n"
        f"{indent}true_partial_reuse = (\n"
        f"{indent}    os.environ[{ENV_NAME!r}]\n"
        f"{indent}    if {ENV_NAME!r} in os.environ\n"
        f"{indent}    else \"1\"\n"
        f"{indent}) == \"1\" and reusable_tokens > 0"
    )
    patched = source[:line_start] + replacement + source[expression_end:]
    compile(patched, str(TARGET_REL), "exec")
    return patched


def apply_patch(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)
    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    if REPAIR_MARKER in original:
        print(f"target: {target}")
        print("state: post")
        print("already repaired: no files were changed and no backup was created")
        return

    patched = transform(original)
    print(f"target: {target}")
    print("state: pre")
    print("change: repair default-on true partial-block reuse environ access")
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
        "marker": REPAIR_MARKER,
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
