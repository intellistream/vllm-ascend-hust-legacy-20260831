#!/usr/bin/env python3
"""Fix the literal marker reference in the Stage-5 runtime trace patch.

The previous trace patch accidentally emitted ``f"[{MARKER}]"`` into the
target module, although ``MARKER`` only existed in the patch script.  This
small corrective patch replaces that reference with a literal string.

It always creates a new full-file backup and a manifest before modifying the
target.  Run ``--dry-run`` first, then apply it and compile the target.
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
TRACE_MARKER = "PEARL_STAGE5_RUNTIME_STATE_TRACE_V2"

BROKEN_BLOCK = (
    "        print(\n"
    "            f\"[{MARKER}] \"\n"
    "            + repr(payload),\n"
    "            flush=True,\n"
    "        )\n"
)
FIXED_BLOCK = (
    "        print(\n"
    f"            \"[{TRACE_MARKER}] \"\n"
    "            + repr(payload),\n"
    "            flush=True,\n"
    "        )\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if FIXED_BLOCK in source:
        return "patched"
    if BROKEN_BLOCK not in source:
        raise RuntimeError(
            f"{TARGET_REL}: broken trace marker block was not found; "
            "no files were changed"
        )
    if source.count(BROKEN_BLOCK) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected exactly one broken trace marker block, "
            f"found {source.count(BROKEN_BLOCK)}; no files were changed"
        )
    return "to-patch"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit:
        backup_dir = Path(explicit).expanduser()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; "
                "choose a new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_runtime_state_trace_fix_backup_v1."
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


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(f"target file not found: {target}")

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"current state: {state}")
    if state == "patched":
        print("already fixed; no files changed")
        return

    patched = original.replace(BROKEN_BLOCK, FIXED_BLOCK, 1)
    if patched == original or FIXED_BLOCK not in patched:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run passed; no files changed")
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
        "marker": TRACE_MARKER,
        "mode": "fix-runtime-trace-marker-name-error",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_atomic(target, patched, target.stat().st_mode)
    print(f"backup saved to: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix PEARL Stage-5 runtime trace marker reference"
    )
    parser.add_argument(
        "--repo",
        default="/root/data/vllm-ascend-hust",
        help="Ascend bridge repository",
    )
    parser.add_argument(
        "--backup-dir",
        help="new backup directory to create; it must not already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source without changing files",
    )
    args = parser.parse_args()

    try:
        apply_patch(
            Path(args.repo).expanduser().resolve(),
            args.backup_dir,
            args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
