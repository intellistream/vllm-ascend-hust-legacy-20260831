#!/usr/bin/env python3
"""Safely patch PEARL Stage-5 persistent Draft runner state.

The patch synchronizes the model runner's token-validity mask after the
scheduler-side prefix is rolled back or extended.  It always saves a complete
copy of the original target file in a new backup directory before replacing
the target file.
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
MARKER = "        # PEARL_STAGE5_PERSISTENT_KV_FIX_V1\n"
ANCHOR = (
    "        input_batch.token_ids_cpu[req_index, : len(prefix_token_ids)] "
    "= prefix_token_ids\n"
)
PATCH_BLOCK = (
    MARKER
    + "        is_token_ids = getattr(input_batch, \"is_token_ids\", None)\n"
    + "        if is_token_ids is not None:\n"
    + "            valid_mask = is_token_ids[req_index]\n"
    + "            if hasattr(valid_mask, \"fill_\"):\n"
    + "                valid_mask.fill_(False)\n"
    + "            else:\n"
    + "                valid_mask[...] = False\n"
    + "            valid_mask[: len(prefix_token_ids)] = True\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if PATCH_BLOCK in source:
        return "patched"
    if MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: patch marker is present but the complete patch "
            "block is missing; no files were changed"
        )
    count = source.count(ANCHOR)
    if count != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected exactly one token_ids anchor, found "
            f"{count}; no files were changed"
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
            f"{repo.name}.pearl_stage5_persistent_kv_backup_v1.{timestamp()}"
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
        print("already patched; no files changed")
        return
    if dry_run:
        patched = original.replace(ANCHOR, ANCHOR + PATCH_BLOCK, 1)
        compile(patched, str(target), "exec")
        print("dry-run passed; no files changed")
        return

    patched = original.replace(ANCHOR, ANCHOR + PATCH_BLOCK, 1)
    if patched == original:
        raise RuntimeError("internal patch replacement made no change")
    compile(patched, str(target), "exec")

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
        "marker": MARKER.strip(),
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    replaced = False
    try:
        write_atomic(target, patched, target.stat().st_mode)
        replaced = True
    except Exception:
        if replaced:
            shutil.copy2(backup_file, target)
        raise

    print(f"backup saved to: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch PEARL Stage-5 persistent Draft token-validity state"
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
        help="validate the expected pre-patch source without changing files",
    )
    args = parser.parse_args()

    try:
        apply_patch(Path(args.repo).expanduser().resolve(), args.backup_dir, args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

