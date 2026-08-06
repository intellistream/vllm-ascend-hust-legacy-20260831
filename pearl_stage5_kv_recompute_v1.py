#!/usr/bin/env python3
"""Safely patch PEARL Stage-5 persistent KV invalidation.

This diagnostic patch keeps the Draft request alive, but releases its KV
blocks and clears the model-runner block table whenever the Target prefix is
synchronized.  The next schedule therefore recomputes the whole current
prefix.  It is intended to prove the KV-lifetime hypothesis before optimizing
to common-prefix reuse.
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
MARKER = "        # PEARL_STAGE5_KV_RECOMPUTE_V1\n"
COMPUTED_ANCHOR = (
    "        request.num_computed_tokens = "
    "max(0, len(prefix_token_ids) - 1)\n"
)
SYNC_ANCHOR = (
    "        input_batch.num_computed_tokens_cpu[req_index] = "
    "request.num_computed_tokens\n"
)
COMPUTED_REPLACEMENT = (
    MARKER
    + "        # Diagnostic mode: rebuild the current prefix from token 0.\n"
    + "        request.num_computed_tokens = 0\n"
)
SYNC_INSERTION = (
    "        # PEARL_STAGE5_KV_RECOMPUTE_V1\n"
    + "        kv_cache_manager = getattr(\n"
    + "            self.core.scheduler, \"kv_cache_manager\", None\n"
    + "        )\n"
    + "        if kv_cache_manager is None:\n"
    + "            raise RuntimeError(\n"
    + "                \"Cannot locate the Draft scheduler KV cache manager\"\n"
    + "            )\n"
    + "        # Drop all blocks owned by this live request.  The request itself\n"
    + "        # remains in the scheduler and will receive fresh blocks next step.\n"
    + "        kv_cache_manager.free(request)\n"
    + "        # Do not let prefix caching immediately reuse the blocks just freed;\n"
    + "        # this run is a correctness diagnostic for persistent KV state.\n"
    + "        request.skip_reading_prefix_cache = True\n"
    + "        block_ids = getattr(req_state, \"block_ids\", None)\n"
    + "        if block_ids is not None:\n"
    + "            req_state.block_ids = tuple([] for _ in block_ids)\n"
    + "        block_table = getattr(input_batch, \"block_table\", None)\n"
    + "        if block_table is None or not hasattr(block_table, \"clear_row\"):\n"
    + "            raise RuntimeError(\n"
    + "                \"Draft input batch has no block table clear_row API\"\n"
    + "            )\n"
    + "        block_table.clear_row(req_index)\n"
)
FULL_PATCH = (COMPUTED_REPLACEMENT, SYNC_INSERTION)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if all(part in source for part in FULL_PATCH):
        return "patched"
    if MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: patch marker is present but the complete patch "
            "block is missing; no files were changed"
        )
    computed_count = source.count(COMPUTED_ANCHOR)
    sync_count = source.count(SYNC_ANCHOR)
    if computed_count != 1 or sync_count != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one computed-token anchor and one "
            f"model-runner anchor, found {computed_count} and {sync_count}; "
            "no files were changed"
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
            f"{repo.name}.pearl_stage5_kv_recompute_backup_v1.{timestamp()}"
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

    patched = original.replace(
        COMPUTED_ANCHOR, COMPUTED_REPLACEMENT, 1
    ).replace(SYNC_ANCHOR, SYNC_INSERTION + SYNC_ANCHOR, 1)
    if patched == original:
        raise RuntimeError("internal patch replacement made no change")
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
        "marker": MARKER.strip(),
        "mode": "persistent-request-full-prefix-recompute",
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
        description="Patch PEARL Stage-5 persistent KV recompute diagnostic"
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
