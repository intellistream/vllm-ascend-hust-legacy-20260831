#!/usr/bin/env python3
"""Keep request-owned KV blocks during the Stage-5 persistent requeue path.

The earlier KV recompute diagnostic intentionally called
"kv_cache_manager.free(request)" and "block_table.clear_row()" inside
"_sync_model_runner_state". That is correct for the safe full-recompute path,
but it destroys the blocks that the scheduler-aware persistent requeue is
trying to preserve.

This patch gates that old cleanup block:

* normal/safe mode: the old full-recompute cleanup remains enabled;
* PEARL_STAGE5_PERSISTENT_REQUEUE=1: request-owned blocks and the model
  runner block table are preserved.

Every real patch operation creates a new full-file backup and manifest.
--dry-run does not modify the repository or create a backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_KV_BYPASS_V2"
REQUEUE_MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_V1"

OLD_BLOCK = (
    "        # PEARL_STAGE5_KV_RECOMPUTE_V1\n"
    "        kv_cache_manager = getattr(\n"
    "            self.core.scheduler, \"kv_cache_manager\", None\n"
    "        )\n"
    "        if kv_cache_manager is None:\n"
    "            raise RuntimeError(\n"
    "                \"Cannot locate the Draft scheduler KV cache manager\"\n"
    "            )\n"
    "        # Drop all blocks owned by this live request.  The request itself\n"
    "        # remains in the scheduler and will receive fresh blocks next step.\n"
    "        kv_cache_manager.free(request)\n"
    "        # Do not let prefix caching immediately reuse the blocks just freed;\n"
    "        # this run is a correctness diagnostic for persistent KV state.\n"
    "        request.skip_reading_prefix_cache = True\n"
    "        block_ids = getattr(req_state, \"block_ids\", None)\n"
    "        if block_ids is not None:\n"
    "            req_state.block_ids = tuple([] for _ in block_ids)\n"
    "        block_table = getattr(input_batch, \"block_table\", None)\n"
    "        if block_table is None or not hasattr(block_table, \"clear_row\"):\n"
    "            raise RuntimeError(\n"
    "                \"Draft input batch has no block table clear_row API\"\n"
    "            )\n"
    "        block_table.clear_row(req_index)\n"
)

NEW_BLOCK = (
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE\", \"0\"\n"
    "        ) != \"1\":\n"
    "            # PEARL_STAGE5_PERSISTENT_REQUEUE_KV_BYPASS_V2\n"
    + textwrap.indent(OLD_BLOCK, "    ")
)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if NEW_BLOCK in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete gated "
            "block is missing; no files were changed"
        )

    if REQUEUE_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: the v1 persistent requeue patch is not present; "
            "apply pearl_stage5_persistent_requeue_v1.py first; no files "
            "were changed"
        )

    count = source.count(OLD_BLOCK)
    if count != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one KV recompute cleanup block, found "
            f"{count}; no files were changed"
        )
    return "to-patch"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit:
        backup_dir = Path(explicit).expanduser()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; choose a "
                "new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_persistent_requeue_kv_bypass_v2."
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
        print("already patched; no files changed")
        return

    patched = original.replace(OLD_BLOCK, NEW_BLOCK, 1)
    if patched == original or MARKER not in patched or NEW_BLOCK not in patched:
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
        "marker": MARKER,
        "mode": "gate-kv-recompute-cleanup-for-persistent-requeue",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE=1",
        "safe_mode": "old cleanup remains active when env is unset",
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
        description="Preserve KV blocks in the Stage-5 persistent requeue path"
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
