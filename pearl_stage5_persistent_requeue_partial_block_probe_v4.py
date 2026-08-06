#!/usr/bin/env python3
"""Add an opt-in full-recompute probe to the persistent-requeue path.

The v3 patch fixed the model-runner persistent-batch lifecycle, but the v3
test still produced a shifted Draft proposal while retaining one request-owned
KV block.  This probe keeps the request/block lifecycle unchanged and only
changes the logical reuse length to zero when explicitly enabled.  It tests
whether reusing a partially populated KV block is the remaining correctness
bug.

Enable only for the diagnostic run with:
    PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE=1

The default persistent-requeue behavior is unchanged.  Every real patch
operation creates a new full-file backup and a manifest; --dry-run changes
nothing and creates no backup.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_BLOCK_PROBE_V4"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3",
)

ANCHOR = (
    "        request.num_computed_tokens = max(0, common_len - 1)\n"
        "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)

REPLACEMENT = (
    "        # " + MARKER + "\n"
    "        # A partial KV block is not treated as reusable in this probe.\n"
    "        # Keep the request-owned block and force the scheduler/model runner\n"
    "        # to recompute the whole current prefix into it.\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            request.num_computed_tokens = 0\n"
    "        else:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if REPLACEMENT in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete v4 probe "
            "is missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required prior patch marker(s) missing: "
            f"{', '.join(missing)}; no files were changed"
        )
    if source.count(ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one persistent-requeue computed-token "
            f"anchor, found {source.count(ANCHOR)}; no files were changed"
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
            f"{repo.name}.pearl_stage5_persistent_requeue_partial_block_probe_v4."
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

    patched = original.replace(ANCHOR, REPLACEMENT, 1)
    if patched == original or MARKER not in patched or REPLACEMENT not in patched:
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
        "mode": "opt-in-force-num-computed-tokens-zero-while-preserving-block",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE=1",
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
        description="Add the v4 partial-block full-recompute diagnostic probe"
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
