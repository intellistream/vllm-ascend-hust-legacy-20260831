#!/usr/bin/env python3
"""Add a conservative partial-block fallback for Stage-5 Draft requeue.

The fresh-request control proved that the current same-Request requeue path
can leave incorrect model-runner state.  A KV block is reusable only when the
computed prefix ends on a complete block boundary.  This opt-in diagnostic
falls back to the already-validated ``_reset_request`` path whenever the
common prefix does not provide at least one complete KV block.

Enable it with:

    PEARL_STAGE5_PERSISTENT_REQUEUE=1
    PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK=1

The existing persistent path is unchanged when the flag is unset.  This is a
conservative correctness guard, not yet the final full-block KV-reuse path.
Every real patch creates a new complete target-file backup; ``--dry-run``
creates neither a backup nor a repository change.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1"

OLD = (
    "        prompt_len = len(self.prompt_token_ids or [])\n"
    "        if common_len < prompt_len:\n"
    "            raise RuntimeError(\n"
    "                \"Target prefix diverged inside the Draft prompt: \"\n"
    "                f\"common_len={common_len} prompt_len={prompt_len}\"\n"
    "            )\n"
    "\n"
    "        # Synchronize the live Request's logical prompt with the complete\n"
)

NEW = (
    "        prompt_len = len(self.prompt_token_ids or [])\n"
    "        if common_len < prompt_len:\n"
    "            raise RuntimeError(\n"
    "                \"Target prefix diverged inside the Draft prompt: \"\n"
    "                f\"common_len={common_len} prompt_len={prompt_len}\"\n"
    "            )\n"
    "\n"
    f"        # {MARKER}\n"
    "        # Do not attempt same-Request KV reuse through a partial block.\n"
    "        # The reset path is the correctness reference established by the\n"
    "        # fresh-request control experiment.\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK\", \"0\"\n"
    "        ) == \"1\":\n"
    "            kv_cache_manager = getattr(\n"
    "                self.core.scheduler, \"kv_cache_manager\", None\n"
    "            )\n"
    "            block_size = getattr(kv_cache_manager, \"block_size\", None)\n"
    "            if block_size is None:\n"
    "                block_size = getattr(\n"
    "                    kv_cache_manager, \"block_size_tokens\", None\n"
    "                )\n"
    "            if not isinstance(block_size, int) or block_size <= 0:\n"
    "                reusable_tokens = max(0, common_len - 1)\n"
    "                reason = \"unknown_block_size\"\n"
    "                can_reuse_full_block = False\n"
    "            else:\n"
    "                reusable_tokens = max(0, common_len - 1)\n"
    "                reason = (\n"
    "                    \"less_than_one_block\"\n"
    "                    if reusable_tokens < block_size\n"
    "                    else \"partial_block\"\n"
    "                )\n"
    "                can_reuse_full_block = (\n"
    "                    reusable_tokens >= block_size\n"
    "                    and reusable_tokens % block_size == 0\n"
    "                )\n"
    "            if not can_reuse_full_block:\n"
    "                if os.environ.get(\n"
    "                    \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
    "                ) == \"1\":\n"
    "                    print(\n"
    "                        \"[PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1] \"\n"
    "                        f\"request={self.request_id!r} \"\n"
    "                        f\"common_len={common_len} \"\n"
    "                        f\"reusable_tokens={reusable_tokens} \"\n"
    "                        f\"block_size={block_size!r} reason={reason} \"\n"
    "                        \"action=reset_request\",\n"
    "                        flush=True,\n"
    "                    )\n"
    "                self._reset_request(prefix_token_ids)\n"
    "                return\n"
    "\n"
    "        # Synchronize the live Request's logical prompt with the complete\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if NEW in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )
    if "PEARL_STAGE5_PERSISTENT_REQUEUE_V1" not in source:
        raise RuntimeError(
            f"{TARGET_REL}: expected the existing persistent-requeue method; "
            "no files were changed"
        )
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one common-prefix anchor, found "
            f"{source.count(OLD)}; no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit is not None:
        backup_dir = Path(explicit).expanduser().resolve()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; "
                "choose a new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_persistent_requeue_partial_fallback_v1."
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
    print(f"target: {target}")
    print(f"state: {state}")
    print(
        "change: partial/non-aligned common prefix falls back to "
        "_reset_request when the opt-in flag is enabled"
    )

    if state == "patched":
        print("already patched: no files were changed and no backup was created")
        return
    patched = original.replace(OLD, NEW, 1)
    if patched == original or MARKER not in patched:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
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
        "mode": "opt-in-partial-kv-block-correctness-fallback",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK=1",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_atomic(target, patched, target.stat().st_mode)
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
