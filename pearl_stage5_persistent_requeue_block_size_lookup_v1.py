#!/usr/bin/env python3
"""Resolve the actual KV block-size owner for the Stage-5 probe.

The partial-block fallback passed its correctness test, but the runtime trace
reported ``block_size=None``.  This patch only broadens the read-only lookup
to the common vLLM locations (manager, block pool, scheduler, and cache
config).  It does not change the fallback or persistent-requeue decision.

Every real patch creates a new full-file backup.  ``--dry-run`` creates no
backup and changes no files.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_BLOCK_SIZE_LOOKUP_V1"
REQUIRED_MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1"

OLD = (
    "            block_size = getattr(kv_cache_manager, \"block_size\", None)\n"
    "            if block_size is None:\n"
    "                block_size = getattr(\n"
    "                    kv_cache_manager, \"block_size_tokens\", None\n"
    "                )\n"
)

NEW = (
    f"            # {MARKER}\n"
    "            scheduler = self.core.scheduler\n"
    "            block_size_candidates = (\n"
    "                (\n"
    "                    \"kv_cache_manager.block_size\",\n"
    "                    getattr(kv_cache_manager, \"block_size\", None),\n"
    "                ),\n"
    "                (\n"
    "                    \"kv_cache_manager.block_size_tokens\",\n"
    "                    getattr(kv_cache_manager, \"block_size_tokens\", None),\n"
    "                ),\n"
    "                (\n"
    "                    \"kv_cache_manager.block_pool.block_size\",\n"
    "                    getattr(\n"
    "                        getattr(kv_cache_manager, \"block_pool\", None),\n"
    "                        \"block_size\",\n"
    "                        None,\n"
    "                    ),\n"
    "                ),\n"
    "                (\n"
    "                    \"kv_cache_manager.cache_config.block_size\",\n"
    "                    getattr(\n"
    "                        getattr(kv_cache_manager, \"cache_config\", None),\n"
    "                        \"block_size\",\n"
    "                        None,\n"
    "                    ),\n"
    "                ),\n"
    "                (\n"
    "                    \"scheduler.block_size\",\n"
    "                    getattr(scheduler, \"block_size\", None),\n"
    "                ),\n"
    "                (\n"
    "                    \"scheduler.cache_config.block_size\",\n"
    "                    getattr(\n"
    "                        getattr(scheduler, \"cache_config\", None),\n"
    "                        \"block_size\",\n"
    "                        None,\n"
    "                    ),\n"
    "                ),\n"
    "                (\n"
    "                    \"core.vllm_config.cache_config.block_size\",\n"
    "                    getattr(\n"
    "                        getattr(\n"
    "                            getattr(self.core, \"vllm_config\", None),\n"
    "                            \"cache_config\",\n"
    "                            None,\n"
    "                        ),\n"
    "                        \"block_size\",\n"
    "                        None,\n"
    "                    ),\n"
    "                ),\n"
    "            )\n"
    "            block_size = None\n"
    "            block_size_source = \"unknown\"\n"
    "            for candidate_name, candidate_value in block_size_candidates:\n"
    "                try:\n"
    "                    candidate_int = int(candidate_value)\n"
    "                except (TypeError, ValueError):\n"
    "                    continue\n"
    "                if candidate_int > 0:\n"
    "                    block_size = candidate_int\n"
    "                    block_size_source = candidate_name\n"
    "                    break\n"
)

TRACE_OLD = (
    "                        f\"block_size={block_size!r} reason={reason} \"\n"
)
TRACE_NEW = (
    "                        f\"block_size={block_size!r} \"\n"
    "                        f\"block_size_source={block_size_source} \"\n"
    "                        f\"reason={reason} \"\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if NEW in source and TRACE_NEW in source:
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete lookup "
            "patch is missing; no files were changed"
        )
    if REQUIRED_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: required partial-fallback patch marker is "
            "missing; no files were changed"
        )
    if source.count(OLD) != 1 or source.count(TRACE_OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one block-size lookup and one trace "
            f"anchor, found {source.count(OLD)} and {source.count(TRACE_OLD)}; "
            "no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit is not None:
        backup_dir = Path(explicit).expanduser().resolve()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; choose a "
                "new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_persistent_requeue_block_size_lookup_v1."
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
    print("change: broaden read-only KV block-size lookup")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    patched = original.replace(OLD, NEW, 1).replace(TRACE_OLD, TRACE_NEW, 1)
    if patched == original or MARKER not in patched or TRACE_NEW not in patched:
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
        "mode": "read-only-kv-block-size-source-discovery",
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
