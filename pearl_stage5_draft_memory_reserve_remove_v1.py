#!/usr/bin/env python3
"""Remove the temporary 0.30 Draft memory reservation.

The Draft previously used ``gpu_memory_utilization=0.30`` because NPU4 had
another workload.  After moving Draft to a free NPU5, restore the normal vLLM
default by removing that temporary argument and its marker.

Every real patch creates a new complete timestamped backup before changing the
target.  ``--dry-run`` creates no backup and changes no files.
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
OLD_MARKER = "# PEARL_STAGE5_DRAFT_MEMORY_RESERVE_V1"
NEW_MARKER = "PEARL_STAGE5_DRAFT_MEMORY_RESERVE_REMOVE_V1"

OLD = (
    "            max_num_batched_tokens=self.max_model_len,\n"
    f"            {OLD_MARKER}\n"
    "            gpu_memory_utilization=0.30,\n"
    "            enforce_eager=True,\n"
)
NEW = (
    "            max_num_batched_tokens=self.max_model_len,\n"
    "            enforce_eager=True,\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if NEW_MARKER in source:
        if OLD_MARKER not in source and "gpu_memory_utilization=0.30" not in source:
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: remove marker is present but the old memory "
            "setting remains; no files were changed"
        )
    if source.count(OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected exactly one temporary memory block, "
            f"found {source.count(OLD)}; no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; choose a "
                "new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_draft_memory_reserve_remove_v1."
            f"{timestamp()}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def write_atomic(target: Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    mode = stat.S_IMODE(target.stat().st_mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_bytes()
    source = original.decode("utf-8")
    state = inspect_state(source)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: remove temporary Draft gpu_memory_utilization=0.30")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    patched_source = source.replace(OLD, NEW, 1)
    # Add a durable marker without reintroducing a runtime setting.
    patched_source = patched_source.replace(
        "            max_num_batched_tokens=self.max_model_len,\n"
        "            enforce_eager=True,\n",
        "            max_num_batched_tokens=self.max_model_len,\n"
        f"            # {NEW_MARKER}\n"
        "            enforce_eager=True,\n",
        1,
    )
    patched = patched_source.encode("utf-8")
    if patched == original or NEW_MARKER not in patched_source:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched_source, str(target), "exec")

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "patch_script": Path(__file__).name,
                "target": str(target),
                "backup_file": str(backup_file),
                "original_sha256": sha256_bytes(original),
                "original_size": len(original),
                "marker": NEW_MARKER,
                "change": "remove temporary Draft memory reservation",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        write_atomic(target, patched)
        compile(patched_source, str(target), "exec")
    except Exception:
        shutil.copy2(backup_file, target)
        raise

    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    print("py_compile: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        apply_patch(args.repo.resolve(), args.backup_dir, args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
