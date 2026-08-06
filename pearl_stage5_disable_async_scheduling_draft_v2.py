#!/usr/bin/env python3
"""Force synchronous scheduling in the Stage-5 in-process Draft engine.

This v2 patch locates the Draft ``EngineArgs(...)`` block and inserts the
option after ``enforce_eager=True`` without assuming a particular ordering of
the other EngineArgs fields.
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
MARKER = "# PEARL_STAGE5_DISABLE_ASYNC_SCHEDULING_DRAFT_V2"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE_V1",
)
ENGINE_START = "        engine_args = EngineArgs(\n"
ENGINE_END = "        )\n        self.engine = LLMEngine.from_engine_args(\n"
ENFORCE_LINE = "            enforce_eager=True,\n"
ASYNC_LINE = "            async_scheduling=False,\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def locate_engine_args(source: str) -> tuple[int, int, str]:
    start = source.find(ENGINE_START)
    if start < 0:
        raise RuntimeError(
            f"{TARGET_REL}: Draft EngineArgs start not found; no files were changed"
        )
    end = source.find(ENGINE_END, start)
    if end < 0:
        raise RuntimeError(
            f"{TARGET_REL}: Draft EngineArgs end not found; no files were changed"
        )
    end += len("        )\n")
    block = source[start:end]
    if block.count(ENFORCE_LINE) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one enforce_eager line inside Draft "
            f"EngineArgs, found {block.count(ENFORCE_LINE)}; no files were changed"
        )
    return start, end, block


def inspect_state(source: str) -> str:
    if MARKER in source:
        if ASYNC_LINE in source:
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )
    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required marker(s) missing: {missing}; "
            "no files were changed"
        )
    _, _, block = locate_engine_args(source)
    if ASYNC_LINE in block:
        raise RuntimeError(
            f"{TARGET_REL}: async_scheduling is already present without the "
            f"{MARKER} marker; inspect manually and do not overwrite"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    backup_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_disable_async_scheduling_draft_v2."
        f"{timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
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
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: Draft EngineArgs async_scheduling=False")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    start, end, block = locate_engine_args(original)
    replacement = block.replace(
        ENFORCE_LINE,
        ENFORCE_LINE + f"{MARKER}\n" + ASYNC_LINE,
        1,
    )
    patched = original[:start] + replacement + original[end:]
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
        "mode": "draft-synchronous-scheduling",
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
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
