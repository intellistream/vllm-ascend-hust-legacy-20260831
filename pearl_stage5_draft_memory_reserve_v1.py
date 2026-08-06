#!/usr/bin/env python3
"""Lower only the Draft worker's NPU memory reservation for a control run.

The change is intentionally limited to ``pearl_stage5_draft.py``.  It allows
the Qwen3-0.6B Draft to start when card 4 is already occupied by a process in
another container.  The script is idempotent, creates a complete timestamped
backup before changing the file, supports ``--dry-run``, writes atomically,
and compiles the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_DRAFT_MEMORY_RESERVE_V1"
PRE_BLOCK = (
    "            max_num_batched_tokens=self.max_model_len,\n"
    "            enforce_eager=True,\n"
)
POST_BLOCK = (
    "            max_num_batched_tokens=self.max_model_len,\n"
    f"            {MARKER}\n"
    "            gpu_memory_utilization=0.30,\n"
    "            enforce_eager=True,\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def backup_dir(repo: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return repo.parent / f"{repo.name}.pearl_stage5_draft_memory_reserve_v1.{stamp}"


def inspect(text: str) -> str:
    if text.count(POST_BLOCK) == 1:
        return "post"
    if MARKER in text or "gpu_memory_utilization=0.30" in text:
        raise RuntimeError("partial or different Draft memory patch already exists")
    count = text.count(PRE_BLOCK)
    if count != 1:
        raise RuntimeError(f"expected one Draft EngineArgs anchor, found {count}")
    return "pre"


def make_backup(target: Path, directory: Path, original: bytes) -> None:
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    saved = directory / target.name
    shutil.copy2(target, saved)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "target": str(target),
                "backup_file": str(saved),
                "sha256_before": sha256(original),
                "size_before": len(original),
                "created_at": datetime.now().isoformat(timespec="microseconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_atomic(target: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.", suffix=".tmp", dir=target.parent
    )
    mode = target.stat().st_mode
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, target)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/data/vllm-ascend-hust"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_bytes()
    text = original.decode("utf-8")
    state = inspect(text)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: Draft gpu_memory_utilization=0.30")

    if state == "post":
        py_compile.compile(str(target), doraise=True)
        print("already patched; py_compile: PASS; no files changed")
        return 0
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    directory = backup_dir(repo)
    make_backup(target, directory, original)
    patched = text.replace(PRE_BLOCK, POST_BLOCK, 1).encode("utf-8")
    try:
        write_atomic(target, patched)
        py_compile.compile(str(target), doraise=True)
    except Exception:
        shutil.copy2(directory / target.name, target)
        raise

    print(f"backup: {directory}")
    print(f"patched: {target}")
    print("py_compile: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
