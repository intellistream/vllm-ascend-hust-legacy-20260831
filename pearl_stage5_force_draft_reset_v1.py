#!/usr/bin/env python3
"""Add an opt-in full-reset mode to the persistent Draft bridge.

When PEARL_DRAFT_FORCE_RESET=1, every Target prefix starts a fresh Draft
request. This is a correctness isolation experiment: it removes persistent
Draft KV reuse without changing the Target or acceptance logic. The default
path remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_DRAFT_FORCE_RESET_V1"
ANCHOR = (
    "        if not prefix_token_ids:\n"
    "            raise ValueError(\"Target prefix must not be empty\")\n"
)
INSERT = (
    "        # PEARL_DRAFT_FORCE_RESET_V1\n"
    "        if os.environ.get(\"PEARL_DRAFT_FORCE_RESET\", \"0\") == \"1\":\n"
    "            self._reset_request(prefix_token_ids)\n"
    "            return\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(text: str) -> tuple[str, str]:
    if MARKER in text:
        return text, "already-patched"
    if text.count(ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET}: expected exactly one sync_prefix anchor; "
            "no files were changed"
        )
    return text.replace(ANCHOR, ANCHOR + INSERT, 1), "to-patch"


def save_backup(target: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    saved = directory / TARGET.name
    shutil.copy2(target, saved)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "target": str(target),
                "backup_file": str(saved),
                "sha256_before": sha256(target),
                "created_at": datetime.now().astimezone().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def restore(repo: Path, directory: Path) -> None:
    source = directory / TARGET.name
    if not source.is_file():
        raise FileNotFoundError(f"backup file does not exist: {source}")
    target = repo / TARGET
    shutil.copy2(source, target)
    print(f"restored: {target}")
    print(f"from:    {source}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-ascend-hust"),
    )
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore-from", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    target = repo / TARGET
    if args.restore_from is not None:
        restore(repo, args.restore_from.resolve())
        return 0
    if not target.is_file():
        raise FileNotFoundError(f"target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")
    updated, state = transform(original)
    print(f"current state: {TARGET}: {state}")
    if state == "already-patched":
        print("no changes needed")
        return 0
    if args.dry_run:
        print("dry-run passed; no files changed")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = (
        args.backup_dir.resolve()
        if args.backup_dir
        else repo.parent
        / f"{repo.name}.pearl_stage5_force_reset_backup_v1.{timestamp}"
    )
    save_backup(target, directory)

    temporary = target.with_name(target.name + ".pearl_stage5_tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(directory / TARGET.name, target)
        raise

    print(f"backup saved to: {directory}")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
