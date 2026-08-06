#!/usr/bin/env python3
"""Fix the Stage-5 HCCL main-flow worker launch for batch-aware coordinators.

v2 added the HCCL transport, but newer Stage-5 coordinators have an extra
``max_num_seqs`` argument in ``launch``.  v3 supplies that argument only in
the HCCL branch and leaves the existing RPC path unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_coordinator.py")
V2_MARKER = "# PEARL_STAGE5_HCCL_MAINFLOW_V1"
V3_MARKER = "# PEARL_STAGE5_HCCL_MAINFLOW_V3"


def transform(source: str) -> str:
    if V3_MARKER in source:
        raise RuntimeError("HCCL main-flow v3 is already applied")
    if V2_MARKER not in source:
        raise RuntimeError(
            "Stage-5 HCCL v2 marker was not found; apply v2 first"
        )

    start = source.find("def _hccl_main(")
    end = source.find("\ndef connect(", start)
    if start < 0 or end <= start:
        raise RuntimeError("cannot locate the HCCL main-flow block")

    block = source[start:end]
    old = "args.max_model_len,\n                args.gamma,"
    new = (
        "args.max_model_len,\n"
        "                getattr(args, \"max_num_seqs\", 1),\n"
        "                args.gamma,"
    )
    count = block.count(old)
    if count != 2:
        raise RuntimeError(
            "expected two HCCL worker launch anchors, "
            f"found {count}; no files were changed"
        )

    block = block.replace(old, new)
    block = block.replace(
        V2_MARKER,
        V2_MARKER + "\n        " + V3_MARKER,
        1,
    )
    return source[:start] + block + source[end:]


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if V3_MARKER in original else 'pre'}")
    print("change: pass max_num_seqs through the HCCL worker launches")
    if V3_MARKER in original:
        print("already patched: no files changed")
        return

    transformed = transform(original)
    if dry_run:
        digest = hashlib.sha256(original.encode()).hexdigest()[:12]
        print(f"dry-run: no files changed; source_sha256={digest}")
        return

    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = (
            repo.parent
            / f"{repo.name}.pearl_stage5_hccl_mainflow_v3.{stamp}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / target.name)
    target.write_text(transformed, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
