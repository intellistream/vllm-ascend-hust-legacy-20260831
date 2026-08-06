#!/usr/bin/env python3
"""Switch Stage-5 from nano-PEARL proposer v1 to API-compatible v2."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_worker.py")
OLD = "pearl_stage5_nanoparl_proposer_v1.PearlNanoPearlProposer"
NEW = "pearl_stage5_nanoparl_proposer_v2.PearlNanoPearlProposer"
MARKER = "PEARL_STAGE5_NANOPEARL_PROPOSER_V2"


def transform(source: str) -> str:
    if MARKER in source:
        return source
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"expected one v1 proposer anchor, found {count}; "
            "no files were changed"
        )
    transformed = source.replace(OLD, NEW, 1)
    transformed = transformed.replace(
        f'"{NEW}",',
        f'"{NEW}",  # {MARKER}',
        1,
    )
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    target = repo / TARGET
    module = repo / "pearl_stage5_nanoparl_proposer_v2.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    if not module.is_file():
        raise FileNotFoundError(module)

    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: accept current vLLM-HUST proposer request_ids API")
    if MARKER in original:
        print("already patched: no files changed")
        return 0

    transformed = transform(original)
    if args.dry_run:
        print("dry-run: no files changed and no backup was created")
        return 0

    backup_dir = args.backup_dir
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_enable_nanoparl_v2.{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / target.name)
    target.write_text(transformed, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

