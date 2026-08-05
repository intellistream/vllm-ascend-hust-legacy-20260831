#!/usr/bin/env python3
"""Switch Stage-5 HCCL transport to the stable-slot bridge v3."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_coordinator.py")
OLD_BRIDGE = '"pearl_stage6_hccl_bridge_v2.py"'
NEW_BRIDGE = '"pearl_stage6_hccl_bridge_v3.py"'
MARKER = "PEARL_STAGE5_HCCL_MAINFLOW_V5"


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("HCCL main-flow v5 is already applied")
    count = source.count(OLD_BRIDGE)
    if count != 1:
        raise RuntimeError(
            f"expected one HCCL bridge v2 reference, found {count}; "
            "no files were changed"
        )
    return source.replace(
        OLD_BRIDGE,
        NEW_BRIDGE + f'  # {MARKER}',
        1,
    )


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    bridge = repo / "pearl_stage6_hccl_bridge_v3.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    if not bridge.is_file():
        raise FileNotFoundError(
            f"missing stable-slot bridge: {bridge}; copy "
            "pearl_stage6_hccl_bridge_v3.py into the repository first"
        )
    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: use stable-slot HCCL bridge v3 in Stage-5")
    if MARKER in original:
        print("already patched: no files changed")
        return
    transformed = transform(original)
    if dry_run:
        digest = hashlib.sha256(original.encode()).hexdigest()[:12]
        print(f"dry-run: no files changed; source_sha256={digest}")
        return
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_hccl_mainflow_v5.{stamp}"
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
