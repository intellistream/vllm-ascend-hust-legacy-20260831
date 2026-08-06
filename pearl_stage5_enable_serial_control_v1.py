#!/usr/bin/env python3
"""Add an opt-in synchronous proposer switch to Stage-5 worker."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_worker.py")
MARKER = "PEARL_STAGE5_SERIAL_CONTROL_V1"
NANO_MODELS = (
    "pearl_stage5_nanoparl_proposer_v3.PearlNanoPearlProposer",
    "pearl_stage5_nanoparl_proposer_v2.PearlNanoPearlProposer",
    "pearl_stage5_nanoparl_proposer_v1.PearlNanoPearlProposer",
)


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("serial control switch is already applied")
    matches = [
        model for model in NANO_MODELS if f'"model": "{model}",' in source
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one current nano-PEARL proposer anchor, found "
            f"{matches!r}; no files were changed"
        )
    old_model = f'"model": "{matches[0]}",'
    new_model = (
        '"model": (\n'
        '                "pearl_stage5_serial_batch_proposer_v1.'
        'PearlSerialBatchProposer"\n'
        '                if os.environ.get("PEARL_STAGE5_SERIAL_CONTROL", "0")'
        ' == "1"\n'
        f'                else "{matches[0]}"\n'
        '            ),'
    )
    return source.replace(old_model, new_model, 1).replace(
        '"""Two-card Stage-5 worker: persistent Draft + custom-proposer Target."""',
        '"""Two-card Stage-5 worker with an opt-in synchronous AC control.\n\n'
        f"{MARKER}\n"
        '"""',
        1,
    )


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: opt-in synchronous batch proposer for AC control")
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
        backup_dir = (
            repo.parent
            / f"{repo.name}.pearl_stage5_serial_control_v1.{stamp}"
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
