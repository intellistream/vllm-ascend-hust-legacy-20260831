#!/usr/bin/env python3
"""Switch Stage-5 Target from the serial batch proposer to nano-PEARL v3.

The previous GSM8K run showed that the Target EngineArgs still loaded
``PearlSerialBatchProposer``.  This patch changes only the Stage-5 worker's
custom proposer string.  It keeps the existing batch>1 API and selects the
request-ID-aware nano-PEARL proposer with the discard-rebase hook.

Only ``pearl_stage5_worker.py`` is backed up.  The operation is idempotent and
refuses ambiguous anchors or an existing backup directory.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_worker.py")
MODULE = Path("pearl_stage5_nanoparl_proposer_v3.py")
NEW = "pearl_stage5_nanoparl_proposer_v3.PearlNanoPearlProposer"
MARKER = "PEARL_STAGE5_NANOPEARL_PROPOSER_V4"

OLD_ANCHORS = (
    "pearl_stage5_serial_batch_proposer_v1.PearlSerialBatchProposer",
    "pearl_stage5_nanoparl_proposer_v1.PearlNanoPearlProposer",
    "pearl_stage5_nanoparl_proposer_v2.PearlNanoPearlProposer",
    "pearl_stage5_proposer.PearlExternalProposer",
)


def _replace_literal(source: str, old: str) -> str:
    for quote in ('"', "'"):
        old_literal = f"{quote}{old}{quote}"
        if source.count(old_literal) == 1:
            new_literal = f"{quote}{NEW}{quote}"
            transformed = source.replace(old_literal, new_literal, 1)
            # Put the marker on its own line.  An inline comment after the
            # string can swallow a same-line closing brace in a dict literal.
            return f"# {MARKER}\n{transformed}"
    raise RuntimeError(
        f"custom proposer anchor {old!r} was not found exactly once"
    )


def transform(source: str) -> str:
    if MARKER in source:
        return source

    matches = [old for old in OLD_ANCHORS if old in source]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one current custom proposer anchor, found "
            f"{matches!r}; no files were changed"
        )

    transformed = _replace_literal(source, matches[0])
    compile(transformed, str(TARGET), "exec")
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    target = repo / TARGET
    module = repo / MODULE
    if not target.is_file():
        raise FileNotFoundError(target)
    if not module.is_file():
        raise FileNotFoundError(
            f"{module} is missing; pearl_stage5_nanoparl_proposer_v3.py "
            "must be present before switching the Target entry point"
        )

    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: select request-ID-aware nano-PEARL v3 proposer for Target")

    if MARKER in original:
        print("already patched: no files changed")
        return 0

    transformed = transform(original)
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    backup_dir = args.backup_dir
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_enable_nanoparl_proposer_v4.{stamp}"
        )
    backup_dir = backup_dir.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    shutil.copy2(target, backup_dir / target.name)
    target.write_text(transformed, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
