#!/usr/bin/env python3
"""Make nano-PEARL prefetch consumption require exact request state.

The previous compatibility check accepted a Target prefix that merely started
with the optimistic Draft prefix.  That is unsafe when Target has appended a
bonus token: the prefetched Draft result was computed without that token.
This patch permits ``consume_prefetch`` only when request ID, gamma, and the
complete prefix are identical.  Any prefix growth or divergence takes the
existing discard/rebase path.

Only ``pearl_stage5_nanoparl_runtime_v1.py`` is backed up.  The operation is
idempotent and refuses ambiguous method anchors.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_nanoparl_runtime_v1.py")
MARKER = "# PEARL_STAGE5_NANOPEARL_EXACT_PREFIX_COMPAT_V1"


def _replace_method(source: str) -> str:
    start = source.find("    def _compatible(")
    if start < 0:
        raise RuntimeError(
            "_compatible method anchor was not found; no files were changed"
        )
    end = source.find("    @staticmethod\n    def _to_wire", start)
    if end < 0 or end <= start:
        raise RuntimeError(
            "_compatible method boundary was not found; no files were changed"
        )
    if source.count("    def _compatible(") != 1:
        raise RuntimeError(
            "expected exactly one _compatible method; no files were changed"
        )

    replacement = '''    @staticmethod
    def _compatible(
        expected: tuple[DraftRequest, ...],
        current: tuple[DraftRequest, ...],
    ) -> bool:
        """Return true only for the exact state used by the prefetch."""
        if len(expected) != len(current):
            return False
        for old, new in zip(expected, current):
            if old.request_id != new.request_id:
                return False
            if old.gamma != new.gamma:
                return False
            # A Target bonus token or any other prefix change invalidates the
            # Draft result computed for the optimistic prefix.
            if new.prefix_token_ids != old.prefix_token_ids:
                return False
        return True
'''
    return source[:start] + replacement + source[end:]


def transform(source: str) -> str:
    if MARKER in source:
        return source
    source = MARKER + "\n" + source
    transformed = _replace_method(source)
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
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: require exact request/prefix match before consuming prefetch")

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
            f"{repo.name}.pearl_stage5_nanoparl_exact_prefix_compat_v1.{stamp}"
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
