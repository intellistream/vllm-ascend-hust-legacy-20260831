#!/usr/bin/env python3
"""Add an opt-in nano-PEARL correctness control for POST_VERIFY.

When ``PEARL_STAGE5_NANOPEARL_FORCE_REBASE=1`` is set, every pending
prefetch is discarded and the current authoritative Target prefix is rebased
before a fresh Draft request.  This disables only the ``consume_prefetch``
fast path; it does not change Target sampling, Draft sampling, or the
discard/rebase implementation.

Only ``pearl_stage5_nanoparl_runtime_v1.py`` is backed up.  The patch is
idempotent and refuses ambiguous anchors.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_nanoparl_runtime_v1.py")
MARKER = "# PEARL_STAGE5_NANOPEARL_FORCE_REBASE_CONTROL_V1"
ENV_NAME = "PEARL_STAGE5_NANOPEARL_FORCE_REBASE"


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    if MARKER in source:
        return source

    source = MARKER + "\n" + source
    source = _replace_once(
        source,
        "        self._trace_fn = trace\n",
        f"        # {MARKER}: diagnostic switch; default keeps existing behavior.\n"
        f"        self._force_rebase = os.environ.get(\n"
        f"            \"{ENV_NAME}\", \"0\"\n"
        "        ) == \"1\"\n"
        "        self._trace_fn = trace\n",
        "force-rebase controller state",
    )
    source = _replace_once(
        source,
        "        consumed_prefetch = False\n"
        "        if pending is not None and self._compatible(pending.requests, current):\n",
        "        consumed_prefetch = False\n"
        "        compatible = (\n"
        "            pending is not None\n"
        "            and self._compatible(pending.requests, current)\n"
        "        )\n"
        "        if self._force_rebase and pending is not None:\n"
        "            self._trace(\n"
        "                \"force_rebase disable_consume_prefetch \"\n"
        "                f\"round={self.round_id} batch={len(current)}\"\n"
        "            )\n"
        "        if compatible and not self._force_rebase:\n",
        "force-rebase consume-prefetch guard",
    )
    compile(source, str(TARGET), "exec")
    return source


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
    print(
        "change: opt-in force discard/rebase control; "
        f"set {ENV_NAME}=1 at runtime"
    )

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
            f"{repo.name}.pearl_stage5_nanoparl_force_rebase_control_v1.{stamp}"
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
