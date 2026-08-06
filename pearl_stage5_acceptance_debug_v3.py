#!/usr/bin/env python3
"""Patch the V1 scheduler acceptance counter with safe, timestamped backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET = Path("vllm/v1/core/sched/scheduler.py")
MARKER = "# PEARL_ACCEPTANCE_DEBUG_V3"
ANCHOR_RE = re.compile(
    r"^(?P<indent>[ \t]*)num_rejected\s*=\s*"
    r"num_draft_tokens\s*-\s*num_accepted\s*$",
    re.MULTILINE,
)


def debug_block(indent: str) -> str:
    return f'''\n{indent}{MARKER}
{indent}if __import__("os").environ.get("PEARL_ACCEPTANCE_DEBUG", "0") == "1":
{indent}    round_draft = int(num_draft_tokens)
{indent}    round_accepted = int(num_accepted)
{indent}    self._pearl_debug_rounds = getattr(self, "_pearl_debug_rounds", 0) + 1
{indent}    self._pearl_debug_draft_tokens = getattr(self, "_pearl_debug_draft_tokens", 0) + round_draft
{indent}    self._pearl_debug_accepted_tokens = getattr(self, "_pearl_debug_accepted_tokens", 0) + round_accepted
{indent}    total_rounds = self._pearl_debug_rounds
{indent}    total_draft = self._pearl_debug_draft_tokens
{indent}    total_accepted = self._pearl_debug_accepted_tokens
{indent}    round_rate = 100.0 * round_accepted / round_draft if round_draft else float("nan")
{indent}    total_rate = 100.0 * total_accepted / total_draft if total_draft else float("nan")
{indent}    mean_acceptance_length = 1.0 + total_accepted / total_rounds
{indent}    print(
{indent}        "[PEARL_ACCEPTANCE_DEBUG_V3] "
{indent}        f"round_draft={{round_draft}} round_accepted={{round_accepted}} "
{indent}        f"round_rejected={{int(num_rejected)}} round_rate={{round_rate:.2f}}% "
{indent}        f"total_draft={{total_draft}} total_accepted={{total_accepted}} "
{indent}        f"total_rate={{total_rate:.2f}}% "
{indent}        f"mean_acceptance_length={{mean_acceptance_length:.3f}}",
{indent}        flush=True,
{indent}    )
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(text: str) -> tuple[str, str]:
    if MARKER in text:
        return text, "already-patched"
    matches = list(ANCHOR_RE.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{TARGET}: expected exactly one acceptance anchor, found "
            f"{len(matches)}; no files were changed"
        )
    match = matches[0]
    return text[:match.end()] + debug_block(match.group("indent")) + text[match.end():], "to-patch"


def backup(target: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    saved = directory / TARGET.name
    shutil.copy2(target, saved)
    (directory / "manifest.json").write_text(
        json.dumps({
            "target": str(target),
            "backup_file": str(saved),
            "sha256_before": sha256(target),
            "created_at": datetime.now().astimezone().isoformat(),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/data/vllm-hust"))
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    updated, state = transform(original)
    print(f"current state: {TARGET}: {state}")
    if state == "already-patched" or args.dry_run:
        print("no files changed" if state == "already-patched" else "dry-run passed; no files changed")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = args.backup_dir.resolve() if args.backup_dir else repo.parent / f"{repo.name}.pearl_stage5_acceptance_backup_v3.{timestamp}"
    backup(target, directory)
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
