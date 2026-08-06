#!/usr/bin/env python3
"""Safely add acceptance logging at vLLM-HUST's scheduler statistics point.

The v1 patch instrumented RejectionSampler.forward, but the HUST runtime
computes the final accepted-token count later in the V1 scheduler.  This patch
adds logging immediately after:

    num_rejected = num_draft_tokens - num_accepted

The original target file is copied to a new timestamped backup directory
before modification.  The patch is enabled only by:

    PEARL_ACCEPTANCE_DEBUG=1

Examples:

    python pearl_stage5_acceptance_debug_v2.py \
        --repo /root/data/vllm-hust --dry-run

    python pearl_stage5_acceptance_debug_v2.py \
        --repo /root/data/vllm-hust
"""

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
PATCH_MARKER = "# PEARL_ACCEPTANCE_DEBUG_V2"
ANCHOR_RE = re.compile(
    r"^(?P<indent>[ \t]*)num_rejected\s*=\s*"
    r"num_draft_tokens\s*-\s*num_accepted\s*$",
    re.MULTILINE,
)

DEBUG_BODY = '''{indent}{marker}
{indent}if os.environ.get("PEARL_ACCEPTANCE_DEBUG", "0") == "1":
{indent}    round_draft = int(num_draft_tokens)
{indent}    round_accepted = int(num_accepted)
{indent}    self._pearl_debug_rounds = getattr(self, "_pearl_debug_rounds", 0) + 1
{indent}    self._pearl_debug_draft_tokens = (
{indent}        getattr(self, "_pearl_debug_draft_tokens", 0) + round_draft
{indent}    )
{indent}    self._pearl_debug_accepted_tokens = (
{indent}        getattr(self, "_pearl_debug_accepted_tokens", 0) + round_accepted
{indent}    )
{indent}    total_rounds = self._pearl_debug_rounds
{indent}    total_draft = self._pearl_debug_draft_tokens
{indent}    total_accepted = self._pearl_debug_accepted_tokens
{indent}    round_rate = (
{indent}        100.0 * round_accepted / round_draft
{indent}        if round_draft > 0
{indent}        else float("nan")
{indent}    )
{indent}    total_rate = (
{indent}        100.0 * total_accepted / total_draft
{indent}        if total_draft > 0
{indent}        else float("nan")
{indent}    )
{indent}    mean_acceptance_length = (
{indent}        1.0 + total_accepted / total_rounds
{indent}        if total_rounds > 0
{indent}        else float("nan")
{indent}    )
{indent}    print(
{indent}        "[PEARL_ACCEPTANCE_DEBUG_V2] "
{indent}        f"round_draft={{round_draft}} "
{indent}        f"round_accepted={{round_accepted}} "
{indent}        f"round_rejected={{int(num_rejected)}} "
{indent}        f"round_rate={{round_rate:.2f}}% "
{indent}        f"total_draft={{total_draft}} "
{indent}        f"total_accepted={{total_accepted}} "
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-hust"),
        help="vllm core repository root",
    )
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore-from", type=Path, default=None)
    return parser


def transform(text: str) -> tuple[str, str]:
    if PATCH_MARKER in text:
        return text, "already-patched"

    matches = list(ANCHOR_RE.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"{TARGET}: expected exactly one scheduler acceptance anchor, "
            f"found {len(matches)}; no files were changed"
        )

    match = matches[0]
    indent = match.group("indent")
    insertion = "\n" + DEBUG_BODY.format(
        indent=indent,
        marker=PATCH_MARKER,
    )
    updated = text[: match.end()] + insertion + text[match.end() :]

    if "\nimport os\n" not in updated:
        future_line = "from __future__ import annotations\n"
        future_end = updated.find(future_line)
        if future_end < 0:
            raise RuntimeError(
                f"{TARGET}: future-import anchor not found; no files were changed"
            )
        future_end += len(future_line)
        updated = updated[:future_end] + "\nimport os\n" + updated[future_end:]

    return updated, "to-patch"


def save_backup(target: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_file = backup_dir / TARGET.name
    shutil.copy2(target, backup_file)
    manifest = {
        "target": str(target),
        "backup_file": str(backup_file),
        "sha256_before": sha256(target),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def restore(repo: Path, backup_dir: Path) -> None:
    source = backup_dir / TARGET.name
    if not source.is_file():
        raise FileNotFoundError(f"backup file does not exist: {source}")
    target = repo / TARGET
    shutil.copy2(source, target)
    print(f"restored: {target}")
    print(f"from:    {source}")


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(f"target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")
    updated, state = transform(original)
    print(f"current state: {TARGET}: {state}")

    if state == "already-patched":
        print("no changes needed")
        return
    if dry_run:
        print("dry-run passed; no files changed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if backup_dir is None:
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_acceptance_backup_v2.{timestamp}"
    save_backup(target, backup_dir)

    temporary = target.with_name(target.name + ".pearl_stage5_tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(backup_dir / TARGET.name, target)
        raise

    print(f"backup saved to: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve()
    if args.restore_from is not None:
        restore(repo, args.restore_from.resolve())
    else:
        apply_patch(
            repo,
            args.backup_dir.resolve() if args.backup_dir else None,
            args.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
