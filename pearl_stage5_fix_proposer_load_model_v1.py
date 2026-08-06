#!/usr/bin/env python3
"""Safely add the vLLM custom-proposer load_model compatibility hook.

The Stage-5 external proposer owns no local draft model.  In this vLLM-HUST
version, ModelRunner still calls ``self.drafter.load_model(self.model)`` after
loading the Target model, so the external proposer needs a no-op method.

This script:
  1. checks the current source using stable anchors;
  2. saves a complete pre-change copy before writing;
  3. applies the change atomically;
  4. supports --dry-run and --restore-from.

Examples:
    python pearl_stage5_fix_proposer_load_model_v1.py \
        --repo /root/data/vllm-ascend-hust --dry-run
    python pearl_stage5_fix_proposer_load_model_v1.py \
        --repo /root/data/vllm-ascend-hust
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET = Path("pearl_stage5_proposer.py")
CLASS_ANCHOR = "class PearlExternalProposer:"
METHOD_ANCHOR = "    def _connect(self) -> None:"
METHOD_INSERT = '''    def load_model(self, model: Any = None) -> None:
        """Compatibility hook for vLLM-HUST's ModelRunner.

        This proposer is model-free: the Draft model lives in the separate
        persistent Draft worker and is queried through AF_UNIX.  The Target
        ModelRunner nevertheless calls ``load_model(self.model)`` for every
        proposer implementation, so this method intentionally does nothing.
        """
        return None

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
        default=Path("."),
        help="vllm-ascend-hust repository root",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="optional explicit backup directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restore-from",
        type=Path,
        default=None,
        help="restore pearl_stage5_proposer.py from a backup directory",
    )
    return parser


def transform(text: str) -> tuple[str, str]:
    if "    def load_model(self, model: Any = None) -> None:" in text:
        return text, "already-patched"

    class_pos = text.find(CLASS_ANCHOR)
    method_pos = text.find(METHOD_ANCHOR, class_pos)
    if class_pos < 0 or method_pos < 0:
        raise RuntimeError(
            f"{TARGET} does not contain the expected stable anchors; "
            "no files were changed"
        )

    return text[:method_pos] + METHOD_INSERT + text[method_pos:], "to-patch"


def backup_target(target: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / TARGET.name)
    manifest = {
        "target": str(target),
        "backup_file": str(backup_dir / TARGET.name),
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


def apply(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
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
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_proposer_backup_v1.{timestamp}"
    backup_target(target, backup_dir)

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
        apply(repo, args.backup_dir.resolve() if args.backup_dir else None, args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
