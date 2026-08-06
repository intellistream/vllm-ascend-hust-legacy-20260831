#!/usr/bin/env python3
"""Safely add the external proposer dummy_run compatibility hook.

During vLLM V1 startup, ModelRunner performs a dummy/profile run to estimate
KV-cache memory and calls ``self.drafter.dummy_run(...)``.  The Stage-5
PearlExternalProposer has no local model, so this hook must be a no-op.

The script saves a complete copy of pearl_stage5_proposer.py before changing
it and supports dry-run and restore.
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
METHOD_INSERT = '''    def dummy_run(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility hook for vLLM-HUST's startup memory profiling.

        The Draft model is resident in the separate Draft worker.  The
        external proposer therefore has no model forward or KV-cache work to
        perform during Target's dummy/profile run.
        """
        return None

'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--backup-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--restore-from", type=Path, default=None)
    return p


def transform(text: str) -> tuple[str, str]:
    if "    def dummy_run(self, *args: Any, **kwargs: Any) -> None:" in text:
        return text, "already-patched"

    class_pos = text.find(CLASS_ANCHOR)
    method_pos = text.find(METHOD_ANCHOR, class_pos)
    if class_pos < 0 or method_pos < 0:
        raise RuntimeError(
            f"{TARGET} does not contain the expected stable anchors; "
            "no files were changed"
        )
    return text[:method_pos] + METHOD_INSERT + text[method_pos:], "to-patch"


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
    backup_file = backup_dir / TARGET.name
    if not backup_file.is_file():
        raise FileNotFoundError(f"backup file does not exist: {backup_file}")
    target = repo / TARGET
    shutil.copy2(backup_file, target)
    print(f"restored: {target}")
    print(f"from:    {backup_file}")


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
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_dummy_run_backup_v1.{timestamp}"
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
    args = parser().parse_args()
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
