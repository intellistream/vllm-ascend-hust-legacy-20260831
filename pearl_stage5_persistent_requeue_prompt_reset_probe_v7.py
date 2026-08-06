#!/usr/bin/env python3
"""Synchronize the Request prompt fields on the persistent-requeue path.

The v5 fresh-block and v6 cached-state controls still produced the same
shifted proposal.  A normal safe reset creates a new Request whose
``prompt_token_ids`` and ``num_prompt_tokens`` equal the complete Target
prefix.  The persistent path previously edited ``_all_token_ids`` and output
IDs but could leave the Request's logical prompt fields tied to the original
short prompt.

This probe makes the live Request represent the complete current prefix as
its prompt before calling the existing token synchronization code.  It is
opt-in only through the existing persistent-requeue branch; the patch itself
does not alter the default safe-reset behavior.

Every real patch operation creates a new full-file backup and manifest.
--dry-run changes nothing and creates no backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PROMPT_RESET_PROBE_V7"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_CACHED_STATE_PROBE_V6",
)

ANCHOR = (
    "        # Synchronize token IDs while the request is still in the persistent\n"
    "        # model-runner batch. Then keep only the common-prefix KV usable.\n"
    "        self._replace_tokens(prefix_token_ids)\n"
)

REPLACEMENT = (
    "        # Synchronize the live Request's logical prompt with the complete\n"
    "        # Target prefix before rebuilding runner-side token state.  A safe\n"
    "        # abort/re-add naturally gets these fields from tokens_input().\n"
    f"        # {MARKER}\n"
    "        self.prompt_token_ids = list(prefix_token_ids)\n"
    "        request.prompt_token_ids = list(prefix_token_ids)\n"
    "        request.num_prompt_tokens = len(prefix_token_ids)\n"
    "        if getattr(request, \"prompt_is_token_ids\", None) is not None:\n"
    "            request.prompt_is_token_ids = [True] * len(prefix_token_ids)\n"
    "\n"
    "        # Synchronize token IDs while the request is still in the persistent\n"
    "        # model-runner batch. Then keep only the common-prefix KV usable.\n"
    "        self._replace_tokens(prefix_token_ids)\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if REPLACEMENT in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete v7 probe "
            "is missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required prior patch marker(s) missing: "
            f"{', '.join(missing)}; no files were changed"
        )
    if source.count(ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one prompt-sync anchor, found "
            f"{source.count(ANCHOR)}; no files were changed"
        )
    return "to-patch"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit:
        backup_dir = Path(explicit).expanduser()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; choose a "
                "new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_persistent_requeue_prompt_reset_probe_v7."
            f"{timestamp()}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def write_atomic(target: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.pearl_stage5.",
        dir=str(target.parent),
        text=True,
    )
    try:
        os.chmod(temp_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(f"target file not found: {target}")

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"current state: {state}")
    if state == "patched":
        print("already patched; no files changed")
        return

    patched = original.replace(ANCHOR, REPLACEMENT, 1)
    if patched == original or MARKER not in patched or REPLACEMENT not in patched:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run passed; no files changed")
        return

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "target": str(target),
        "backup_file": str(backup_file),
        "original_sha256": sha256_bytes(original_bytes),
        "original_size": len(original_bytes),
        "marker": MARKER,
        "mode": "synchronize-live-request-prompt-fields-with-complete-prefix",
        "related_env": "PEARL_STAGE5_PERSISTENT_REQUEUE=1",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_atomic(target, patched, target.stat().st_mode)
    print(f"backup saved to: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add the v7 persistent-requeue prompt-field sync probe"
    )
    parser.add_argument(
        "--repo",
        default="/root/data/vllm-ascend-hust",
        help="Ascend bridge repository",
    )
    parser.add_argument(
        "--backup-dir",
        help="new backup directory to create; it must not already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source without changing files",
    )
    args = parser.parse_args()

    try:
        apply_patch(
            Path(args.repo).expanduser().resolve(),
            args.backup_dir,
            args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
