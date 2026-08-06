#!/usr/bin/env python3
"""Make Stage-5 Request lifecycle reset the safe default.

The persistent Draft bridge previously edited a Request that was already in
the scheduler's RUNNING state.  The 586 experiment showed that correctness is
restored when the old Request is aborted and a fresh Request is added, which
re-enters the normal WAITING -> RUNNING prefill lifecycle.

This patch changes only the prefix-change branch in ``sync_prefix``:

* default: call the existing ``_reset_request`` helper;
* opt-in experiment: set ``PEARL_DRAFT_PERSISTENT_REUSE=1`` to retain the
  old direct mutation path.

The model and Draft engine process remain alive; only the scheduler Request
and its KV state are recreated.  Every real patch operation first saves a
full copy of the target file in a new backup directory and writes a manifest.
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
MARKER = "PEARL_STAGE5_REQUEST_LIFECYCLE_RESET_V1"

DIRECT_BRANCH = (
    "        if prefix_token_ids != self.committed_token_ids:\n"
    "            self._replace_tokens(prefix_token_ids)\n"
)
PATCHED_BRANCH = (
    "        if prefix_token_ids != self.committed_token_ids:\n"
    f"            # {MARKER}\n"
    "            if os.environ.get(\n"
    "                \"PEARL_DRAFT_PERSISTENT_REUSE\", \"0\"\n"
    "            ) != \"1\":\n"
    "                self._reset_request(prefix_token_ids)\n"
    "                return\n"
    "            self._replace_tokens(prefix_token_ids)\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if PATCHED_BRANCH in source:
        return "patched"
    if MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: lifecycle-reset marker is present but the "
            "complete block is missing; no files were changed"
        )
    count = source.count(DIRECT_BRANCH)
    if count != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one direct prefix-sync branch, found "
            f"{count}; no files were changed"
        )
    return "to-patch"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    if explicit:
        backup_dir = Path(explicit).expanduser()
        if backup_dir.exists():
            raise FileExistsError(
                f"backup directory already exists: {backup_dir}; "
                "choose a new path so no backup is overwritten"
            )
    else:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_request_lifecycle_reset_backup_v1."
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

    patched = original.replace(DIRECT_BRANCH, PATCHED_BRANCH, 1)
    if patched == original or PATCHED_BRANCH not in patched:
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
        "mode": "safe-request-lifecycle-reset-by-default",
        "unsafe_reuse_opt_in_env": "PEARL_DRAFT_PERSISTENT_REUSE=1",
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
        description="Make PEARL Stage-5 Request lifecycle reset the default"
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
