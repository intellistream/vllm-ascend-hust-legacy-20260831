#!/usr/bin/env python3
"""Safely rebase PEARL Stage-5 persistent request metadata.

The persistent Draft request is made to look like a fresh request whose
prompt is the current Target prefix.  This diagnostic step complements the
KV-block reset: it synchronizes prompt/output boundaries in Request,
CachedRequestState, and InputBatch before the next full-prefix recompute.
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
MARKER = "        # PEARL_STAGE5_REQUEST_REBASE_V1\n"

REQUEST_OUTPUT_BLOCK = (
    "        del request._output_token_ids[:]\n"
    "        request._output_token_ids.extend(\n"
    "            prefix_token_ids[len(self.prompt_token_ids) :]\n"
    "        )\n"
)
REQUEST_REPLACEMENT = (
    "        del request._output_token_ids[:]\n"
    + MARKER
    + "        # Rebase the live request to Target's current prefix.\n"
    + "        request.prompt_token_ids = list(prefix_token_ids)\n"
    + "        request.num_prompt_tokens = len(prefix_token_ids)\n"
)

REQ_STATE_BLOCK = (
    "        output_ids = prefix_token_ids[len(self.prompt_token_ids or []) :]\n"
    "        del req_state.output_token_ids[:]\n"
    "        req_state.output_token_ids.extend(output_ids)\n"
    "        req_state.num_computed_tokens = request.num_computed_tokens\n"
)
REQ_STATE_REPLACEMENT = (
    MARKER
    + "        # Keep the model-runner prompt boundary identical to a fresh\n"
    + "        # request created from this Target prefix.\n"
    + "        req_state.prompt_token_ids = list(prefix_token_ids)\n"
    + "        req_state.num_prompt_tokens = len(prefix_token_ids)\n"
    + "        del req_state.output_token_ids[:]\n"
    + "        req_state.num_computed_tokens = request.num_computed_tokens\n"
)

SYNC_ANCHOR = (
    "        input_batch.num_computed_tokens_cpu[req_index] = "
    "request.num_computed_tokens\n"
)
SYNC_INSERTION = (
    "        if hasattr(input_batch, \"num_prompt_tokens\"):\n"
    "            input_batch.num_prompt_tokens[req_index] = len(prefix_token_ids)\n"
)
PATCHED_PARTS = (
    REQUEST_REPLACEMENT,
    REQ_STATE_REPLACEMENT,
    SYNC_INSERTION,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if all(part in source for part in PATCHED_PARTS):
        return "patched"
    if MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: rebase marker is present but the complete patch "
            "is missing; no files were changed"
        )
    counts = (
        source.count(REQUEST_OUTPUT_BLOCK),
        source.count(REQ_STATE_BLOCK),
        source.count(SYNC_ANCHOR),
    )
    if counts != (1, 1, 1):
        raise RuntimeError(
            f"{TARGET_REL}: expected one request block, one runner-state "
            f"block, and one InputBatch anchor, found {counts}; "
            "no files were changed"
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
            f"{repo.name}.pearl_stage5_request_rebase_backup_v1.{timestamp()}"
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

    patched = (
        original.replace(REQUEST_OUTPUT_BLOCK, REQUEST_REPLACEMENT, 1)
        .replace(REQ_STATE_BLOCK, REQ_STATE_REPLACEMENT, 1)
        .replace(SYNC_ANCHOR, SYNC_INSERTION + SYNC_ANCHOR, 1)
    )
    if patched == original:
        raise RuntimeError("internal patch replacement made no change")
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
        "marker": MARKER.strip(),
        "mode": "persistent-request-target-prefix-rebase",
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
        description="Patch PEARL Stage-5 persistent request metadata rebase"
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

