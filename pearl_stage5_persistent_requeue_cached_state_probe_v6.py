#!/usr/bin/env python3
"""Add an opt-in cached-request-state reset probe.

The v5 fresh-block control still produced the shifted Draft proposal.  The
remaining difference from the safe abort-and-readd path is that the Draft
model runner keeps the old request state in ``runner.requests``.  This probe
removes that cached state after removing the input-batch row, so the next
WAITING schedule must construct a fresh model-runner request state for the
same request ID.

Enable together with the v5 fresh-block control:
    PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK=1
    PEARL_STAGE5_PERSISTENT_REQUEUE_RESET_CACHED_STATE=1

This is a diagnostic control, not the final optimization.  The default
persistent-requeue behavior remains unchanged.  Every real patch operation
creates a new full-file backup and manifest; --dry-run changes nothing and
creates no backup.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_CACHED_STATE_PROBE_V6"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5",
)

CALL_ANCHOR = (
    "        self._remove_request_from_model_runner_batch()\n"
    "\n"
    "        # Do not call _preempt_request(), finish_requests(), or\n"
)

CALL_REPLACEMENT = (
    "        self._remove_request_from_model_runner_batch()\n"
    f"        # {MARKER}\n"
    "        # The v5 control clears the old KV/block ownership.  This extra\n"
    "        # opt-in step also drops the cached request state, so the next\n"
    "        # WAITING schedule must build a fresh runner state for this ID.\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_RESET_CACHED_STATE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            self._drop_request_from_model_runner_cache()\n"
    "\n"
    "        # Do not call _preempt_request(), finish_requests(), or\n"
)

METHOD_ANCHOR = (
    "    def _requeue_request_preserve_kv(\n"
    "        self, prefix_token_ids: list[int]\n"
    "    ) -> None:\n"
)

NEW_HELPER = (
    "    def _drop_request_from_model_runner_cache(self) -> None:\n"
    "        \"\"\"Drop only the cached runner request state for this probe.\"\"\"\n"
    "        if self.request_id is None:\n"
    "            raise RuntimeError(\n"
    "                \"Cannot drop cached state for a missing Draft request\"\n"
    "            )\n"
    "\n"
    "        executor = getattr(self.core, \"model_executor\", None)\n"
    "        driver_worker = getattr(executor, \"driver_worker\", None)\n"
    "        runner = getattr(driver_worker, \"model_runner\", None)\n"
    "        if runner is None:\n"
    "            workers = getattr(executor, \"workers\", None)\n"
    "            if workers:\n"
    "                runner = getattr(workers[0], \"model_runner\", None)\n"
    "        if runner is None:\n"
    "            raise RuntimeError(\n"
    "                \"Cannot locate the in-process Draft model runner while \"\n"
    "                \"dropping cached request state\"\n"
    "            )\n"
    "\n"
    "        requests = getattr(runner, \"requests\", None)\n"
    "        if requests is None or not hasattr(requests, \"pop\"):\n"
    "            raise RuntimeError(\n"
    "                \"Draft model runner has no mutable request-state cache\"\n"
    "            )\n"
    "        requests.pop(self.request_id, None)\n"
        "        if os.environ.get(\n"
        "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
        "        ) == \"1\":\n"
        "            print(\n"
        "                \"[PEARL_STAGE5_PERSISTENT_REQUEUE_CACHED_STATE_PROBE_V6] \"\n"
        "                f\"dropped request-state cache for {self.request_id!r}\",\n"
        "                flush=True,\n"
        "            )\n"
    "\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if CALL_REPLACEMENT in source and NEW_HELPER in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete v6 probe "
            "is missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required prior patch marker(s) missing: "
            f"{', '.join(missing)}; no files were changed"
        )
    if source.count(CALL_ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one v3 batch-removal call anchor, found "
            f"{source.count(CALL_ANCHOR)}; no files were changed"
        )
    if source.count(METHOD_ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one persistent-requeue method anchor, "
            "no files were changed"
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
            f"{repo.name}.pearl_stage5_persistent_requeue_cached_state_probe_v6."
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

    patched = original.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
    patched = patched.replace(METHOD_ANCHOR, NEW_HELPER + METHOD_ANCHOR, 1)
    if (
        patched == original
        or MARKER not in patched
        or CALL_REPLACEMENT not in patched
        or NEW_HELPER not in patched
    ):
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
        "mode": "drop-cached-runner-request-state-after-fresh-block-control",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_RESET_CACHED_STATE=1",
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
        description="Add the v6 cached-runner-state reset diagnostic probe"
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
