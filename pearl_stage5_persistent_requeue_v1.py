#!/usr/bin/env python3
"""Add an opt-in scheduler-aware persistent-KV requeue for PEARL Stage-5.

The safe Stage-5 path aborts the Draft Request and creates a new one. This
patch adds an experimental path that synchronizes the Target prefix, keeps the
request-owned KV blocks, moves the request from Scheduler.running to
Scheduler.waiting without calling the normal preemption/free path, and lets
the next scheduler step re-enter the normal waiting -> running path.

The experiment is disabled by default. Enable it with:
    PEARL_STAGE5_PERSISTENT_REQUEUE=1

Every real patch operation first saves a complete copy of the target file in a
new timestamped backup directory and writes a manifest. --dry-run does not
modify the repository or create a backup.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_V1"
LIFECYCLE_MARKER = "PEARL_STAGE5_REQUEST_LIFECYCLE_RESET_V1"

NEW_BRANCH = (
    "        if prefix_token_ids != self.committed_token_ids:\n"
    "            # PEARL_STAGE5_PERSISTENT_REQUEUE_V1\n"
    "            if os.environ.get(\n"
    "                \"PEARL_STAGE5_PERSISTENT_REQUEUE\", \"0\"\n"
    "            ) == \"1\":\n"
    "                self._requeue_request_preserve_kv(prefix_token_ids)\n"
    "                return\n"
    "            if os.environ.get(\n"
    "                \"PEARL_DRAFT_PERSISTENT_REUSE\", \"0\"\n"
    "            ) == \"1\":\n"
    "                self._replace_tokens(prefix_token_ids)\n"
    "                return\n"
    "            self._reset_request(prefix_token_ids)\n"
    "            return\n"
)

METHOD_ANCHOR = "    def _reset_request(self, prefix_token_ids: list[int]) -> None:\n"

NEW_METHOD = (
    "    def _requeue_request_preserve_kv(\n"
    "        self, prefix_token_ids: list[int]\n"
    "    ) -> None:\n"
    "        \"\"\"Requeue the live Draft request without freeing its KV blocks.\"\"\"\n"
    "        from vllm.v1.request import RequestStatus\n"
    "\n"
    "        if self.request_id is None:\n"
    "            raise RuntimeError(\"Cannot requeue a missing Draft request\")\n"
    "\n"
    "        request = self._request()\n"
    "        scheduler = self.core.scheduler\n"
    "        if request.status != RequestStatus.RUNNING:\n"
    "            raise RuntimeError(\n"
    "                \"Persistent requeue requires a RUNNING request, got \"\n"
    "                f\"{request.status!s} for {request.request_id!r}\"\n"
    "            )\n"
    "        if request not in scheduler.running:\n"
    "            raise RuntimeError(\n"
    "                \"Draft request is not present in scheduler.running: \"\n"
    "                f\"{request.request_id!r}\"\n"
    "            )\n"
    "\n"
    "        old_prefix = self.committed_token_ids\n"
    "        common_len = 0\n"
    "        for old_token, new_token in zip(old_prefix, prefix_token_ids):\n"
    "            if old_token != new_token:\n"
    "                break\n"
    "            common_len += 1\n"
    "\n"
    "        prompt_len = len(self.prompt_token_ids or [])\n"
    "        if common_len < prompt_len:\n"
    "            raise RuntimeError(\n"
    "                \"Target prefix diverged inside the Draft prompt: \"\n"
    "                f\"common_len={common_len} prompt_len={prompt_len}\"\n"
    "            )\n"
    "\n"
    "        # Synchronize token IDs while the request is still in the persistent\n"
    "        # model-runner batch. Then keep only the common-prefix KV usable.\n"
    "        self._replace_tokens(prefix_token_ids)\n"
    "        request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
    "\n"
    "        # Do not call _preempt_request(), finish_requests(), or\n"
    "        # kv_cache_manager.free(): all of those release request blocks.\n"
    "        scheduler.running.remove(request)\n"
    "        scheduler._inflight_prefills.discard(request)\n"
    "        request.status = RequestStatus.WAITING\n"
    "        scheduler.waiting.prepend_request(request)\n"
    "\n"
    "        if os.environ.get(\"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\") == \"1\":\n"
    "            print(\n"
    "                \"[PEARL_STAGE5_PERSISTENT_REQUEUE_V1] \"\n"
    "                \"request=%s common_len=%d num_computed_tokens=%d \"\n"
    "                \"status=%s running=%d waiting=%d\"\n"
    "                % (\n"
    "                    request.request_id,\n"
    "                    common_len,\n"
    "                    request.num_computed_tokens,\n"
    "                    request.status.name,\n"
    "                    len(scheduler.running),\n"
    "                    len(scheduler.waiting),\n"
    "                ),\n"
    "                flush=True,\n"
    "            )\n"
    "\n"
)

# Keep the exact lifecycle branch explicit: if the fork has drifted, abort
# instead of applying a fuzzy replacement to the wrong code.
EXPECTED_OLD_BRANCH = (
    "        if prefix_token_ids != self.committed_token_ids:\n"
    f"            # {LIFECYCLE_MARKER}\n"
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
    if MARKER in source:
        if NEW_BRANCH in source and NEW_METHOD in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )

    count = source.count(EXPECTED_OLD_BRANCH)
    if count != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one lifecycle-reset branch, found "
            f"{count}; no files were changed"
        )
    if source.count(METHOD_ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one _reset_request method anchor; "
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
            f"{repo.name}.pearl_stage5_persistent_requeue_backup_v1."
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

    patched = original.replace(EXPECTED_OLD_BRANCH, NEW_BRANCH, 1)
    patched = patched.replace(
        METHOD_ANCHOR,
        NEW_METHOD + METHOD_ANCHOR,
        1,
    )
    if patched == original or MARKER not in patched or NEW_METHOD not in patched:
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
        "mode": "opt-in-scheduler-aware-persistent-kv-requeue",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE=1",
        "direct_reuse_env": "PEARL_DRAFT_PERSISTENT_REUSE=1",
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
        description="Add opt-in persistent-KV requeue to PEARL Stage-5"
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
