#!/usr/bin/env python3
"""Add an opt-in fresh-block control probe for persistent requeue.

The v3 test fixed the persistent model-runner batch lifecycle, while the v4
probe showed that forcing ``num_computed_tokens=0`` still produced the shifted
Draft proposal when the old request-owned KV block was retained.  This probe
keeps the same request and scheduler requeue path but releases the old KV
blocks and clears the model-runner block table before the next schedule.

Enable only for the control run with:
    PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK=1

This is a diagnostic control, not the final nano-PEARL optimization.  The
normal persistent-requeue behavior remains unchanged unless the environment
variable is set.  Every real patch operation creates a new full-file backup
and manifest; --dry-run changes nothing and creates no backup.
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
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_BLOCK_PROBE_V4",
)

V4_ANCHOR = (
    "        # PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_BLOCK_PROBE_V4\n"
    "        # A partial KV block is not treated as reusable in this probe.\n"
    "        # Keep the request-owned block and force the scheduler/model runner\n"
    "        # to recompute the whole current prefix into it.\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            request.num_computed_tokens = 0\n"
    "        else:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)

V5_REPLACEMENT = (
    "        # " + MARKER + "\n"
    "        # Release the old request-owned KV only for the explicit fresh-\n"
    "        # block control.  The request object and scheduler lifecycle stay\n"
    "        # alive, but the next schedule must allocate a new block.\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK\", \"0\"\n"
    "        ) == \"1\":\n"
    "            self._release_request_kv_for_fresh_recompute(request)\n"
    "            request.num_computed_tokens = 0\n"
    "        elif os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            request.num_computed_tokens = 0\n"
    "        else:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)

METHOD_ANCHOR = (
    "    def _requeue_request_preserve_kv(\n"
    "        self, prefix_token_ids: list[int]\n"
    "    ) -> None:\n"
)

NEW_HELPER = (
    "    def _release_request_kv_for_fresh_recompute(self, request: Any) -> None:\n"
    "        \"\"\"Release old KV and clear runner-side block ownership.\"\"\"\n"
    "        scheduler = self.core.scheduler\n"
    "        kv_cache_manager = getattr(scheduler, \"kv_cache_manager\", None)\n"
    "        if kv_cache_manager is None:\n"
    "            raise RuntimeError(\n"
    "                \"Cannot locate the Draft scheduler KV cache manager\"\n"
    "            )\n"
    "        kv_cache_manager.free(request)\n"
    "\n"
    "        if self.request_id is None:\n"
    "            raise RuntimeError(\n"
    "                \"Cannot clear KV state for a missing Draft request\"\n"
    "            )\n"
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
    "                \"clearing fresh-recompute KV state\"\n"
    "            )\n"
    "\n"
    "        req_state = getattr(runner, \"requests\", {}).get(self.request_id)\n"
    "        input_batch = getattr(runner, \"input_batch\", None)\n"
    "        if req_state is None or input_batch is None:\n"
    "            raise RuntimeError(\n"
    "                \"Draft model runner has no cached state for fresh \"\n"
    "                f\"recompute request {self.request_id!r}\"\n"
    "            )\n"
    "\n"
    "        block_ids = getattr(req_state, \"block_ids\", None)\n"
    "        if block_ids is not None:\n"
    "            req_state.block_ids = tuple([] for _ in block_ids)\n"
    "        req_index = input_batch.req_id_to_index.get(self.request_id)\n"
    "        block_table = getattr(input_batch, \"block_table\", None)\n"
    "        if req_index is not None and block_table is not None:\n"
    "            if not hasattr(block_table, \"clear_row\"):\n"
    "                raise RuntimeError(\n"
    "                    \"Draft input batch block table has no clear_row API\"\n"
    "                )\n"
    "            block_table.clear_row(req_index)\n"
    "\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if V5_REPLACEMENT in source and NEW_HELPER in source:
            return "patched"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete v5 probe "
            "is missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required prior patch marker(s) missing: "
            f"{', '.join(missing)}; no files were changed"
        )
    if source.count(V4_ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one v4 computed-token anchor, found "
            f"{source.count(V4_ANCHOR)}; no files were changed"
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
            f"{repo.name}.pearl_stage5_persistent_requeue_fresh_block_probe_v5."
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

    patched = original.replace(V4_ANCHOR, V5_REPLACEMENT, 1)
    patched = patched.replace(METHOD_ANCHOR, NEW_HELPER + METHOD_ANCHOR, 1)
    if (
        patched == original
        or MARKER not in patched
        or V5_REPLACEMENT not in patched
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
        "mode": "same-request-requeue-with-fresh-kv-block-control",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK=1",
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
        description="Add the v5 fresh-block persistent-requeue control probe"
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
