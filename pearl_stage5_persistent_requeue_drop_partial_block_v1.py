#!/usr/bin/env python3
"""Add an opt-in selective tail-block release for Stage-5 requeue.

The full-block trace showed that a valid 128-token block was followed by an
old unhashed partial block.  This patch keeps complete blocks associated with
the live request and releases only blocks beyond
``reusable_tokens // manager.block_size`` through the existing block-pool
API.

Enable the behavior only for the diagnostic run with:

    PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK=1

The default persistent-requeue path is unchanged.  Every real modification
creates a new full-file backup before the atomic replacement.
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
MARKER = "# PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1",
)

METHOD_ANCHOR = (
    "    def _requeue_request_preserve_kv(\n"
    "        self, prefix_token_ids: list[int]\n"
    "    ) -> None:\n"
)

HELPER = r'''    # PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1
    def _drop_partial_request_blocks(
        self,
        request: Any,
        reusable_tokens: int,
    ) -> None:
        """Keep aligned full blocks and return only stale tail blocks."""

        scheduler = self.core.scheduler
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        if not managers:
            raise RuntimeError(
                "Cannot locate single-type KV managers for selective tail release"
            )

        request_id = request.request_id
        kept_block_ids: list[list[int]] = []
        for manager_index, manager in enumerate(managers):
            req_to_blocks = getattr(manager, "req_to_blocks", None)
            cached_counts = getattr(manager, "num_cached_block", None)
            block_pool = getattr(manager, "block_pool", None)
            if req_to_blocks is None or cached_counts is None or block_pool is None:
                raise RuntimeError(
                    "KV manager %d lacks req_to_blocks, num_cached_block, or "
                    "block_pool" % manager_index
                )

            req_blocks = req_to_blocks.get(request_id)
            if req_blocks is None:
                raise RuntimeError(
                    "KV manager %d has no block list for request %r"
                    % (manager_index, request_id)
                )

            try:
                block_size = int(getattr(manager, "block_size"))
            except (TypeError, ValueError, AttributeError) as exc:
                raise RuntimeError(
                    "KV manager %d has no valid block_size" % manager_index
                ) from exc
            if block_size <= 0:
                raise RuntimeError(
                    "KV manager %d has invalid block_size=%r"
                    % (manager_index, block_size)
                )

            keep_count = min(len(req_blocks), reusable_tokens // block_size)
            stale_blocks = list(req_blocks[keep_count:])
            if stale_blocks:
                del req_blocks[keep_count:]

                free_blocks = getattr(block_pool, "free_blocks", None)
                if not callable(free_blocks):
                    raise RuntimeError(
                        "KV manager %d block pool has no free_blocks API"
                        % manager_index
                    )
                # The vLLM block pool expects tail blocks to be freed in
                # reverse allocation order.
                free_blocks(reversed(stale_blocks))

            cached_count = int(cached_counts.get(request_id, 0))
            if cached_count > keep_count:
                cached_counts[request_id] = keep_count

            kept_block_ids.append(
                [int(block.block_id) for block in req_blocks]
            )

        # Keep the in-process model-runner state aligned with the manager's
        # request bookkeeping before the persistent batch row is removed.
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner for tail release"
            )

        req_state = getattr(runner, "requests", {}).get(request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for selective tail release"
            )

        req_state.block_ids = tuple(list(group) for group in kept_block_ids)
        req_index = input_batch.req_id_to_index.get(request_id)
        block_table = getattr(input_batch, "block_table", None)
        if req_index is not None and block_table is not None:
            if not hasattr(block_table, "clear_row") or not hasattr(
                block_table, "append_row"
            ):
                raise RuntimeError(
                    "Draft input batch block table lacks clear_row/append_row"
                )
            block_table.clear_row(req_index)
            block_table.append_row(kept_block_ids, req_index)

        if os.environ.get("PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1] "
                f"request={request_id!r} reusable_tokens={reusable_tokens} "
                f"kept_block_ids={kept_block_ids}",
                flush=True,
            )
'''

CALL_OLD = (
    "        self._replace_tokens(prefix_token_ids)\n"
    "        # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5\n"
)
CALL_NEW = (
    "        self._replace_tokens(prefix_token_ids)\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK\", \"0\"\n"
    "        ) == \"1\":\n"
    "            self._drop_partial_request_blocks(\n"
    "                request, max(0, common_len - 1)\n"
    "            )\n"
    "        # PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        if HELPER in source and CALL_NEW in source:
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required markers missing: {missing}; "
            "no files were changed"
        )
    if source.count(METHOD_ANCHOR) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one requeue method anchor; "
            "no files were changed"
        )
    if source.count(CALL_OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one replace/call anchor, found "
            f"{source.count(CALL_OLD)}; no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    backup_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_persistent_requeue_drop_partial_block_v1."
        f"{timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
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
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: selectively release stale partial tail blocks")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    patched = original.replace(METHOD_ANCHOR, HELPER + "\n" + METHOD_ANCHOR, 1)
    patched = patched.replace(CALL_OLD, CALL_NEW, 1)
    if MARKER not in patched:
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
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
        "mode": "opt-in-selective-partial-tail-block-release",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK=1",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        write_atomic(target, patched, target.stat().st_mode)
    except Exception:
        shutil.copy2(backup_file, target)
        raise
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
