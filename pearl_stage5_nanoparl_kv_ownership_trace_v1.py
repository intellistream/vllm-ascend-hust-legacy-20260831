#!/usr/bin/env python3
"""Trace physical Draft KV ownership across nano-PEARL in-place rollback.

This is an observability-only follow-up to
``pearl_stage5_nanoparl_inplace_rollback_v1.py``.  It does not free, allocate,
or reorder KV blocks.  Around the existing ``accepted_len``/``valid_len``
update it records the request's block IDs before and after the rollback.

The trace is enabled by ``PEARL_STAGE5_NANOPEARL_KV_TRACE=1`` (or the already
used ``PEARL_STAGE5_NANOPEARL_TRACE=1``).  A successful true in-place reuse
round should report ``same_manager_block_ids=True`` and
``same_runner_block_ids=True``.  If the active vLLM KV manager has a topology
that cannot be inspected, the trace reports ``available=False`` instead of
changing runtime behavior.

Only ``pearl_stage5_draft.py`` is backed up; no recursive repository copy is
performed, so build symlinks cannot make the backup fail.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path


TARGET_REL = Path("pearl_stage5_draft.py")
INPLACE_MARKER = "PEARL_STAGE5_NANOPEARL_INPLACE_ROLLBACK_V1"
TRACE_MARKER = "# PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1"


HELPER = r'''    # PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1
    def _nanoparl_kv_ownership_snapshot(self, request_id: str) -> dict:
        """Return non-mutating manager/runner block-ID snapshots."""

        snapshot = {
            "available": False,
            "reason": None,
            "manager_block_ids": [],
            "num_cached_blocks": [],
            "runner_block_ids": [],
        }

        def normalize_ids(values: Any) -> list[int]:
            if values is None:
                return []
            if not isinstance(values, (list, tuple)):
                values = [values]
            result: list[int] = []
            for value in values:
                if isinstance(value, (list, tuple)):
                    result.extend(normalize_ids(value))
                    continue
                block_id = getattr(value, "block_id", value)
                try:
                    result.append(int(block_id))
                except (TypeError, ValueError):
                    result.append(-1)
            return result

        scheduler = self.core.scheduler
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        managers = getattr(coordinator, "single_type_managers", None)
        if managers is None and getattr(kv_cache_manager, "req_to_blocks", None) is not None:
            managers = [kv_cache_manager]
        if not managers:
            snapshot["reason"] = "single_type_managers_unavailable"
        else:
            manager_seen = False
            for manager in managers:
                req_to_blocks = getattr(manager, "req_to_blocks", None)
                cached_counts = getattr(manager, "num_cached_block", None)
                if req_to_blocks is None or cached_counts is None:
                    snapshot["reason"] = "manager_bookkeeping_unavailable"
                    continue
                blocks = req_to_blocks.get(request_id)
                if blocks is None:
                    snapshot["reason"] = "request_block_list_unavailable"
                    continue
                manager_seen = True
                snapshot["manager_block_ids"].append(normalize_ids(blocks))
                try:
                    cached_count = int(cached_counts.get(request_id, 0))
                except (TypeError, ValueError):
                    cached_count = -1
                snapshot["num_cached_blocks"].append(cached_count)
            snapshot["available"] = manager_seen

        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is not None:
            requests = getattr(runner, "requests", None)
            req_state = requests.get(request_id) if requests is not None else None
            if req_state is not None:
                snapshot["runner_block_ids"] = normalize_ids(
                    getattr(req_state, "block_ids", None)
                )

        return snapshot
'''


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.nanoparl_kv_trace.",
        dir=str(path.parent),
        text=True,
    )
    try:
        os.chmod(temp_name, stat.S_IMODE(mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def transform(source: str) -> str:
    if TRACE_MARKER in source:
        compile(source, str(TARGET_REL), "exec")
        return source

    if INPLACE_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: the in-place rollback patch is not present; "
            "apply pearl_stage5_nanoparl_inplace_rollback_v1.py first; "
            "no files were changed"
        )

    method_start = source.find("    def _requeue_request_preserve_kv(")
    if method_start < 0:
        raise RuntimeError(
            f"{TARGET_REL}: cannot find _requeue_request_preserve_kv(); "
            "no files were changed"
        )

    next_method = source.find("\n    def ", method_start + 5)
    method_end = len(source) if next_method < 0 else next_method + 1
    method = source[method_start:method_end]

    branch_marker = f"        # {INPLACE_MARKER}\n"
    branch_start = method.find(branch_marker)
    if branch_start < 0:
        raise RuntimeError(
            f"{TARGET_REL}: cannot find the in-place rollback branch; "
            "no files were changed"
        )

    accepted_anchor = "            accepted_len = common_len\n"
    if method[branch_start:].count(accepted_anchor) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one accepted_len anchor in the in-place "
            "branch; no files were changed"
        )
    method = method.replace(
        accepted_anchor,
        "            kv_before = self._nanoparl_kv_ownership_snapshot(\n"
        "                request.request_id\n"
        "            )\n"
        + accepted_anchor,
        1,
    )

    sync_line = "            self._sync_model_runner_state(prefix_token_ids, request)\n"
    if method[branch_start:].count(sync_line) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one in-place runner sync anchor; "
            "no files were changed"
        )
    sync_replacement = (
        sync_line
        + "            kv_after = self._nanoparl_kv_ownership_snapshot(\n"
        + "                request.request_id\n"
        + "            )\n"
        + "            if (\n"
        + "                os.environ.get(\"PEARL_STAGE5_NANOPEARL_KV_TRACE\", \"0\")\n"
        + "                == \"1\"\n"
        + "                or os.environ.get(\"PEARL_STAGE5_NANOPEARL_TRACE\", \"0\")\n"
        + "                == \"1\"\n"
        + "            ):\n"
        + "                print(\n"
        + "                    \"[PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1] \"\n"
        + "                    + repr({\n"
        + "                        \"request\": request.request_id,\n"
        + "                        \"accepted_len\": accepted_len,\n"
        + "                        \"valid_len\": valid_len,\n"
        + "                        \"before\": kv_before,\n"
        + "                        \"after\": kv_after,\n"
        + "                        \"same_manager_block_ids\": (\n"
        + "                            kv_before[\"manager_block_ids\"]\n"
        + "                            == kv_after[\"manager_block_ids\"]\n"
        + "                        ),\n"
        + "                        \"same_runner_block_ids\": (\n"
        + "                            kv_before[\"runner_block_ids\"]\n"
        + "                            == kv_after[\"runner_block_ids\"]\n"
        + "                        ),\n"
        + "                    }),\n"
        + "                    flush=True,\n"
        + "                )\n"
    )
    method = method.replace(sync_line, sync_replacement, 1)

    transformed = source[:method_start] + HELPER + method + source[method_end:]
    compile(transformed, str(TARGET_REL), "exec")
    return transformed


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    transformed = transform(original)

    print(f"target: {target}")
    print("state: post-inplace rollback")
    print("change: non-mutating physical KV ownership trace")
    if transformed == original:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_kv_ownership_trace_v1.{_timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    shutil.copy2(target, backup_dir / TARGET_REL.name)
    print(f"backup: {backup_dir}")
    _write_atomic(target, transformed, target.stat().st_mode)
    print(f"patched: {target}")
    print(f"source_sha256_before: {_sha256(original_bytes)[:12]}")
    print(f"source_sha256_after: {_sha256(transformed.encode('utf-8'))[:12]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
