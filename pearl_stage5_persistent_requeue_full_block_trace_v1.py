#!/usr/bin/env python3
"""Add a compact, opt-in trace for the Stage-5 full-block requeue path.

This is a diagnostic-only patch.  It does not alter scheduling, token IDs,
KV allocation, or the requeue decision.  The trace is emitted only when
``PEARL_STAGE5_FULL_BLOCK_TRACE=1`` is present in the Draft worker.

Every non-dry-run invocation creates a new full-file backup immediately
before replacing the target.  Existing backup directories are never reused.
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
MARKER = "# PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3",
)

METHOD_ANCHOR = "    def _add_request(self, prefix_token_ids: list[int]) -> None:\n"

TRACE_METHOD = r'''    # PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_BLOCK_TRACE_V1
    def _trace_full_block_state(
        self,
        label: str,
        request: Any | None = None,
    ) -> None:
        """Print compact request/runner/KV state when explicitly enabled."""

        if os.environ.get("PEARL_STAGE5_FULL_BLOCK_TRACE", "0") != "1":
            return

        def _get(obj: Any, name: str, default: Any = None) -> Any:
            if obj is None:
                return default
            try:
                return getattr(obj, name, default)
            except Exception as exc:
                return f"<{name}:{type(exc).__name__}:{exc}>"

        def _len(value: Any) -> int | str:
            if value is None:
                return 0
            try:
                return len(value)
            except Exception:
                return "?"

        def _short(value: Any, limit: int = 12) -> Any:
            if value is None:
                return None
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
            except Exception as exc:
                return f"<{type(value).__name__}:{type(exc).__name__}:{exc}>"
            if isinstance(value, (list, tuple)):
                value = list(value)
                if len(value) > limit:
                    return value[:limit] + [f"...(+{len(value) - limit})"]
                return value
            return value

        def _request_ids(value: Any) -> Any:
            if value is None:
                return []
            try:
                items = list(value.keys()) if isinstance(value, dict) else list(value)
            except Exception:
                return "?"
            result = []
            for item in items[:8]:
                result.append(_get(item, "request_id", item))
            if len(items) > 8:
                result.append(f"...(+{len(items) - 8})")
            return result

        def _blocks(value: Any) -> Any:
            if value is None:
                return None
            try:
                value = getattr(value, "blocks", value)
            except Exception:
                pass
            try:
                groups = list(value)
            except Exception:
                return _short(value)
            summary = []
            for group in groups[:8]:
                try:
                    block_ids = list(group)
                    summary.append({
                        "len": len(block_ids),
                        "head": block_ids[:8],
                    })
                except Exception:
                    summary.append(_short(group, 8))
            if len(groups) > 8:
                summary.append({"groups_more": len(groups) - 8})
            return summary

        def _batch_value(batch: Any, field: str, index: Any) -> Any:
            if batch is None or index is None:
                return None
            try:
                return _short(getattr(batch, field)[index])
            except Exception as exc:
                return f"<{field}:{type(exc).__name__}:{exc}>"

        if request is None:
            try:
                request = self._request()
            except Exception:
                request = None

        request_id = _get(self, "request_id")
        core = _get(self, "core")
        scheduler = _get(core, "scheduler")
        executor = _get(core, "model_executor")
        runner = _get(_get(executor, "driver_worker"), "model_runner")
        if runner is None:
            workers = _get(executor, "workers", [])
            try:
                if workers:
                    runner = _get(workers[0], "model_runner")
            except Exception:
                pass

        runner_requests = _get(runner, "requests", {})
        try:
            req_state = runner_requests.get(request_id)
        except Exception:
            req_state = None
        input_batch = _get(runner, "input_batch")
        try:
            req_index = _get(input_batch, "req_id_to_index", {}).get(request_id)
        except Exception:
            req_index = None

        kv_cache_manager = _get(scheduler, "kv_cache_manager")
        try:
            owned_blocks = kv_cache_manager.get_blocks(request_id)
        except Exception:
            owned_blocks = None

        request_all = _get(request, "_all_token_ids", [])
        request_output = _get(request, "_output_token_ids", [])
        req_state_output = _get(req_state, "output_token_ids", [])
        payload = {
            "label": label,
            "request_id": request_id,
            "request": {
                "status": _short(_get(request, "status")),
                "num_computed_tokens": _short(
                    _get(request, "num_computed_tokens")
                ),
                "num_tokens": _short(_get(request, "num_tokens")),
                "num_prompt_tokens": _short(
                    _get(request, "num_prompt_tokens")
                ),
                "all_len": _len(request_all),
                "output_len": _len(request_output),
            },
            "runner": {
                "num_computed_tokens": _short(
                    _get(req_state, "num_computed_tokens")
                ),
                "output_len": _len(req_state_output),
                "block_ids": _blocks(_get(req_state, "block_ids")),
            },
            "input_batch": {
                "req_index": _short(req_index),
                "num_computed_tokens_cpu": _batch_value(
                    input_batch, "num_computed_tokens_cpu", req_index
                ),
                "num_tokens_no_spec": _batch_value(
                    input_batch, "num_tokens_no_spec", req_index
                ),
                "num_tokens": _batch_value(input_batch, "num_tokens", req_index),
                "token_ids_head": _batch_value(
                    input_batch, "token_ids_cpu", req_index
                ),
            },
            "kv_manager_blocks": _blocks(owned_blocks),
            "scheduler": {
                "running": _request_ids(_get(scheduler, "running")),
                "waiting": _request_ids(_get(scheduler, "waiting")),
            },
        }
        print(f"[{MARKER}] " + repr(payload), flush=True)
'''

BEFORE_REPLACE_OLD = (
    "        # Synchronize token IDs while the request is still in the persistent\n"
    "        # model-runner batch. Then keep only the common-prefix KV usable.\n"
    "        self._replace_tokens(prefix_token_ids)\n"
)
BEFORE_REPLACE_NEW = (
    "        # Synchronize token IDs while the request is still in the persistent\n"
    "        # model-runner batch. Then keep only the common-prefix KV usable.\n"
    "        self._trace_full_block_state(\n"
    "            \"full_requeue.before_replace\", request=request\n"
    "        )\n"
    "        self._replace_tokens(prefix_token_ids)\n"
)

AFTER_SYNC_OLD = (
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
    "        # PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3\n"
)
AFTER_SYNC_NEW = (
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
    "        self._trace_full_block_state(\n"
    "            \"full_requeue.after_sync\", request=request\n"
    "        )\n"
    "        # PEARL_STAGE5_PERSISTENT_REQUEUE_BATCH_REMOVE_V3\n"
)

AFTER_ENQUEUE_OLD = (
    "        scheduler.waiting.prepend_request(request)\n"
    "\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
    "        ) == \"1\":\n"
)
AFTER_ENQUEUE_NEW = (
    "        scheduler.waiting.prepend_request(request)\n"
    "\n"
    "        if os.environ.get(\"PEARL_STAGE5_FULL_BLOCK_TRACE\", \"0\") == \"1\":\n"
    "            self._pearl_full_block_trace_pending = True\n"
    "            self._trace_full_block_state(\n"
    "                \"full_requeue.after_enqueue\", request=request\n"
    "            )\n"
    "\n"
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
    "        ) == \"1\":\n"
)

STEP_GET_OUTPUT_OLD = "            outputs = self.core_client.get_output()\n"
STEP_GET_OUTPUT_NEW = (
    "            trace_pending = bool(\n"
    "                getattr(self, \"_pearl_full_block_trace_pending\", False)\n"
    "            )\n"
    "            if trace_pending:\n"
    "                self._trace_full_block_state(\n"
    "                    \"full_requeue.before_get_output\"\n"
    "                )\n"
    "            outputs = self.core_client.get_output()\n"
    "            if trace_pending:\n"
    "                self._trace_full_block_state(\n"
    "                    \"full_requeue.after_get_output\"\n"
    "                )\n"
    "                self._pearl_full_block_trace_pending = False\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if MARKER in source:
        required = (
            "    def _trace_full_block_state(\n",
            "full_requeue.before_replace",
            "full_requeue.after_sync",
            "full_requeue.after_enqueue",
            "full_requeue.before_get_output",
            "full_requeue.after_get_output",
        )
        if all(item in source for item in required):
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete trace is missing; "
            "no files were changed"
        )
    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required markers missing: {missing}; no files were changed"
        )
    checks = {
        "method anchor": source.count(METHOD_ANCHOR),
        "replace anchor": source.count(BEFORE_REPLACE_OLD),
        "sync anchor": source.count(AFTER_SYNC_OLD),
        "enqueue anchor": source.count(AFTER_ENQUEUE_OLD),
        "get_output anchor": source.count(STEP_GET_OUTPUT_OLD),
    }
    bad = {name: count for name, count in checks.items() if count != 1}
    if bad:
        raise RuntimeError(
            f"{TARGET_REL}: expected one exact anchor each, found {bad}; "
            "no files were changed"
        )
    return "pre"


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    backup_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_persistent_requeue_full_block_trace_v1.{timestamp()}"
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
    print("change: opt-in compact trace for full-block persistent requeue")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    patched = original.replace(METHOD_ANCHOR, TRACE_METHOD + "\n" + METHOD_ANCHOR, 1)
    patched = patched.replace(BEFORE_REPLACE_OLD, BEFORE_REPLACE_NEW, 1)
    patched = patched.replace(AFTER_SYNC_OLD, AFTER_SYNC_NEW, 1)
    patched = patched.replace(AFTER_ENQUEUE_OLD, AFTER_ENQUEUE_NEW, 1)
    patched = patched.replace(STEP_GET_OUTPUT_OLD, STEP_GET_OUTPUT_NEW, 1)
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
        "mode": "diagnostic-only-opt-in-full-block-state-trace",
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
