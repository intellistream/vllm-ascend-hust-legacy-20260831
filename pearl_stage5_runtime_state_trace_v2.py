#!/usr/bin/env python3
"""Add opt-in runtime state tracing to the nano-PEARL Stage-5 Draft bridge.

This is a diagnostic patch only.  It does not change scheduling, KV
allocation, token IDs, or acceptance behavior unless the process is started
with ``PEARL_DRAFT_STATE_TRACE=1``; without that variable the inserted trace
calls return immediately.

The patch records the live state at three boundaries:

* before and after ``_replace_tokens`` synchronizes Target's prefix;
* immediately before and after the in-process Draft ``get_output`` step;
* immediately before returning a newly sampled Draft token.

Every real patch operation creates a new full-file backup and a manifest.
Run ``--dry-run`` before the real patch.  The target file is intentionally
limited to ``pearl_stage5_draft.py``.
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
MARKER = "PEARL_STAGE5_RUNTIME_STATE_TRACE_V2"

METHOD_ANCHOR = (
    "    def _add_request(self, prefix_token_ids: list[int]) -> None:\n"
)

TRACE_METHOD = r'''    # PEARL_STAGE5_RUNTIME_STATE_TRACE_V2
    def _trace_state(
        self,
        label: str,
        request: Any | None = None,
    ) -> None:
        """Print compact scheduler/runner state when explicitly enabled."""

        if os.environ.get("PEARL_DRAFT_STATE_TRACE") != "1":
            return

        def _safe_len(value: Any) -> int | str:
            if value is None:
                return 0
            try:
                return len(value)
            except Exception:
                return "?"

        def _short(value: Any, limit: int = 16) -> Any:
            if value is None:
                return None
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "tolist"):
                    value = value.tolist()
            except Exception as exc:
                return f"<{type(value).__name__}: {type(exc).__name__}: {exc}>"
            try:
                if isinstance(value, (list, tuple)):
                    if len(value) > limit:
                        return list(value[:limit]) + [
                            f"...(+{len(value) - limit})"
                        ]
                    return list(value)
                return value
            except Exception as exc:
                return f"<{type(value).__name__}: {type(exc).__name__}: {exc}>"

        def _get(obj: Any, name: str, default: Any = None) -> Any:
            if obj is None:
                return default
            try:
                return getattr(obj, name, default)
            except Exception as exc:
                return f"<{name}: {type(exc).__name__}: {exc}>"

        def _ids(value: Any, limit: int = 12) -> Any:
            if value is None:
                return []
            if isinstance(value, dict):
                value = value.keys()
            try:
                items = list(value)
            except Exception:
                return _short(value, limit)
            result = []
            for item in items[:limit]:
                result.append(_get(item, "request_id", item))
            if len(items) > limit:
                result.append(f"...(+{len(items) - limit})")
            return result

        def _block_summary(value: Any) -> Any:
            if value is None:
                return None
            try:
                groups = list(value)
            except Exception:
                return _short(value)
            lengths = []
            heads = []
            for group in groups[:8]:
                try:
                    block_ids = list(group)
                    lengths.append(len(block_ids))
                    heads.append(block_ids[:8])
                except Exception:
                    lengths.append("?")
                    heads.append(_short(group, 8))
            return {
                "groups": len(groups),
                "lengths": lengths,
                "heads": heads,
            }

        def _batch_value(batch: Any, field: str, index: Any) -> Any:
            if batch is None or index is None:
                return None
            try:
                values = getattr(batch, field)
                return _short(values[index])
            except Exception as exc:
                return f"<{field}: {type(exc).__name__}: {exc}>"

        if request is None:
            try:
                request = self._request()
            except Exception:
                request = None

        core = _get(self, "core")
        scheduler = _get(core, "scheduler")
        scheduler_requests = _get(scheduler, "requests", {})
        request_id = _get(self, "request_id")

        executor = _get(core, "model_executor")
        driver_worker = _get(executor, "driver_worker")
        runner = _get(driver_worker, "model_runner")
        if runner is None:
            workers = _get(executor, "workers", [])
            try:
                if workers:
                    runner = _get(workers[0], "model_runner")
            except Exception:
                pass

        runner_requests = _get(runner, "requests", {})
        req_state = None
        try:
            req_state = runner_requests.get(request_id)
        except Exception:
            pass

        input_batch = _get(runner, "input_batch")
        req_index = None
        try:
            req_index = _get(input_batch, "req_id_to_index", {}).get(request_id)
        except Exception:
            pass

        request_all_ids = _get(request, "_all_token_ids", [])
        request_output_ids = _get(request, "_output_token_ids", [])
        req_state_output_ids = _get(req_state, "output_token_ids", [])
        payload = {
            "label": label,
            "request_id": request_id,
            "committed_len": _safe_len(_get(self, "committed_token_ids", [])),
            "committed_head": _short(
                _get(self, "committed_token_ids", []), 12
            ),
            "request": {
                "status": _short(_get(request, "status")),
                "num_prompt_tokens": _short(
                    _get(request, "num_prompt_tokens")
                ),
                "num_tokens": _short(_get(request, "num_tokens")),
                "num_computed_tokens": _short(
                    _get(request, "num_computed_tokens")
                ),
                "all_len": _safe_len(request_all_ids),
                "all_head": _short(request_all_ids, 12),
                "prompt_len": _safe_len(_get(request, "prompt_token_ids", [])),
                "output_len": _safe_len(request_output_ids),
                "output_head": _short(request_output_ids, 12),
                "spec_len": _safe_len(_get(request, "spec_token_ids", [])),
                "num_output_placeholders": _short(
                    _get(request, "num_output_placeholders")
                ),
                "skip_reading_prefix_cache": _short(
                    _get(request, "skip_reading_prefix_cache")
                ),
                "block_hashes_len": _safe_len(
                    _get(request, "block_hashes", [])
                ),
            },
            "req_state": {
                "num_prompt_tokens": _short(
                    _get(req_state, "num_prompt_tokens")
                ),
                "num_tokens": _short(_get(req_state, "num_tokens")),
                "num_computed_tokens": _short(
                    _get(req_state, "num_computed_tokens")
                ),
                "prompt_len": _safe_len(
                    _get(req_state, "prompt_token_ids", [])
                ),
                "output_len": _safe_len(req_state_output_ids),
                "output_head": _short(req_state_output_ids, 12),
                "block_ids": _block_summary(_get(req_state, "block_ids")),
            },
            "input_batch": {
                "req_index": _short(req_index),
                "num_prompt_tokens": _batch_value(
                    input_batch, "num_prompt_tokens", req_index
                ),
                "num_tokens_no_spec": _batch_value(
                    input_batch, "num_tokens_no_spec", req_index
                ),
                "num_tokens": _batch_value(
                    input_batch, "num_tokens", req_index
                ),
                "num_computed_tokens_cpu": _batch_value(
                    input_batch, "num_computed_tokens_cpu", req_index
                ),
                "token_ids_head": _batch_value(
                    input_batch, "token_ids_cpu", req_index
                ),
                "is_token_ids_head": _batch_value(
                    input_batch, "is_token_ids", req_index
                ),
                "spec_token_ids": _batch_value(
                    input_batch, "spec_token_ids", req_index
                ),
            },
            "scheduler": {
                "request_ids": _ids(scheduler_requests),
                "running": _ids(_get(scheduler, "running")),
                "waiting": _ids(_get(scheduler, "waiting")),
                "prev_step_scheduled_req_ids": _short(
                    _get(scheduler, "prev_step_scheduled_req_ids")
                ),
                "finished_req_ids": _short(
                    _get(scheduler, "finished_req_ids")
                ),
            },
        }
        print(
            f"[{MARKER}] "
            + repr(payload),
            flush=True,
        )
'''


REPLACE_BEFORE_ANCHOR = (
    "        request = self._request()\n"
    "        assert self.prompt_token_ids is not None\n"
)
REPLACE_BEFORE_INSERTION = (
    REPLACE_BEFORE_ANCHOR
    + "        self._trace_state(\"replace.before\", request=request)\n"
)

REPLACE_AFTER_ANCHOR = (
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)
REPLACE_AFTER_INSERTION = (
    REPLACE_AFTER_ANCHOR
    + "        self._trace_state(\"replace.after\", request=request)\n"
)

STEP_GET_OUTPUT_ANCHOR = "            outputs = self.core_client.get_output()\n"
STEP_GET_OUTPUT_INSERTION = (
    "            self._trace_state(\"step.before_get_output\")\n"
    + STEP_GET_OUTPUT_ANCHOR
    + "            self._trace_state(\"step.after_get_output\")\n"
)

STEP_RETURN_ANCHOR = "                    return int(output.new_token_ids[-1])\n"
STEP_RETURN_INSERTION = (
    "                    self._trace_state(\"step.return\")\n"
    + STEP_RETURN_ANCHOR
)

PATCH_PARTS = (
    "    def _trace_state(\n",
    "        self._trace_state(\"replace.before\", request=request)\n",
    "        self._trace_state(\"replace.after\", request=request)\n",
    "            self._trace_state(\"step.before_get_output\")\n",
    "            self._trace_state(\"step.after_get_output\")\n",
    "                    self._trace_state(\"step.return\")\n",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> str:
    if all(part in source for part in PATCH_PARTS):
        return "patched"
    if MARKER in source:
        raise RuntimeError(
            f"{TARGET_REL}: trace marker is present but the complete V2 "
            "trace patch is missing; no files were changed"
        )
    counts = {
        "method_anchor": source.count(METHOD_ANCHOR),
        "replace_before": source.count(REPLACE_BEFORE_ANCHOR),
        "replace_after": source.count(REPLACE_AFTER_ANCHOR),
        "step_get_output": source.count(STEP_GET_OUTPUT_ANCHOR),
        "step_return": source.count(STEP_RETURN_ANCHOR),
    }
    if counts != {
        "method_anchor": 1,
        "replace_before": 1,
        "replace_after": 1,
        "step_get_output": 1,
        "step_return": 1,
    }:
        raise RuntimeError(
            f"{TARGET_REL}: expected stable Stage-5 anchors, found {counts}; "
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
            f"{repo.name}.pearl_stage5_runtime_state_trace_backup_v2."
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

    patched = original
    patched = patched.replace(
        METHOD_ANCHOR,
        TRACE_METHOD + "\n" + METHOD_ANCHOR,
        1,
    )
    patched = patched.replace(
        REPLACE_BEFORE_ANCHOR,
        REPLACE_BEFORE_INSERTION,
        1,
    )
    patched = patched.replace(
        REPLACE_AFTER_ANCHOR,
        REPLACE_AFTER_INSERTION,
        1,
    )
    patched = patched.replace(
        STEP_GET_OUTPUT_ANCHOR,
        STEP_GET_OUTPUT_INSERTION,
        1,
    )
    patched = patched.replace(
        STEP_RETURN_ANCHOR,
        STEP_RETURN_INSERTION,
        1,
    )
    if patched == original:
        raise RuntimeError("internal patch replacement made no change")
    if not all(part in patched for part in PATCH_PARTS):
        raise RuntimeError(
            "internal patch validation failed: one or more trace blocks "
            "were not inserted; no files were changed"
        )
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
        "mode": "opt-in-runtime-state-trace-only",
        "trace_env": "PEARL_DRAFT_STATE_TRACE=1",
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
        description="Patch PEARL Stage-5 Draft runtime state tracing"
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
