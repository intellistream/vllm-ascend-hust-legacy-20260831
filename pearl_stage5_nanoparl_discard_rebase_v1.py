#!/usr/bin/env python3
"""Patch nano-PEARL so stale optimistic Draft state is explicitly rebased.

A rejected PRE/POST-VERIFY lookahead must not leave the persistent Draft
request at Target-prefix-plus-draft.  This patch adds an ordered rebase_batch
RPC.  The controller waits for a stale request, resets each affected Draft
slot to Target's authoritative prefix, and only then requests new tokens.

Only four Stage-5 source files are backed up; this avoids recursive copying of
dynamic build symlinks under csrc/build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path


MARKERS = {
    "pearl_stage5_nanoparl_runtime_v1.py":
        "# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_RUNTIME_V1",
    "pearl_stage5_nanoparl_proposer_v3.py":
        "# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_PROPOSER_V1",
    "pearl_stage5_worker.py":
        "# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_WORKER_V1",
    "pearl_stage5_draft.py":
        "# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_DRAFT_V1",
}


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(source: str, method: str, next_method: str, replacement: str) -> str:
    start = source.find(f"    def {method}")
    end = source.find(f"    def {next_method}", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method}: method boundary not found; no files were changed"
        )
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def transform_runtime(source: str) -> str:
    marker = MARKERS["pearl_stage5_nanoparl_runtime_v1.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    source = replace_once(
        source,
        "BatchRequestFn = Callable[[list[dict]], list[list[int]]]\n"
        "TraceFn = Callable[[str], None]\n",
        "BatchRequestFn = Callable[[list[dict]], list[list[int]]]\n"
        "RebaseBatchFn = Callable[[list[dict]], None]\n"
        "TraceFn = Callable[[str], None]\n",
        "runtime callback type anchors",
    )
    source = replace_once(
        source,
        "        request_batch: BatchRequestFn,\n"
        "        *,\n"
        "        trace: TraceFn | None = None,\n"
        "    ) -> None:\n",
        "        request_batch: BatchRequestFn,\n"
        "        *,\n"
        "        trace: TraceFn | None = None,\n"
        "        rebase_batch: RebaseBatchFn | None = None,\n"
        "    ) -> None:\n",
        "runtime constructor anchor",
    )
    source = replace_once(
        source,
        "        self._request_batch = request_batch\n"
        "        self._trace_fn = trace\n",
        "        self._request_batch = request_batch\n"
        "        self._rebase_batch = rebase_batch\n"
        "        self._rebase_on_discard = (\n"
        "            os.environ.get(\n"
        "                \"PEARL_STAGE5_NANOPEARL_SAFE_DISCARD_REBASE\", \"1\"\n"
        "            )\n"
        "            != \"0\"\n"
        "        )\n"
        "        self._trace_fn = trace\n",
        "runtime rebase state anchor",
    )
    new_drain = """    def _drain_stale(self, pending: _PendingPrefetch) -> None:
        # The transport is ordered.  Wait before sending rebase_batch so
        # response/round pairs cannot be mixed.
        try:
            pending.future.result()
        except Exception as exc:
            self._trace(f"prefetch_discard_error={exc!r}")

    def _rebase_current(self, current: tuple[DraftRequest, ...]) -> None:
        if not self._rebase_on_discard or self._rebase_batch is None:
            return
        self._trace(
            "discard_rebase_start "
            f"round={self.round_id} batch={len(current)}"
        )
        try:
            self._rebase_batch(self._to_wire(current))
        except Exception as exc:
            self._trace(f"discard_rebase_error={exc!r}")
            raise
        self._trace(
            "discard_rebase_done "
            f"round={self.round_id} batch={len(current)}"
        )
"""
    source = replace_method(source, "_drain_stale(", "_start_prefetch(", new_drain)
    source = replace_once(
        source,
        "            except Exception as exc:\n"
        "                self._trace(f\"post_verify_prefetch_error={exc!r}\")\n"
        "                results = self._call_batch(current)\n"
        "                self.mode = PearlMode.PRE_VERIFY\n",
        "            except Exception as exc:\n"
        "                self._trace(f\"post_verify_prefetch_error={exc!r}\")\n"
        "                self._rebase_current(current)\n"
        "                results = self._call_batch(current)\n"
        "                self.mode = PearlMode.PRE_VERIFY\n",
        "runtime failed-prefetch rebase anchor",
    )
    source = replace_once(
        source,
        "            if pending is not None:\n"
        "                self._drain_stale(pending)\n"
        "                self._trace(\n"
        "                    \"pre_verify discard_prefetch \"\n"
        "                    f\"round={self.round_id} batch={len(current)}\"\n"
        "                )\n"
        "            results = self._call_batch(current)\n",
        "            if pending is not None:\n"
        "                self._drain_stale(pending)\n"
        "                self._trace(\n"
        "                    \"pre_verify discard_prefetch \"\n"
        "                    f\"round={self.round_id} batch={len(current)}\"\n"
        "                )\n"
        "                self._rebase_current(current)\n"
        "            results = self._call_batch(current)\n",
        "runtime stale-prefetch rebase anchor",
    )
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


def transform_proposer(source: str) -> str:
    marker = MARKERS["pearl_stage5_nanoparl_proposer_v3.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    source = replace_once(
        source,
        "from pearl_stage5_nanoparl_runtime_v1 import DraftRequest\n",
        "from pearl_stage5_nanoparl_runtime_v1 import (\n"
        "    DraftRequest,\n"
        "    NanoPearlPrefetchController,\n"
        "    trace_from_env,\n"
        ")\n",
        "proposer runtime imports",
    )
    methods = """    def __init__(self, vllm_config: Any) -> None:
        super().__init__(vllm_config)
        # Replace v1's ordinary controller with the rebase-aware controller.
        self._nano_controller.close()
        self._nano_controller = NanoPearlPrefetchController(
            self._request_batch_for_nano,
            trace=trace_from_env(),
            rebase_batch=self._rebase_batch_for_nano,
        )

    def _rebase_batch_for_nano(
        self, requests: list[dict[str, Any]]
    ) -> None:
        with self._nano_io_lock:
            response = self._request(
                {"cmd": "rebase_batch", "requests": requests}
            )
        if response.get("status") != "result":
            raise RuntimeError(
                "Draft rebase_batch returned an invalid response: "
                f"{response!r}"
            )

"""
    source = replace_once(
        source,
        "    @staticmethod\n    def _row_ids_fallback",
        methods + "    @staticmethod\n    def _row_ids_fallback",
        "proposer rebase methods anchor",
    )
    compile(source, "pearl_stage5_nanoparl_proposer_v3.py", "exec")
    return source


def transform_worker(source: str) -> str:
    marker = MARKERS["pearl_stage5_worker.py"]
    if marker in source:
        return source
    source = marker + "\n" + source
    branch = """                        if command == "rebase_batch":
                            raw_requests = message.get("requests")
                            if not isinstance(raw_requests, list) or not raw_requests:
                                raise ValueError(
                                    "rebase_batch requires a non-empty requests list"
                                )
                            requests = []
                            for index, item in enumerate(raw_requests):
                                if not isinstance(item, dict):
                                    raise ValueError(
                                        f"rebase request {index} must be an object"
                                    )
                                prefix = item.get("prefix_token_ids")
                                if not isinstance(prefix, list) or not prefix:
                                    raise ValueError(
                                        "rebase prefix_token_ids must be non-empty"
                                    )
                                requests.append(
                                    {
                                        "request_id": str(
                                            item.get("request_id", f"row-{index}")
                                        ),
                                        "prefix_token_ids": [int(x) for x in prefix],
                                    }
                                )
                            rebase_batch = getattr(engine, "rebase_batch", None)
                            if not callable(rebase_batch):
                                raise RuntimeError(
                                    "Draft engine has no rebase_batch(); "
                                    "stale nano-PEARL state cannot be safely discarded"
                                )
                            result = rebase_batch(requests)
                            print(
                                "[draft] rebase_batch "
                                f"batch_size={len(requests)} "
                                f"request_ids={[item['request_id'] for item in requests]}",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {"status": "result", "results": result or []},
                            )
                            continue

"""
    source = replace_once(
        source,
        '                        if command == "draft_batch":\n',
        branch + '                        if command == "draft_batch":\n',
        "worker rebase command anchor",
    )
    compile(source, "pearl_stage5_worker.py", "exec")
    return source


def transform_draft(source: str) -> str:
    marker = MARKERS["pearl_stage5_draft.py"]
    if marker in source:
        return source
    if source.count("    def propose_batch(") != 1:
        raise RuntimeError(
            "pearl_stage5_draft.py: expected one propose_batch() method; "
            "apply the validated batch>1 patch first; no files were changed"
        )
    source = marker + "\n" + source
    method = """    def rebase_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        \"\"\"Drop optimistic KV and rebuild each requested Draft slot.

        This is a fresh-request operation used only after a lookahead rejection.
        Compatible POST-VERIFY rounds still consume the prefetched result.
        \"\"\"
        if not isinstance(requests, list) or not requests:
            raise ValueError("rebase_batch requires a non-empty requests list")

        join_pipeline = getattr(self, "_join_pipeline", None)
        if callable(join_pipeline):
            join_pipeline()

        normalized: list[tuple[str, list[int]]] = []
        for index, item in enumerate(requests):
            if not isinstance(item, dict):
                raise ValueError(f"rebase request {index} must be an object")
            request_id = str(item.get("request_id", f"row-{index}"))
            prefix = item.get("prefix_token_ids")
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(
                    f"rebase prefix_token_ids must be non-empty for {request_id!r}"
                )
            normalized.append((request_id, [int(x) for x in prefix]))

        results: list[dict[str, Any]] = []
        with self._lock:
            for external_id, prefix in normalized:
                self._activate_request(external_id)
                old_internal_id = self.request_id
                if old_internal_id is not None:
                    self._pending_tokens.pop(str(old_internal_id), None)

                # Always abort and recreate the slot.  Do not call sync_prefix:
                # the stale optimistic tail must not be reused.
                self._reset_request(prefix)
                new_internal_id = self.request_id
                if new_internal_id is not None:
                    self._pending_tokens.pop(str(new_internal_id), None)
                results.append(
                    {
                        "request_id": external_id,
                        "prefix_len": len(prefix),
                    }
                )
                if os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1":
                    print(
                        "[PEARL_STAGE5_NANOPEARL_REBASE_V1] "
                        f"request_id={external_id!r} prefix_len={len(prefix)} "
                        f"old_internal={old_internal_id!r} "
                        f"new_internal={new_internal_id!r}",
                        flush=True,
                    )
        return results

"""
    source = replace_once(
        source,
        "    def propose_batch(",
        method + "    def propose_batch(",
        "draft rebase method anchor",
    )
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


TRANSFORMS = {
    Path("pearl_stage5_nanoparl_runtime_v1.py"): transform_runtime,
    Path("pearl_stage5_nanoparl_proposer_v3.py"): transform_proposer,
    Path("pearl_stage5_worker.py"): transform_worker,
    Path("pearl_stage5_draft.py"): transform_draft,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.pearl_rebase.",
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


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    originals: dict[Path, bytes] = {}
    transformed: dict[Path, str] = {}
    for relative, transform in TRANSFORMS.items():
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        raw = target.read_bytes()
        originals[relative] = raw
        source = raw.decode("utf-8")
        print(f"target: {target}")
        print(f"state: {'post' if MARKERS[relative.name] in source else 'pre'}")
        transformed[relative] = transform(source)

    print("change: explicit stale-prefetch Draft rebase (correctness-first)")
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    if backup_dir_arg is None:
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_nanoparl_discard_rebase_v1.{timestamp()}"
        )
    else:
        backup_dir = backup_dir_arg.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "mode": "targeted-file-backup",
        "source_sha256": {str(k): sha256(v) for k, v in originals.items()},
    }
    for relative, raw in originals.items():
        backup_target = backup_dir / relative
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        backup_target.write_bytes(raw)
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"backup: {backup_dir}")
    for relative, transformed_source in transformed.items():
        target = repo / relative
        write_atomic(target, transformed_source, target.stat().st_mode)
        print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-ascend-hust"),
    )
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

