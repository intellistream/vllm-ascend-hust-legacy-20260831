#!/usr/bin/env python3
"""Use one Draft Engine step for all requests in a Stage-5 batch.

v2 fixed correctness by discarding cross-request outputs and driving each
request afresh.  v3 keeps the prefix synchronization phase serialized, then
collects all active request outputs from the same EngineCore step.  This is the
first actual Draft-side batch execution; the Target verification batch and the
request-isolated partial-block KV path remain unchanged.
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
from textwrap import dedent


MARKER_V1 = "# PEARL_STAGE5_BATCH_GT1_V1"
MARKER_V2 = "# PEARL_STAGE5_BATCH_GT1_V2"
MARKER_V3 = "# PEARL_STAGE5_BATCH_GT1_V3"
TARGETS = (
    Path("pearl_stage5_draft.py"),
    Path("pearl_stage5_worker.py"),
)


DRAFT_METHODS = dedent(
    '''
        def _collect_batch_tokens(
            self,
            internal_to_external: dict[str, str],
            target_counts: dict[str, int],
        ) -> dict[str, list[int]]:
            """Advance all active Draft requests and collect their tokens.

            A call to ``get_output`` advances one EngineCore step.  All
            requests that are ready in the Draft scheduler are therefore
            verified from the same batched model forward.  Extra tokens from
            a single output are queued and consumed on later draft rounds.
            """
            collected = {external_id: [] for external_id in target_counts}
            pending = self._pending_tokens
            batch_step = 0

            while any(
                len(collected[external_id]) < count
                for external_id, count in target_counts.items()
            ):
                batch_step += 1
                outputs = self.core_client.get_output()
                if os.environ.get("PEARL_STAGE5_DRAFT_BATCH_TRACE", "0") == "1":
                    print(
                        "[PEARL_STAGE5_DRAFT_BATCH_TRACE] "
                        f"step={batch_step} "
                        f"output_request_ids={[str(output.request_id) for output in outputs.outputs]} ",
                        flush=True,
                    )
                for output in outputs.outputs:
                    internal_id = str(output.request_id)
                    external_id = internal_to_external.get(internal_id)
                    if external_id is None:
                        # An output for a request outside this RPC is stale
                        # with respect to the current Target prefix.  Do not
                        # leak it into the next batch.
                        continue
                    new_token_ids = getattr(output, "new_token_ids", None) or []
                    if new_token_ids:
                        queue = pending.setdefault(internal_id, deque())
                        queue.extend(int(token_id) for token_id in new_token_ids)
                    if output.finished and not new_token_ids:
                        if len(collected[external_id]) < target_counts[external_id]:
                            raise RuntimeError(
                                "Draft request finished before returning enough "
                                f"tokens: request={external_id!r} "
                                f"reason={output.finish_reason!r}"
                            )

                for internal_id, external_id in internal_to_external.items():
                    count = target_counts[external_id]
                    queue = pending.get(internal_id)
                    while (
                        queue
                        and len(collected[external_id]) < count
                    ):
                        collected[external_id].append(queue.popleft())

                for internal_id, external_id in internal_to_external.items():
                    if len(collected[external_id]) >= target_counts[external_id]:
                        continue
                    if internal_id not in self.core.scheduler.requests:
                        raise RuntimeError(
                            "Draft request disappeared while decoding: "
                            f"request={external_id!r} internal={internal_id!r}"
                        )

            return collected

        def propose_batch(
            self,
            requests: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            """Synchronize prefixes, then draft all rows in one batch."""
            if not requests:
                return []
            if len(requests) > self.max_num_seqs:
                raise ValueError(
                    f"Draft batch={len(requests)} exceeds "
                    f"max_num_seqs={self.max_num_seqs}"
                )

            normalized: list[tuple[str, list[int], int]] = []
            seen_ids: set[str] = set()
            for index, item in enumerate(requests):
                if not isinstance(item, dict):
                    raise ValueError(f"Draft request {index} must be an object")
                external_id = str(item.get("request_id", f"row-{index}"))
                if external_id in seen_ids:
                    raise ValueError(
                        f"duplicate external Draft request ID: {external_id!r}"
                    )
                seen_ids.add(external_id)
                prefix = item.get("prefix_token_ids")
                gamma = int(item.get("gamma", 0))
                if not isinstance(prefix, list) or not prefix:
                    raise ValueError(
                        f"prefix_token_ids must be non-empty for {external_id!r}"
                    )
                if gamma < 0:
                    raise ValueError(f"gamma must be non-negative for {external_id!r}")
                normalized.append(
                    (external_id, [int(token_id) for token_id in prefix], gamma)
                )

            with self._lock:
                internal_to_external: dict[str, str] = {}
                target_counts: dict[str, int] = {}
                for external_id, prefix, gamma in normalized:
                    self._activate_request(external_id)
                    self.sync_prefix(prefix)
                    internal_id = self.request_id
                    if internal_id is None:
                        raise RuntimeError(
                            f"Draft request was not created for {external_id!r}"
                        )
                    internal_id = str(internal_id)
                    # Anything left here was produced before this Target
                    # prefix synchronization and must not be reused.
                    self._pending_tokens.pop(internal_id, None)
                    internal_to_external[internal_id] = external_id
                    target_counts[external_id] = gamma

                active_counts = {
                    external_id: gamma
                    for external_id, gamma in target_counts.items()
                    if gamma > 0
                }
                if active_counts:
                    collected = self._collect_batch_tokens(
                        internal_to_external={
                            internal_id: external_id
                            for internal_id, external_id in internal_to_external.items()
                            if target_counts[external_id] > 0
                        },
                        target_counts=active_counts,
                    )
                else:
                    collected = {external_id: [] for external_id in target_counts}

                return [
                    {
                        "request_id": external_id,
                        "draft_token_ids": collected.get(external_id, [])[:gamma],
                    }
                    for external_id, _prefix, gamma in normalized
                ]

        def propose(
            self,
            request_id: str | list[int],
            prefix_token_ids: list[int] | int,
            gamma: int | None = None,
        ) -> list[int]:
            """Compatibility wrapper for the legacy single-request RPC."""
            if gamma is None:
                external_id = "target-0"
                legacy_prefix = request_id
                legacy_gamma = prefix_token_ids
                if not isinstance(legacy_prefix, list):
                    raise TypeError("legacy Draft prefix must be a list")
                prefix_token_ids = legacy_prefix
                gamma = int(legacy_gamma)
            else:
                external_id = str(request_id)
                if not isinstance(prefix_token_ids, list):
                    raise TypeError("Draft prefix must be a list")

            result = self.propose_batch(
                [
                    {
                        "request_id": external_id,
                        "prefix_token_ids": prefix_token_ids,
                        "gamma": int(gamma),
                    }
                ]
            )
            return result[0]["draft_token_ids"] if result else []
    '''
)


WORKER_LOOP = dedent(
    '''
    def _draft_proposal_loop(
        server: socket.socket,
        engine: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                server.settimeout(0.5)
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                with conn.makefile("r", encoding="utf-8") as reader:
                    while not stop_event.is_set():
                        message = receive_message(reader)
                        if message is None:
                            break
                        try:
                            command = message.get("cmd")
                            if command == "draft_batch":
                                raw_requests = message.get("requests")
                                if not isinstance(raw_requests, list) or not raw_requests:
                                    raise ValueError(
                                        "draft_batch requires a non-empty requests list"
                                    )
                                requests = []
                                for index, request in enumerate(raw_requests):
                                    if not isinstance(request, dict):
                                        raise ValueError(
                                            f"draft request {index} must be an object"
                                        )
                                    request_id = str(
                                        request.get("request_id", f"row-{index}")
                                    )
                                    prefix = request.get("prefix_token_ids")
                                    gamma = int(request.get("gamma", 0))
                                    if not isinstance(prefix, list) or not prefix:
                                        raise ValueError(
                                            "prefix_token_ids must be non-empty for "
                                            f"{request_id!r}"
                                        )
                                    requests.append(
                                        {
                                            "request_id": request_id,
                                            "prefix_token_ids": [int(x) for x in prefix],
                                            "gamma": gamma,
                                        }
                                    )

                                print(
                                    "[draft] batch proposal "
                                    f"batch_size={len(requests)} "
                                    f"request_ids={[item['request_id'] for item in requests]} "
                                    f"gamma={[item['gamma'] for item in requests]}",
                                    flush=True,
                                )
                                results = engine.propose_batch(requests)
                                by_id = {
                                    str(item["request_id"]): item
                                    for item in results
                                }
                                for request in requests:
                                    item = by_id[request["request_id"]]
                                    print(
                                        "[draft] proposal "
                                        f"request_id={request['request_id']!r} "
                                        f"prefix_len={len(request['prefix_token_ids'])} "
                                        f"gamma={request['gamma']} "
                                        f"draft_ids={item['draft_token_ids']}",
                                        flush=True,
                                    )
                                send_message(conn, {"status": "result", "results": results})
                                continue

                            if command != "draft":
                                raise ValueError(f"unknown proposal command: {command!r}")
                            prefix = message.get("prefix_token_ids")
                            gamma = int(message.get("gamma", 0))
                            request_id = str(message.get("request_id", "target-0"))
                            if not isinstance(prefix, list) or not prefix:
                                raise ValueError("prefix_token_ids must be non-empty")
                            draft_ids = engine.propose(
                                request_id,
                                [int(x) for x in prefix],
                                gamma,
                            )
                            print(
                                "[draft] proposal "
                                f"request_id={request_id!r} "
                                f"prefix_len={len(prefix)} gamma={gamma} "
                                f"draft_ids={draft_ids}",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {
                                    "status": "result",
                                    "draft_token_ids": draft_ids,
                                    "prefix_len": len(prefix),
                                    "request_id": request_id,
                                },
                            )
                        except Exception as exc:
                            traceback.print_exc()
                            send_message(conn, {"status": "error", "error": repr(exc)})
    '''
)


def replace_method(source: str, method_name: str, next_method: str, block: str) -> str:
    start = source.find(f"    def {method_name}")
    end = source.find(f"    def {next_method}", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method_name}: method boundary not found; no files were changed"
        )
    return source[:start] + block.strip("\n") + "\n\n" + source[end:]


def replace_function(source: str, function_name: str, next_function: str, block: str) -> str:
    start = source.find(f"def {function_name}")
    end = source.find(f"def {next_function}", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{function_name}: function boundary not found; no files were changed"
        )
    return source[:start] + block.strip("\n") + "\n\n" + source[end:]


def transform_draft(source: str) -> str:
    if MARKER_V3 in source:
        raise RuntimeError("Draft v3 marker already exists")
    if MARKER_V2 not in source or MARKER_V1 not in source:
        raise RuntimeError(
            "Draft source is not in batch v2 state; no files were changed"
        )
    return MARKER_V3 + "\n" + replace_method(
        source, "propose(", "shutdown(", "\n".join(
            "    " + line if line else "" for line in DRAFT_METHODS.strip("\n").splitlines()
        )
    )


def transform_worker(source: str) -> str:
    if MARKER_V3 in source:
        raise RuntimeError("Worker v3 marker already exists")
    if MARKER_V1 not in source:
        raise RuntimeError(
            "Worker source is not in batch v1 state; no files were changed"
        )
    return MARKER_V3 + "\n" + replace_function(
        source, "_draft_proposal_loop(", "_run_draft(", WORKER_LOOP
    )


PATCHERS = {
    Path("pearl_stage5_draft.py"): transform_draft,
    Path("pearl_stage5_worker.py"): transform_worker,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_batch_gt1_v3.{stamp}"
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
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


def apply_patch(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    originals: dict[Path, bytes] = {}
    patched: dict[Path, str] = {}
    for relative in TARGETS:
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        originals[relative] = target.read_bytes()

    for relative in TARGETS:
        source = originals[relative].decode("utf-8")
        transformed = PATCHERS[relative](source)
        compile(transformed, str(repo / relative), "exec")
        patched[relative] = transformed
        print(f"target: {repo / relative}")
        print("state: pre-v3")

    print(
        "change: batched Draft EngineCore steps after request-wise prefix sync"
    )
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_arg)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "marker": MARKER_V3,
        "mode": "batched_draft_engine_steps_after_prefix_sync",
        "files": {},
    }
    for relative in TARGETS:
        target = repo / relative
        backup_file = backup_dir / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_file)
        manifest["files"][str(relative)] = {
            "target": str(target),
            "backup_file": str(backup_file),
            "original_sha256": sha256_bytes(originals[relative]),
            "original_size": len(originals[relative]),
        }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    written: list[Path] = []
    try:
        for relative in TARGETS:
            target = repo / relative
            write_atomic(target, patched[relative], target.stat().st_mode)
            written.append(relative)
    except Exception:
        for relative in written:
            shutil.copy2(backup_dir / relative, repo / relative)
        raise
    print(f"backup: {backup_dir}")
    for relative in TARGETS:
        print(f"patched: {repo / relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
