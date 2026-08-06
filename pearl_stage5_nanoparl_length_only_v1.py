#!/usr/bin/env python3
"""Install the strict accepted_len/valid_len-only nano-PEARL fast path.

The existing commit-state path keeps the Draft Request and KV blocks alive,
but still sends a complete Target prefix and calls _replace_tokens.  This
patch adds a conservative fast path.  It is used only when the Target prefix
is exactly the optimistic prefix already prefetched, every draft token was
accepted, there is no replacement token, and the request is not finished.
Those rows carry lengths only.  Every other row uses the existing full-prefix
commit path.

Enable at runtime with PEARL_STAGE5_NANOPEARL_LENGTH_ONLY=1.  The patch is
guarded by the existing commit-state markers and backs up only changed files.
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


MARKERS = {
    "runtime": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V1",
    "worker": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V1",
    "draft": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V1",
}

TARGETS = (
    Path("pearl_stage5_nanoparl_runtime_v1.py"),
    Path("pearl_stage5_worker.py"),
    Path("pearl_stage5_draft.py"),
)


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(source: str, method: str, replacement: str, name: str) -> str:
    anchor = f"    def {method}("
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one {anchor} method, found {count}; "
            "no files were changed"
        )
    start = source.index(anchor)
    end = source.find("\n    def ", start + len(anchor))
    if end < 0:
        end = len(source)
    return source[:start] + replacement.rstrip() + "\n" + source[end + 1 :]


def transform_runtime(source: str) -> str:
    marker = MARKERS["runtime"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_RUNTIME_V1" not in source:
        raise RuntimeError(
            "runtime: commit-state patch is missing; apply it first; "
            "no files were changed"
        )

    source = marker + "\n" + source
    source = replace_once(
        source,
        "        self._commit_batch = commit_batch\n",
        "        self._commit_batch = commit_batch\n"
        "        self._length_only_enabled = (\n"
        "            os.environ.get(\n"
        '                "PEARL_STAGE5_NANOPEARL_LENGTH_ONLY", "0"\n'
        "            ) == \"1\"\n"
        "        )\n",
        "runtime length-only flag",
    )

    replacement = '''    def _eligible_length_only_ids(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        pending_by_id: dict[str, tuple[DraftRequest, DraftResult]],
        pending_error: Exception | None,
    ) -> set[str]:
        """Return rows whose existing optimistic prefix is authoritative."""
        if (
            not self._length_only_enabled
            or pending_error is not None
            or not pending_by_id
        ):
            return set()
        eligible: set[str] = set()
        for request in current:
            result = explicit.get(request.request_id)
            matched = pending_by_id.get(request.request_id)
            if result is None or matched is None:
                continue
            old, old_result = matched
            if result.finished or not result.all_accepted:
                continue
            if result.replacement_token_id is not None:
                continue
            if result.draft_len != len(old_result.draft_token_ids):
                continue
            if request.gamma != old.gamma:
                continue
            if request.prefix_token_ids != old.prefix_token_ids:
                continue
            eligible.add(request.request_id)
        if eligible:
            self._trace(
                "length_only_eligible "
                f"round={self.round_id} batch={len(eligible)} "
                f"rows={','.join(sorted(eligible))}"
            )
        return eligible

    def _commit_current(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        length_only_ids: set[str] | None = None,
    ) -> set[str]:
        """Commit Target boundaries, using lengths for safe rows."""
        if self._commit_batch is None or not explicit:
            return set()

        length_only_ids = length_only_ids or set()
        updates: list[dict] = []
        committed_ids: set[str] = set()
        for request in current:
            result = explicit.get(request.request_id)
            if result is None:
                continue
            valid_len = max(0, len(request.prefix_token_ids) - 1)
            length_only = (
                self._length_only_enabled
                and request.request_id in length_only_ids
            )
            update = {
                "request_id": request.request_id,
                "gamma": request.gamma,
                "accepted_len": result.accepted_len,
                "draft_len": result.draft_len,
                "valid_len": valid_len,
                "target_prefix_len": len(request.prefix_token_ids),
                "replacement_token_id": result.replacement_token_id,
                "finished": result.finished,
                "length_only": length_only,
            }
            if not length_only:
                update["prefix_token_ids"] = list(request.prefix_token_ids)
            updates.append(update)
            committed_ids.add(request.request_id)

        if not updates:
            return set()

        self._trace(
            "commit_batch_start "
            f"round={self.round_id} batch={len(updates)} "
            f"accepted={sum(item['accepted_len'] for item in updates)} "
            f"valid={sum(item['valid_len'] for item in updates)} "
            f"length_only={sum(bool(item['length_only']) for item in updates)}"
        )
        self._commit_batch(updates)
        self._trace(
            "commit_batch_done "
            f"round={self.round_id} batch={len(updates)} "
            f"rows={','.join(sorted(committed_ids))}"
        )
        return committed_ids
'''
    source = replace_method(source, "_commit_current", replacement, "runtime commit")
    source = replace_once(
        source,
        "        committed_ids = self._commit_current(current, explicit)\n",
        "        length_only_ids = self._eligible_length_only_ids(\n"
        "            current,\n"
        "            explicit,\n"
        "            pending_by_id,\n"
        "            pending_error,\n"
        "        )\n"
        "        committed_ids = self._commit_current(\n"
        "            current, explicit, length_only_ids\n"
        "        )\n",
        "runtime length-only dispatch",
    )
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


def transform_worker(source: str) -> str:
    marker = MARKERS["worker"]
    if marker in source:
        compile(source, "pearl_stage5_worker.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_WORKER_V1" not in source:
        raise RuntimeError(
            "worker: commit-state patch is missing; apply it first; "
            "no files were changed"
        )

    start_anchor = '                        if command == "commit_batch":'
    end_anchor = '                        if command == "draft_batch":'
    if source.count(start_anchor) != 1 or source.count(end_anchor) != 1:
        raise RuntimeError(
            "worker: expected one commit_batch/draft_batch command pair; "
            "no files were changed"
        )
    start = source.index(start_anchor)
    end = source.index(end_anchor, start)
    branch = '''                        if command == "commit_batch":
                            raw_updates = message.get("updates")
                            if not isinstance(raw_updates, list) or not raw_updates:
                                raise ValueError(
                                    "commit_batch requires a non-empty updates list"
                                )
                            commit_batch = getattr(engine, "commit_batch", None)
                            if not callable(commit_batch):
                                raise RuntimeError(
                                    "Draft engine has no commit_batch(); "
                                    "apply the nano-PEARL commit-state patch first"
                                )
                            updates = []
                            for index, item in enumerate(raw_updates):
                                if not isinstance(item, dict):
                                    raise ValueError(
                                        f"commit update {index} must be an object"
                                    )
                                length_only = bool(item.get("length_only", False))
                                prefix = item.get("prefix_token_ids")
                                target_prefix_len = int(
                                    item.get(
                                        "target_prefix_len",
                                        len(prefix) if isinstance(prefix, list) else -1,
                                    )
                                )
                                valid_len = int(item.get("valid_len", -1))
                                accepted_len = int(item.get("accepted_len", 0))
                                draft_len = int(item.get("draft_len", 0))
                                replacement = item.get("replacement_token_id")
                                finished = bool(item.get("finished", False))
                                if length_only:
                                    if prefix is not None:
                                        raise ValueError(
                                            "length-only commit must not carry prefix_token_ids"
                                        )
                                    if target_prefix_len <= 0:
                                        raise ValueError(
                                            "length-only target_prefix_len must be positive"
                                        )
                                    if valid_len < 0 or valid_len > target_prefix_len - 1:
                                        raise ValueError(
                                            "length-only valid_len must be in "
                                            "[0, target_prefix_len-1]"
                                        )
                                    if (
                                        finished
                                        or accepted_len != draft_len
                                        or replacement is not None
                                    ):
                                        raise ValueError(
                                            "length-only commit requires all "
                                            "draft tokens accepted without replacement"
                                        )
                                else:
                                    if not isinstance(prefix, list) or not prefix:
                                        raise ValueError(
                                            "commit prefix_token_ids must be non-empty"
                                        )
                                    if target_prefix_len != len(prefix):
                                        raise ValueError(
                                            "target_prefix_len does not match prefix length"
                                        )
                                    if valid_len < 0 or valid_len > len(prefix) - 1:
                                        raise ValueError(
                                            "commit valid_len must be in [0, prefix_len-1]"
                                        )
                                normalized = {
                                    "request_id": str(
                                        item.get("request_id", f"row-{index}")
                                    ),
                                    "gamma": int(item.get("gamma", 0)),
                                    "accepted_len": accepted_len,
                                    "draft_len": draft_len,
                                    "valid_len": valid_len,
                                    "target_prefix_len": target_prefix_len,
                                    "replacement_token_id": (
                                        None if replacement is None else int(replacement)
                                    ),
                                    "finished": finished,
                                    "length_only": length_only,
                                }
                                if not length_only:
                                    normalized["prefix_token_ids"] = [
                                        int(x) for x in prefix
                                    ]
                                updates.append(normalized)
                            result = commit_batch(updates)
                            print(
                                "[draft] commit_batch "
                                f"batch_size={len(updates)} "
                                f"accepted={sum(x['accepted_len'] for x in updates)} "
                                f"valid={sum(x['valid_len'] for x in updates)} "
                                f"length_only={sum(bool(x['length_only']) for x in updates)}",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {"status": "result", "results": result or []},
                            )
                            continue

'''
    source = source[:start] + branch + source[end:]
    source = marker + "\n" + source
    compile(source, "pearl_stage5_worker.py", "exec")
    return source


LENGTH_ONLY_DRAFT_METHOD = '''    def _sync_model_runner_lengths_only(
        self,
        request: Any,
        target_prefix_len: int,
    ) -> None:
        """Update active length metadata without copying token IDs."""
        executor = getattr(self.core, "model_executor", None)
        driver_worker = getattr(executor, "driver_worker", None)
        runner = getattr(driver_worker, "model_runner", None)
        if runner is None:
            workers = getattr(executor, "workers", None)
            if workers:
                runner = getattr(workers[0], "model_runner", None)
        if runner is None:
            raise RuntimeError(
                "Cannot locate the in-process Draft model runner for "
                "length-only commit"
            )
        req_state = getattr(runner, "requests", {}).get(self.request_id)
        input_batch = getattr(runner, "input_batch", None)
        if req_state is None or input_batch is None:
            raise RuntimeError(
                "Draft model runner has no cached state for length-only "
                f"request {self.request_id!r}"
            )
        req_index = input_batch.req_id_to_index.get(self.request_id)
        if req_index is None:
            raise RuntimeError(
                "Draft model runner input batch has no index for "
                f"length-only request {self.request_id!r}"
            )

        request.num_computed_tokens = target_prefix_len - 1
        req_state.num_computed_tokens = request.num_computed_tokens
        if hasattr(input_batch, "num_computed_tokens_cpu"):
            input_batch.num_computed_tokens_cpu[req_index] = (
                request.num_computed_tokens
            )
        input_batch.num_tokens_no_spec[req_index] = target_prefix_len
        if hasattr(input_batch, "num_tokens"):
            input_batch.num_tokens[req_index] = target_prefix_len
        input_batch.spec_token_ids[req_index].clear()

    def commit_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Commit Target boundaries, with a strict length-only fast path."""
        if not isinstance(updates, list) or not updates:
            raise ValueError("commit_batch requires a non-empty updates list")
        if os.environ.get("PEARL_STAGE5_NANOPEARL_COMMIT_STATE", "0") != "1":
            rebase_batch = getattr(self, "rebase_batch", None)
            if callable(rebase_batch):
                return rebase_batch(updates)
            raise RuntimeError(
                "commit state is disabled and Draft has no rebase_batch fallback"
            )

        join_pipeline = getattr(self, "_join_pipeline", None)
        if callable(join_pipeline):
            join_pipeline()

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(updates):
            if not isinstance(item, dict):
                raise ValueError(f"commit update {index} must be an object")
            external_id = str(item.get("request_id", f"row-{index}"))
            length_only = bool(item.get("length_only", False))
            prefix = item.get("prefix_token_ids")
            target_prefix_len = int(
                item.get(
                    "target_prefix_len",
                    len(prefix) if isinstance(prefix, list) else -1,
                )
            )
            valid_len = int(item.get("valid_len", -1))
            accepted_len = int(item.get("accepted_len", 0))
            draft_len = int(item.get("draft_len", 0))
            replacement = item.get("replacement_token_id")
            finished = bool(item.get("finished", False))
            if length_only:
                if prefix is not None:
                    raise ValueError(
                        f"length-only commit carried token IDs for {external_id!r}"
                    )
                if target_prefix_len <= 0 or valid_len != target_prefix_len - 1:
                    raise ValueError(
                        f"invalid length-only boundary for {external_id!r}"
                    )
                if (
                    finished
                    or accepted_len != draft_len
                    or replacement is not None
                ):
                    raise ValueError(
                        f"unsafe length-only verifier result for {external_id!r}"
                    )
                normalized_prefix = None
            else:
                if not isinstance(prefix, list) or not prefix:
                    raise ValueError(
                        f"commit prefix_token_ids must be non-empty for {external_id!r}"
                    )
                normalized_prefix = [int(x) for x in prefix]
                if target_prefix_len != len(normalized_prefix):
                    raise ValueError(
                        f"target_prefix_len mismatch for {external_id!r}"
                    )
                if valid_len < 0 or valid_len > len(normalized_prefix) - 1:
                    raise ValueError(
                        f"invalid valid_len={valid_len} for {external_id!r}"
                    )
            if accepted_len < 0 or draft_len < 0 or accepted_len > draft_len:
                raise ValueError(
                    f"invalid accepted/draft lengths for {external_id!r}"
                )
            normalized.append(
                {
                    "request_id": external_id,
                    "prefix_token_ids": normalized_prefix,
                    "target_prefix_len": target_prefix_len,
                    "valid_len": valid_len,
                    "accepted_len": accepted_len,
                    "draft_len": draft_len,
                    "replacement_token_id": replacement,
                    "finished": finished,
                    "length_only": length_only,
                }
            )

        results: list[dict[str, Any]] = []
        with self._lock:
            for item in normalized:
                external_id = item["request_id"]
                valid_len = item["valid_len"]
                target_prefix_len = item["target_prefix_len"]
                self._activate_request(external_id)
                request = self._request()
                scheduler = self.core.scheduler

                if item["finished"]:
                    results.append(
                        {
                            "request_id": external_id,
                            "valid_len": valid_len,
                            "action": "skip_finished",
                        }
                    )
                    continue

                from vllm.v1.request import RequestStatus

                if not (
                    request.status == RequestStatus.RUNNING
                    and request in scheduler.running
                ):
                    raise RuntimeError(
                        "commit_batch requires a RUNNING persistent Request: "
                        f"{external_id!r} status={request.status!s}"
                    )

                if item["length_only"]:
                    if len(self.committed_token_ids) < target_prefix_len:
                        raise RuntimeError(
                            "length-only commit exceeds the persistent Draft "
                            f"sequence for {external_id!r}: "
                            f"have={len(self.committed_token_ids)} "
                            f"need={target_prefix_len}"
                        )
                    request.is_prefill_chunk = False
                    request.spec_token_ids = []
                    request.num_output_placeholders = 0
                    self._sync_model_runner_lengths_only(
                        request,
                        target_prefix_len,
                    )
                    inflight_prefills = getattr(
                        scheduler, "_inflight_prefills", None
                    )
                    if inflight_prefills is not None:
                        inflight_prefills.discard(request)
                    results.append(
                        {
                            "request_id": external_id,
                            "accepted_len": item["accepted_len"],
                            "draft_len": item["draft_len"],
                            "valid_len": valid_len,
                            "common_len": target_prefix_len,
                            "action": "update_lengths_only_keep_kv",
                        }
                    )
                    if os.environ.get(
                        "PEARL_STAGE5_NANOPEARL_TRACE", "0"
                    ) == "1":
                        print(
                            "[PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_V1] "
                            f"request_id={external_id!r} "
                            f"accepted_len={item['accepted_len']} "
                            f"draft_len={item['draft_len']} "
                            f"valid_len={valid_len} "
                            f"target_prefix_len={target_prefix_len} "
                            "action=update_lengths_only_keep_kv",
                            flush=True,
                        )
                    continue

                prefix = item["prefix_token_ids"]
                old_prefix = list(self.committed_token_ids)
                common_len = 0
                for old_token, new_token in zip(old_prefix, prefix):
                    if old_token != new_token:
                        break
                    common_len += 1
                if common_len < valid_len:
                    raise RuntimeError(
                        "commit prefix diverged before valid_len for "
                        f"{external_id!r}: common_len={common_len} "
                        f"valid_len={valid_len}"
                    )

                self.prompt_token_ids = list(prefix)
                request.prompt_token_ids = list(prefix)
                request.num_prompt_tokens = len(prefix)
                if getattr(request, "prompt_is_token_ids", None) is not None:
                    request.prompt_is_token_ids = [True] * len(prefix)
                self._replace_tokens(prefix)
                request.num_computed_tokens = valid_len
                request.is_prefill_chunk = False
                request.spec_token_ids = []
                request.num_output_placeholders = 0
                self._sync_model_runner_state(prefix, request)
                inflight_prefills = getattr(scheduler, "_inflight_prefills", None)
                if inflight_prefills is not None:
                    inflight_prefills.discard(request)

                results.append(
                    {
                        "request_id": external_id,
                        "accepted_len": item["accepted_len"],
                        "draft_len": item["draft_len"],
                        "valid_len": valid_len,
                        "common_len": common_len,
                        "action": "update_lengths_keep_kv",
                    }
                )
                if os.environ.get(
                    "PEARL_STAGE5_NANOPEARL_TRACE", "0"
                ) == "1":
                    print(
                        "[PEARL_STAGE5_NANOPEARL_COMMIT_STATE_V1] "
                        f"request_id={external_id!r} "
                        f"accepted_len={item['accepted_len']} "
                        f"draft_len={item['draft_len']} "
                        f"valid_len={valid_len} "
                        f"common_len={common_len} "
                        "action=update_lengths_keep_kv",
                        flush=True,
                    )
        return results
'''


def transform_draft(source: str) -> str:
    marker = MARKERS["draft"]
    if marker in source:
        compile(source, "pearl_stage5_draft.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_DRAFT_V1" not in source:
        raise RuntimeError(
            "draft: commit-state patch is missing; apply it first; "
            "no files were changed"
        )
    source = replace_method(
        source,
        "commit_batch",
        LENGTH_ONLY_DRAFT_METHOD,
        "draft commit_batch",
    )
    source = marker + "\n" + source
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


TRANSFORMS = {
    Path("pearl_stage5_nanoparl_runtime_v1.py"): transform_runtime,
    Path("pearl_stage5_worker.py"): transform_worker,
    Path("pearl_stage5_draft.py"): transform_draft,
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_backup_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, destination)


def write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.pearl_length_only.",
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
    transformed: dict[Path, str] = {}
    originals: dict[Path, bytes] = {}
    modes: dict[Path, int] = {}
    changed: list[Path] = []

    for relative, transform in TRANSFORMS.items():
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        raw = target.read_bytes()
        original = raw.decode("utf-8")
        updated = transform(original)
        originals[relative] = raw
        modes[relative] = target.stat().st_mode
        transformed[relative] = updated
        print(f"target: {target}")
        print(f"state: {'post' if updated == original else 'pre'}")
        if updated != original:
            changed.append(relative)

    print("change: strict accepted_len/valid_len-only nano-PEARL commit path")
    if not changed:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_nanoparl_length_only_v1.{timestamp()}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    for relative in changed:
        copy_backup_file(repo / relative, backup_dir / relative)
    (backup_dir / "MANIFEST.sha256").write_text(
        "\n".join(
            f"{sha256(originals[relative])}  {relative}"
            for relative in changed
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"backup: {backup_dir}")
    for relative in changed:
        target = repo / relative
        write_atomic(target, transformed[relative], modes[relative])
        print(f"patched: {target}")


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
