#!/usr/bin/env python3
"""Add a batch-level accepted/valid-length commit to nano-PEARL.

The existing explicit-verification path knows the Target result, but the
Draft side still receives a rebase_batch request for rows that cannot consume
the optimistic look-ahead. That is a useful correctness fallback; it is not
yet the nano-PEARL control plane.

This patch adds an opt-in commit_batch RPC. One ordered message carries, for
every active row:

* accepted_len / draft_len from the Target verifier;
* valid_len (the vLLM num_computed_tokens boundary);
* the authoritative Target prefix and optional replacement token.

When PEARL_STAGE5_NANOPEARL_COMMIT_STATE=1 is set, Draft updates the existing
persistent Request/model-runner row in one batch lock and retains the
Request-owned KV blocks. It does not call _reset_request or remove the row
from the model-runner batch. The current rebase_batch path stays available as
the default compatibility fallback until the new path is tested.

This is deliberately a new, source-guarded patch script. It backs up only the
files it changes, so dynamic build symlinks under csrc/build cannot break the
backup. It does not require async scheduling or HCCL; the message is
transported over the existing ordered Draft socket first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MARKERS = {
    "runtime": "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_RUNTIME_V1",
    "proposer_v1": "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_PROPOSER_V1",
    "proposer_v3": "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_PROPOSER_V3",
    "worker": "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_WORKER_V1",
    "draft": "# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_DRAFT_V1",
}

TARGETS = (
    Path("pearl_stage5_nanoparl_runtime_v1.py"),
    Path("pearl_stage5_nanoparl_proposer_v1.py"),
    Path("pearl_stage5_nanoparl_proposer_v3.py"),
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


def insert_before_method(source: str, method: str, text: str, name: str) -> str:
    anchor = f"    def {method}"
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected one {anchor}() anchor, found {count}; "
            "no files were changed"
        )
    return source.replace(anchor, text.rstrip() + "\n\n" + anchor, 1)


def transform_runtime(source: str) -> str:
    marker = MARKERS["runtime"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_RUNTIME_V1" not in source:
        raise RuntimeError(
            "runtime: explicit verification patch is missing; apply the "
            "validated explicit-verify patch first; no files were changed"
        )

    source = marker + "\n" + source
    source = replace_once(
        source,
        "TraceFn = Callable[[str], None]\n",
        "TraceFn = Callable[[str], None]\n"
        "CommitBatchFn = Callable[[list[dict]], None]\n",
        "runtime commit callback type",
    )
    source = replace_once(
        source,
        "        trace: TraceFn | None = None,\n",
        "        trace: TraceFn | None = None,\n"
        "        commit_batch: CommitBatchFn | None = None,\n",
        "runtime commit constructor argument",
    )
    source = replace_once(
        source,
        "        self._request_batch = request_batch\n",
        "        self._request_batch = request_batch\n"
        "        self._commit_batch = commit_batch\n",
        "runtime commit callback state",
    )

    commit_method = '''    def _commit_current(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
    ) -> set[str]:
        """Commit Target verification boundaries to Draft in one batch.

        valid_len follows the existing Ascend V1 convention used by the
        in-place rollback patch: the final Target token remains the next token
        to compute, therefore num_computed_tokens = len(prefix) - 1.
        """
        if self._commit_batch is None or not explicit:
            return set()

        updates: list[dict] = []
        committed_ids: set[str] = set()
        for request in current:
            result = explicit.get(request.request_id)
            if result is None:
                continue
            valid_len = max(0, len(request.prefix_token_ids) - 1)
            updates.append(
                {
                    "request_id": request.request_id,
                    "prefix_token_ids": list(request.prefix_token_ids),
                    "gamma": request.gamma,
                    "accepted_len": result.accepted_len,
                    "draft_len": result.draft_len,
                    "valid_len": valid_len,
                    "target_prefix_len": len(request.prefix_token_ids),
                    "replacement_token_id": result.replacement_token_id,
                    "finished": result.finished,
                }
            )
            committed_ids.add(request.request_id)

        if not updates:
            return set()

        self._trace(
            "commit_batch_start "
            f"round={self.round_id} batch={len(updates)} "
            f"accepted={sum(item['accepted_len'] for item in updates)} "
            f"valid={sum(item['valid_len'] for item in updates)}"
        )
        self._commit_batch(updates)
        self._trace(
            "commit_batch_done "
            f"round={self.round_id} batch={len(updates)} "
            f"rows={','.join(sorted(committed_ids))}"
        )
        return committed_ids
'''
    source = insert_before_method(
        source,
        "_call_batch(",
        commit_method,
        "runtime commit method",
    )

    source = replace_once(
        source,
        "        selected: dict[str, DraftResult] = {}\n",
        "        committed_ids = self._commit_current(current, explicit)\n\n"
        "        selected: dict[str, DraftResult] = {}\n",
        "runtime commit dispatch",
    )
    source = replace_once(
        source,
        "                rebase_current = getattr(self, \"_rebase_current\", None)\n"
        "                if callable(rebase_current):\n"
        "                    rebase_current(tuple(refresh_requests))\n",
        "                rebase_current = getattr(self, \"_rebase_current\", None)\n"
        "                if callable(rebase_current):\n"
        "                    uncommitted_refresh = tuple(\n"
        "                        request\n"
        "                        for request in refresh_requests\n"
        "                        if request.request_id not in committed_ids\n"
        "                    )\n"
        "                    if uncommitted_refresh:\n"
        "                        rebase_current(uncommitted_refresh)\n",
        "runtime commit/rebase exclusion",
    )
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


COMMIT_METHOD = '''    def _commit_batch_for_nano(
        self, updates: list[dict[str, Any]]
    ) -> None:
        """Send one ordered accepted/valid-length commit to Draft."""
        if not updates:
            return
        with self._nano_io_lock:
            response = self._request(
                {"cmd": "commit_batch", "updates": updates}
            )
        if response.get("status") != "result":
            raise RuntimeError(
                "Draft commit_batch returned an invalid response: "
                f"{response!r}"
            )
'''


def transform_proposer_v1(source: str) -> str:
    marker = MARKERS["proposer_v1"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_proposer_v1.py", "exec")
        return source
    if "NanoPearlPrefetchController" not in source:
        raise RuntimeError(
            "proposer_v1: nano-PEARL controller anchor is missing; "
            "no files were changed"
        )
    source = marker + "\n" + source
    source = replace_once(
        source,
        "            trace=trace_from_env(),\n"
        "        )\n",
        "            trace=trace_from_env(),\n"
        "            commit_batch=self._commit_batch_for_nano,\n"
        "        )\n",
        "proposer_v1 controller commit callback",
    )
    source = insert_before_method(
        source,
        "_request_batch_for_nano(",
        COMMIT_METHOD,
        "proposer_v1 commit method",
    )
    compile(source, "pearl_stage5_nanoparl_proposer_v1.py", "exec")
    return source


def transform_proposer_v3(source: str) -> str:
    marker = MARKERS["proposer_v3"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_proposer_v3.py", "exec")
        return source
    source = marker + "\n" + source

    # v3 has a custom controller constructor after the discard-rebase patch.
    # If it does not, v1's inherited constructor supplies the callback.
    custom_anchor = "            rebase_batch=self._rebase_batch_for_nano,\n"
    if custom_anchor in source:
        source = replace_once(
            source,
            custom_anchor,
            "            commit_batch=self._commit_batch_for_nano,\n"
            + custom_anchor,
            "proposer_v3 controller commit callback",
        )
        source = insert_before_method(
            source,
            "_rebase_batch_for_nano(",
            COMMIT_METHOD,
            "proposer_v3 commit method",
        )
    compile(source, "pearl_stage5_nanoparl_proposer_v3.py", "exec")
    return source


COMMIT_WORKER_BRANCH = '''                        if command == "commit_batch":
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
                                prefix = item.get("prefix_token_ids")
                                if not isinstance(prefix, list) or not prefix:
                                    raise ValueError(
                                        "commit prefix_token_ids must be non-empty"
                                    )
                                target_prefix_len = int(
                                    item.get("target_prefix_len", len(prefix))
                                )
                                valid_len = int(item.get("valid_len", -1))
                                if target_prefix_len != len(prefix):
                                    raise ValueError(
                                        "target_prefix_len does not match prefix length"
                                    )
                                if valid_len < 0 or valid_len > max(0, len(prefix) - 1):
                                    raise ValueError(
                                        "commit valid_len must be in [0, prefix_len-1]"
                                    )
                                updates.append(
                                    {
                                        "request_id": str(
                                            item.get("request_id", f"row-{index}")
                                        ),
                                        "prefix_token_ids": [int(x) for x in prefix],
                                        "gamma": int(item.get("gamma", 0)),
                                        "accepted_len": int(
                                            item.get("accepted_len", 0)
                                        ),
                                        "draft_len": int(item.get("draft_len", 0)),
                                        "valid_len": valid_len,
                                        "target_prefix_len": target_prefix_len,
                                        "replacement_token_id": (
                                            None
                                            if item.get("replacement_token_id") is None
                                            else int(item["replacement_token_id"])
                                        ),
                                        "finished": bool(item.get("finished", False)),
                                    }
                                )
                            result = commit_batch(updates)
                            print(
                                "[draft] commit_batch "
                                f"batch_size={len(updates)} "
                                f"accepted={sum(x['accepted_len'] for x in updates)} "
                                f"valid={sum(x['valid_len'] for x in updates)}",
                                flush=True,
                            )
                            send_message(
                                conn,
                                {"status": "result", "results": result or []},
                            )
                            continue

'''


def transform_worker(source: str) -> str:
    marker = MARKERS["worker"]
    if marker in source:
        compile(source, "pearl_stage5_worker.py", "exec")
        return source
    source = marker + "\n" + source
    source = replace_once(
        source,
        '                        if command == "draft_batch":\n',
        COMMIT_WORKER_BRANCH
        + '                        if command == "draft_batch":\n',
        "worker commit_batch command",
    )
    compile(source, "pearl_stage5_worker.py", "exec")
    return source


COMMIT_DRAFT_METHOD = '''    def commit_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Commit Target boundaries while retaining the persistent Draft KV.

        The fast path intentionally updates the live Request and runner row
        in place. valid_len is the exact num_computed_tokens value; the final
        Target token is left as the next token to compute. Token IDs are
        synchronized only to repair the small accepted/replacement suffix;
        no Request reset, block free, or row re-add occurs.
        """
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
            prefix = item.get("prefix_token_ids")
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(
                    f"commit prefix_token_ids must be non-empty for {external_id!r}"
                )
            prefix = [int(x) for x in prefix]
            target_prefix_len = int(item.get("target_prefix_len", len(prefix)))
            valid_len = int(item.get("valid_len", -1))
            if target_prefix_len != len(prefix):
                raise ValueError(
                    f"target_prefix_len mismatch for {external_id!r}"
                )
            if valid_len < 0 or valid_len > max(0, len(prefix) - 1):
                raise ValueError(
                    f"invalid valid_len={valid_len} for {external_id!r}"
                )
            accepted_len = int(item.get("accepted_len", 0))
            draft_len = int(item.get("draft_len", 0))
            if accepted_len < 0 or draft_len < 0 or accepted_len > draft_len:
                raise ValueError(
                    f"invalid accepted/draft lengths for {external_id!r}"
                )
            normalized.append(
                {
                    "request_id": external_id,
                    "prefix_token_ids": prefix,
                    "valid_len": valid_len,
                    "accepted_len": accepted_len,
                    "draft_len": draft_len,
                    "replacement_token_id": item.get("replacement_token_id"),
                    "finished": bool(item.get("finished", False)),
                }
            )

        results: list[dict[str, Any]] = []
        with self._lock:
            for item in normalized:
                external_id = item["request_id"]
                prefix = item["prefix_token_ids"]
                valid_len = item["valid_len"]
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

                # Keep the live Request/model-runner row. This is the same
                # state transition as the in-place rollback path, but driven
                # by an explicit batch accepted/valid-length message.
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
                if os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1":
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
    if "    def _replace_tokens(" not in source:
        raise RuntimeError(
            "draft: _replace_tokens() is missing; the validated persistent "
            "Draft engine is not present; no files were changed"
        )
    if "    def _sync_model_runner_state(" not in source:
        raise RuntimeError(
            "draft: _sync_model_runner_state() is missing; no files were "
            "changed"
        )
    source = marker + "\n" + source
    if "    def rebase_batch(" in source:
        source = insert_before_method(
            source,
            "rebase_batch(",
            COMMIT_DRAFT_METHOD,
            "draft commit method before rebase_batch",
        )
    elif "    def propose_batch(" in source:
        source = insert_before_method(
            source,
            "propose_batch(",
            COMMIT_DRAFT_METHOD,
            "draft commit method before propose_batch",
        )
    else:
        raise RuntimeError(
            "draft: neither rebase_batch() nor propose_batch() was found; "
            "no files were changed"
        )
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


TRANSFORMS = {
    Path("pearl_stage5_nanoparl_runtime_v1.py"): transform_runtime,
    Path("pearl_stage5_nanoparl_proposer_v1.py"): transform_proposer_v1,
    Path("pearl_stage5_nanoparl_proposer_v3.py"): transform_proposer_v3,
    Path("pearl_stage5_worker.py"): transform_worker,
    Path("pearl_stage5_draft.py"): transform_draft,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_atomic(path: Path, data: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.pearl_commit_state.",
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

    print("change: batch-level nano-PEARL accepted_len/valid_len commit")
    if not changed:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_commit_state_v1.{timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "mode": "batch-accepted-valid-length-commit",
        "enable_env": "PEARL_STAGE5_NANOPEARL_COMMIT_STATE=1",
        "async_scheduling": False,
        "changed_files": [str(path) for path in changed],
        "original_sha256": {
            str(path): sha256(originals[path]) for path in changed
        },
    }
    for relative in changed:
        saved = backup_dir / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(originals[relative])
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        for relative in changed:
            write_atomic(
                repo / relative,
                transformed[relative],
                modes[relative],
            )
    except BaseException:
        for relative in changed:
            try:
                (repo / relative).write_bytes(originals[relative])
            except BaseException:
                pass
        raise

    print(f"backup: {backup_dir}")
    for relative in changed:
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

