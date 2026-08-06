#!/usr/bin/env python3
"""Add an opt-in, correctness-first Draft/Target lookahead pipeline.

The existing Stage-5 batch path already batches Draft model execution, but
the Target custom proposer waits for the whole Draft RPC.  This patch keeps
that path as the default and adds ``PEARL_STAGE5_PIPELINE=1``:

* return the requested ``gamma`` Draft tokens;
* continue the same Draft requests in a background thread for a short
  lookahead window while Target verifies the returned tokens;
* reuse the lookahead only when the next Target prefix is an exact prefix of
  the already-generated Draft stream;
* otherwise discard the speculative stream and use the existing synchronous
  batch path for the whole batch.

It patches only ``pearl_stage5_draft.py`` and requires the already validated
``PEARL_STAGE5_BATCH_GT1_V3`` Draft implementation.  The pipeline is opt-in
so the validated batch=128 path remains unchanged until explicitly tested.
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


TARGET = Path("pearl_stage5_draft.py")
REQUIRED_MARKER = "# PEARL_STAGE5_BATCH_GT1_V3"
MARKER = "# PEARL_STAGE5_PIPELINE_V1"
AC_SAFE_MARKER = "# PEARL_STAGE5_PIPELINE_AC_SAFE_V1"


PIPELINE_METHODS = dedent(
    '''
        def _pipeline_enabled(self) -> bool:
            if os.environ.get("PEARL_STAGE5_PIPELINE", "0") != "1":
                return False

            explicit = os.environ.get("PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE")
            if explicit is not None:
                return explicit == "1"

            # Nano-PEARL's runtime controller already overlaps Target verification
            # with the next Draft request.  The older Draft-side lookahead layer
            # changes proposal trajectories in commit-state runs and lowers GSM8K
            # AC, so keep it opt-in until that path is made verify-result aware.
            if os.environ.get("PEARL_STAGE5_NANOPEARL_COMMIT_STATE", "0") == "1":
                return False

            return True

        def _pipeline_trace(self, message: str) -> None:
            if os.environ.get("PEARL_STAGE5_PIPELINE_TRACE", "0") == "1":
                print(
                    "[PEARL_STAGE5_PIPELINE_V1] " + message,
                    flush=True,
                )

        def _pipeline_lookahead(self, gamma: int) -> int:
            raw = os.environ.get("PEARL_STAGE5_PIPELINE_LOOKAHEAD")
            if raw is None:
                # Need gamma+1 extra tokens to cover the common case where
                # Target accepts all gamma tokens and its bonus token equals
                # the first lookahead token.
                return max(1, int(gamma) + 1)
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(
                    "PEARL_STAGE5_PIPELINE_LOOKAHEAD must be an integer"
                ) from exc
            if value < 0:
                raise ValueError(
                    "PEARL_STAGE5_PIPELINE_LOOKAHEAD must be non-negative"
                )
            return value

        def _join_pipeline(self) -> None:
            thread = self._pipeline_thread
            if thread is not None:
                # Join outside self._lock: the worker acquires the same lock
                # around EngineCore access.
                thread.join()
                self._pipeline_thread = None
            error = self._pipeline_error
            self._pipeline_error = None
            if error is not None:
                self._pipeline_candidates.clear()
                self._pipeline_trace(
                    "prefetch_error=" + repr(error) + " action=fallback"
                )

        def _pipeline_prefetch_worker(
            self,
            internal_to_external: dict[str, str],
            target_counts: dict[str, int],
            base_generated: dict[str, list[int]],
        ) -> None:
            try:
                with self._lock:
                    collected = self._collect_batch_tokens(
                        internal_to_external=internal_to_external,
                        target_counts=target_counts,
                    )
                    for internal_id, external_id in internal_to_external.items():
                        candidate = self._pipeline_candidates.get(external_id)
                        if candidate is None:
                            continue
                        if candidate["generated"] != base_generated[external_id]:
                            # A future code path changed the candidate while
                            # this worker was running.  Never append tokens to
                            # a different prefix.
                            continue
                        candidate["generated"].extend(
                            int(token_id)
                            for token_id in collected.get(external_id, [])
                        )
                        pending = self._pending_tokens.pop(internal_id, None)
                        if pending:
                            candidate["generated"].extend(int(x) for x in pending)
                    self._pipeline_trace(
                        "prefetch_done "
                        f"batch={len(internal_to_external)} "
                        f"tokens={sum(len(value) for value in collected.values())}"
                    )
            except BaseException as exc:
                self._pipeline_error = exc

        def _start_pipeline_prefetch(
            self,
            internal_to_external: dict[str, str],
            target_counts: dict[str, int],
            base_generated: dict[str, list[int]],
        ) -> None:
            positive_counts = {
                external_id: count
                for external_id, count in target_counts.items()
                if count > 0
            }
            if not positive_counts:
                self._pipeline_thread = None
                return
            filtered_mapping = {
                internal_id: external_id
                for internal_id, external_id in internal_to_external.items()
                if external_id in positive_counts
            }
            filtered_base = {
                external_id: list(base_generated[external_id])
                for external_id in positive_counts
            }
            self._pipeline_trace(
                "prefetch_start "
                f"batch={len(filtered_mapping)} "
                f"lookahead={sum(positive_counts.values())}"
            )
            self._pipeline_thread = threading.Thread(
                target=self._pipeline_prefetch_worker,
                args=(filtered_mapping, positive_counts, filtered_base),
                name="pearl-stage5-draft-prefetch",
                daemon=True,
            )
            self._pipeline_thread.start()

        def _pipeline_candidate_can_serve(
            self,
            external_id: str,
            prefix: list[int],
            gamma: int,
        ) -> bool:
            candidate = self._pipeline_candidates.get(external_id)
            if candidate is None:
                return False
            internal_id = str(candidate["internal_id"])
            state = self._states.get(external_id)
            if state is None or str(state.get("request_id")) != internal_id:
                return False
            generated = candidate["generated"]
            if len(prefix) < len(candidate["base_prefix"]):
                return False
            if generated[: len(prefix)] != prefix:
                return False
            return len(generated) >= len(prefix) + int(gamma)

        def propose_batch(
            self,
            requests: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            """Draft a batch, optionally overlapping a bounded lookahead."""
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

            pipeline = self._pipeline_enabled()
            self._join_pipeline()

            with self._lock:
                if pipeline and all(
                    self._pipeline_candidate_can_serve(external_id, prefix, gamma)
                    for external_id, prefix, gamma in normalized
                ):
                    results = []
                    internal_to_external: dict[str, str] = {}
                    target_counts: dict[str, int] = {}
                    base_generated: dict[str, list[int]] = {}
                    for external_id, prefix, gamma in normalized:
                        candidate = self._pipeline_candidates[external_id]
                        generated = candidate["generated"]
                        draft_ids = generated[len(prefix) : len(prefix) + gamma]
                        self._activate_request(external_id)
                        internal_to_external[str(candidate["internal_id"])] = external_id
                        target_counts[external_id] = self._pipeline_lookahead(gamma)
                        base_generated[external_id] = list(generated)
                        results.append(
                            {
                                "request_id": external_id,
                                "draft_token_ids": [int(x) for x in draft_ids],
                            }
                        )
                    self._pipeline_trace(
                        "reuse batch="
                        f"{len(normalized)} action=consume_prefetch"
                    )
                    self._start_pipeline_prefetch(
                        internal_to_external,
                        target_counts,
                        base_generated,
                    )
                    return results

                if pipeline and self._pipeline_candidates:
                    self._pipeline_trace(
                        "fallback batch="
                        f"{len(normalized)} reason=prefix_mismatch_or_short_tail"
                    )
                self._pipeline_candidates.clear()

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

                results = [
                    {
                        "request_id": external_id,
                        "draft_token_ids": collected.get(external_id, [])[:gamma],
                    }
                    for external_id, _prefix, gamma in normalized
                ]

                if not pipeline:
                    return results

                lookahead = {
                    external_id: self._pipeline_lookahead(gamma)
                    for external_id, _prefix, gamma in normalized
                    if gamma > 0
                }
                base_generated: dict[str, list[int]] = {}
                for external_id, prefix, gamma in normalized:
                    internal_id = next(
                        key
                        for key, value in internal_to_external.items()
                        if value == external_id
                    )
                    generated = list(prefix) + list(collected.get(external_id, []))
                    pending = self._pending_tokens.pop(internal_id, None)
                    if pending:
                        generated.extend(int(x) for x in pending)
                    base_generated[external_id] = generated
                    candidate = {
                        "internal_id": internal_id,
                        "base_prefix": list(prefix),
                        "generated": generated,
                    }
                    self._pipeline_candidates[external_id] = candidate
                    already_generated = max(
                        0,
                        len(generated) - len(prefix) - int(gamma),
                    )
                    lookahead[external_id] = max(
                        0, lookahead[external_id] - already_generated
                    )
                self._start_pipeline_prefetch(
                    internal_to_external,
                    lookahead,
                    base_generated,
                )
                return results
    '''
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(source: str, method_name: str, next_method: str, block: str) -> str:
    start = source.find(f"    def {method_name}")
    end = source.find(f"    def {next_method}", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method_name}: method boundary not found; no files were changed"
        )
    return source[:start] + block.strip("\n") + "\n\n" + source[end:]


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("pipeline marker already exists")
    if REQUIRED_MARKER not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py is not in batch v3 state; no files were changed"
        )
    source = AC_SAFE_MARKER + "\n" + MARKER + "\n" + source
    source = replace_once(
        source,
        "        self._pending_tokens: dict[str, deque[int]] = {}\n"
        "        self._request_counter = 0\n",
        "        self._pending_tokens: dict[str, deque[int]] = {}\n"
        "        self._pipeline_candidates: dict[str, dict[str, Any]] = {}\n"
        "        self._pipeline_thread: threading.Thread | None = None\n"
        "        self._pipeline_error: BaseException | None = None\n"
        "        self._request_counter = 0\n",
        "pipeline state anchor",
    )
    indented = "\n".join(
        "    " + line if line else ""
        for line in PIPELINE_METHODS.strip("\n").splitlines()
    )
    source = replace_method(source, "propose_batch(", "propose(", indented)
    # Extend shutdown with a safe join before it touches EngineCore state.
    source = replace_once(
        source,
        "    def shutdown(self) -> None:\n        with self._lock:\n",
        "    def shutdown(self) -> None:\n"
        "        thread = self._pipeline_thread\n"
        "        if thread is not None:\n"
        "            thread.join()\n"
        "            self._pipeline_thread = None\n"
        "        with self._lock:\n",
        "pipeline shutdown join anchor",
    )
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_pipeline_v1.{stamp}"
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
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_bytes()
    source = original.decode("utf-8")
    if MARKER in source:
        print(f"target: {target}")
        print("state: post")
        print("already patched: no files were changed and no backup was created")
        return
    transformed = transform(source)
    print(f"target: {target}")
    print("state: pre")
    print("change: opt-in Draft lookahead pipeline with synchronous fallback")
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_arg)
    backup_file = backup_dir / TARGET
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "marker": MARKER,
        "mode": "opt_in_draft_lookahead_pipeline",
        "files": {
            str(TARGET): {
                "target": str(target),
                "backup_file": str(backup_file),
                "original_sha256": sha256_bytes(original),
                "original_size": len(original),
            }
        },
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        write_atomic(target, transformed, target.stat().st_mode)
    except Exception:
        shutil.copy2(backup_file, target)
        raise
    print(f"backup: {backup_dir}")
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
