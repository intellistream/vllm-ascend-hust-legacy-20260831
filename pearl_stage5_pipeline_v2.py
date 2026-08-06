#!/usr/bin/env python3
"""Make Stage-5 lookahead reuse row-wise instead of batch all-or-nothing.

This patch is applied after ``pearl_stage5_pipeline_v1.py``.  v1 required
every row in a Target batch to match its Draft lookahead before reusing any
row.  That is too strict for batch=128: one rejected row forced the whole
batch back through synchronous Draft generation.

v2 keeps compatible rows on their prefetched stream, rebases only the
incompatible rows, and lets one EngineCore step service both sets.  The
pipeline remains opt-in through ``PEARL_STAGE5_PIPELINE=1``.
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
REQUIRED_MARKER = "# PEARL_STAGE5_PIPELINE_V1"
MARKER = "# PEARL_STAGE5_PIPELINE_V2"


MIXED_METHODS = dedent(
    '''
        def propose_batch(
            self,
            requests: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            """Reuse compatible rows and rebase only mismatched rows."""
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
            if not pipeline:
                self._pipeline_candidates.clear()

            with self._lock:
                fast_rows: dict[str, dict[str, Any]] = {}
                if pipeline:
                    for external_id, prefix, gamma in normalized:
                        if self._pipeline_candidate_can_serve(
                            external_id, prefix, gamma
                        ):
                            fast_rows[external_id] = self._pipeline_candidates[
                                external_id
                            ]

                fallback_count = len(normalized) - len(fast_rows)
                if pipeline:
                    self._pipeline_trace(
                        "row_partition "
                        f"batch={len(normalized)} "
                        f"reuse_rows={len(fast_rows)} "
                        f"fallback_rows={fallback_count}"
                    )

                internal_to_external: dict[str, str] = {}
                target_counts: dict[str, int] = {}
                draft_by_id: dict[str, list[int]] = {}

                for external_id, prefix, gamma in normalized:
                    self._activate_request(external_id)
                    candidate = fast_rows.get(external_id)
                    if candidate is not None:
                        generated = candidate["generated"]
                        draft_by_id[external_id] = [
                            int(token_id)
                            for token_id in generated[
                                len(prefix) : len(prefix) + gamma
                            ]
                        ]
                        internal_id = str(candidate["internal_id"])
                        internal_to_external[internal_id] = external_id
                        # A compatible row already has its current proposal.
                        # Count zero here: if EngineCore advances this row while
                        # serving fallback rows, its output is captured into
                        # _pending_tokens and appended below.
                        target_counts[external_id] = 0
                        continue

                    if pipeline:
                        self._pipeline_candidates.pop(external_id, None)
                    self.sync_prefix(prefix)
                    internal_id = self.request_id
                    if internal_id is None:
                        raise RuntimeError(
                            f"Draft request was not created for {external_id!r}"
                        )
                    internal_id = str(internal_id)
                    # Discard output that belongs to the old Target prefix.
                    self._pending_tokens.pop(internal_id, None)
                    internal_to_external[internal_id] = external_id
                    target_counts[external_id] = gamma

                if any(count > 0 for count in target_counts.values()):
                    collected = self._collect_batch_tokens(
                        internal_to_external=internal_to_external,
                        target_counts=target_counts,
                    )
                else:
                    collected = {
                        external_id: [] for external_id, _prefix, _gamma in normalized
                    }

                base_generated: dict[str, list[int]] = {}
                for external_id, prefix, gamma in normalized:
                    internal_id = next(
                        key
                        for key, value in internal_to_external.items()
                        if value == external_id
                    )
                    candidate = fast_rows.get(external_id)
                    if candidate is not None:
                        generated = list(candidate["generated"])
                    else:
                        generated = list(prefix) + list(
                            collected.get(external_id, [])
                        )
                    # Capture any token produced for a fast row while the
                    # fallback rows were being advanced, and any output beyond
                    # the requested gamma for a fallback row.
                    pending = self._pending_tokens.pop(internal_id, None)
                    if pending:
                        generated.extend(int(token_id) for token_id in pending)

                    # Fallback rows were synchronously advanced above.  Their
                    # current proposal must be returned even when the
                    # row-wise pipeline is enabled; otherwise a mixed batch
                    # would silently return an empty proposal for every
                    # rebased row.
                    if candidate is None:
                        draft_by_id[external_id] = [
                            int(token_id)
                            for token_id in collected.get(external_id, [])[:gamma]
                        ]

                    if pipeline:
                        if candidate is None:
                            candidate = {
                                "internal_id": internal_id,
                                "base_prefix": list(prefix),
                                "generated": generated,
                            }
                            self._pipeline_candidates[external_id] = candidate
                        else:
                            candidate["generated"] = generated
                        base_generated[external_id] = list(generated)
                    elif candidate is None:
                        draft_by_id[external_id] = [
                            int(token_id)
                            for token_id in collected.get(external_id, [])[:gamma]
                        ]

                results = [
                    {
                        "request_id": external_id,
                        "draft_token_ids": draft_by_id.get(
                            external_id,
                            [],
                        )[:gamma],
                    }
                    for external_id, _prefix, gamma in normalized
                ]
                if not pipeline:
                    return results

                # Start a fresh lookahead for every current row.  Compatible
                # rows continue from their already-generated stream; rebased
                # rows continue from the new Target prefix.
                lookahead = {
                    external_id: self._pipeline_lookahead(gamma)
                    for external_id, _prefix, gamma in normalized
                    if gamma > 0
                }
                self._start_pipeline_prefetch(
                    internal_to_external,
                    lookahead,
                    base_generated,
                )
                return results
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


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("pipeline v2 marker already exists")
    if REQUIRED_MARKER not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py is not in pipeline v1 state; no files were changed"
        )
    source = MARKER + "\n" + source
    indented = "\n".join(
        "    " + line if line else ""
        for line in MIXED_METHODS.strip("\n").splitlines()
    )
    source = replace_method(source, "propose_batch(", "propose(", indented)
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_pipeline_v2.{stamp}"
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
    print("change: row-wise Draft lookahead reuse for mixed batch results")
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
        "mode": "row_wise_pipeline_reuse_with_batch_fallback_rows",
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
