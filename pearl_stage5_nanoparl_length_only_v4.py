#!/usr/bin/env python3
"""Use a strict accepted_len/valid_len wire payload for nano-PEARL.

For eligible rows the Target-side runtime now sends only the request ID,
accepted length, valid length, and the length-only discriminator.  Draft
derives ``target_prefix_len = valid_len + 1`` and reconstructs the remaining
bookkeeping from its resident KV/speculative state.  Non-eligible rows keep
the existing full-prefix commit path.
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
    "runtime": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V4",
    "worker": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V4",
}

TARGETS = (
    Path("pearl_stage5_nanoparl_runtime_v1.py"),
    Path("pearl_stage5_worker.py"),
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


RUNTIME_COMMIT_V4 = '''    def _commit_current(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        length_only_ids: set[str] | None = None,
    ) -> set[str]:
        """Commit boundaries; strict rows carry only accepted/valid lengths."""
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
            if length_only:
                # Strict length-only payload.  The receiver derives the
                # target boundary from valid_len and uses its resident Draft
                # state for all token/KV bookkeeping.
                update = {
                    "request_id": request.request_id,
                    "accepted_len": result.accepted_len,
                    "valid_len": valid_len,
                    "length_only": True,
                }
            else:
                update = {
                    "request_id": request.request_id,
                    "gamma": request.gamma,
                    "accepted_len": result.accepted_len,
                    "draft_len": result.draft_len,
                    "valid_len": valid_len,
                    "target_prefix_len": len(request.prefix_token_ids),
                    "replacement_token_id": result.replacement_token_id,
                    "finished": result.finished,
                    "length_only": False,
                    "prefix_token_ids": list(request.prefix_token_ids),
                }
            updates.append(update)
            committed_ids.add(request.request_id)

        if not updates:
            return set()

        self._trace(
            "commit_batch_start "
            f"round={self.round_id} batch={len(updates)} "
            f"accepted={sum(item['accepted_len'] for item in updates)} "
            f"valid={sum(item['valid_len'] for item in updates)} "
            f"length_only={sum(bool(item['length_only']) for item in updates)} "
            "wire_length_fields=accepted_len,valid_len"
        )
        self._commit_batch(updates)
        self._trace(
            "commit_batch_done "
            f"round={self.round_id} batch={len(updates)} "
            f"rows={','.join(sorted(committed_ids))}"
        )
        return committed_ids
'''


def transform_runtime(source: str) -> str:
    marker = MARKERS["runtime"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V2" not in source:
        raise RuntimeError(
            "runtime: length-only v2 is missing; apply v2 first; "
            "no files were changed"
        )
    source = replace_method(
        source,
        "_commit_current",
        RUNTIME_COMMIT_V4,
        "strict length-only runtime commit",
    )
    source = marker + "\n" + source
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


def transform_worker(source: str) -> str:
    marker = MARKERS["worker"]
    if marker in source:
        compile(source, "pearl_stage5_worker.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V2" not in source:
        raise RuntimeError(
            "worker: length-only v2 is missing; apply v2 first; "
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
                                valid_len = int(item.get("valid_len", -1))
                                accepted_len = int(item.get("accepted_len", 0))
                                if length_only:
                                    # Strict wire form: no prefix or other
                                    # target-side token metadata is accepted.
                                    if prefix is not None:
                                        raise ValueError(
                                            "strict length-only commit must not "
                                            "carry prefix_token_ids"
                                        )
                                    if valid_len < 0:
                                        raise ValueError(
                                            "strict length-only valid_len must be non-negative"
                                        )
                                    target_prefix_len = valid_len + 1
                                    # Draft's verifier contract requires an
                                    # all-accepted length-only row.  The
                                    # runtime eligibility check established
                                    # that property before sending this row.
                                    draft_len = accepted_len
                                    replacement = None
                                    finished = False
                                    gamma = 0
                                else:
                                    if not isinstance(prefix, list) or not prefix:
                                        raise ValueError(
                                            "commit prefix_token_ids must be non-empty"
                                        )
                                    target_prefix_len = int(
                                        item.get(
                                            "target_prefix_len", len(prefix)
                                        )
                                    )
                                    draft_len = int(item.get("draft_len", 0))
                                    replacement = item.get("replacement_token_id")
                                    finished = bool(item.get("finished", False))
                                    gamma = int(item.get("gamma", 0))
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
                                    "gamma": gamma,
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
                                f"length_only={sum(bool(x['length_only']) for x in updates)} "
                                "wire_length_fields=accepted_len,valid_len",
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


TRANSFORMS = {
    Path("pearl_stage5_nanoparl_runtime_v1.py"): transform_runtime,
    Path("pearl_stage5_worker.py"): transform_worker,
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
        prefix=f".{path.name}.pearl_length_only_v4.",
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
    modes: dict[Path, int] = {}
    changed: list[Path] = []

    for relative, transform_fn in TRANSFORMS.items():
        target = repo / relative
        raw = target.read_bytes()
        original = raw.decode("utf-8")
        updated = transform_fn(original)
        originals[relative] = raw
        transformed[relative] = updated
        modes[relative] = target.stat().st_mode
        print(f"target: {target}")
        print(f"state: {'post' if updated == original else 'pre'}")
        if updated != original:
            changed.append(relative)

    print("change: strict accepted_len/valid_len length-only nano-PEARL wire")
    if not changed:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_length_only_v4.{timestamp()}"
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
    for relative in changed:
        write_atomic(repo / relative, transformed[relative], modes[relative])
        print(f"patched: {repo / relative}")
    print(f"backup: {backup_dir}")


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
