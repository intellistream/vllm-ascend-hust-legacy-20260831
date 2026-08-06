#!/usr/bin/env python3
"""Repair local Draft bookkeeping after a nano-PEARL length-only commit.

v2 correctly enabled the length-only wire protocol, but the Draft-side
``committed_token_ids`` cache was not advanced after the local Runner lengths
were updated.  The next round could therefore report ``have=N, need=N+1``
even though the token was already resident in the Draft speculative state.

This patch only repairs Draft-local state.  It does not add token IDs to the
length-only RPC.  The extra IDs are taken from the Draft model runner's own
speculative token state or token buffer.  If neither local source can prove
the requested boundary, the existing safety exception remains enabled.
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


TARGET = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V3"


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


SYNC_METHOD_V3 = '''    def _sync_model_runner_lengths_only(
        self,
        request: Any,
        target_prefix_len: int,
    ) -> None:
        """Advance lengths and Draft-local bookkeeping without RPC token IDs."""
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

        # The length-only RPC deliberately carries no token IDs.  When the
        # Target boundary advances into Draft's optimistic look-ahead, adopt
        # those IDs from the already resident Draft row before clearing its
        # speculative placeholders.  This also keeps committed_token_ids in
        # sync across consecutive length-only rounds.
        committed = [int(x) for x in self.committed_token_ids]
        adopted = 0
        if len(committed) < target_prefix_len:
            needed = target_prefix_len - len(committed)
            candidates: list[int] = []

            spec_rows = getattr(input_batch, "spec_token_ids", None)
            if spec_rows is not None:
                try:
                    values = spec_rows[req_index]
                    if hasattr(values, "tolist"):
                        values = values.tolist()
                    if isinstance(values, (list, tuple)):
                        candidates = [int(x) for x in values if int(x) >= 0]
                except (IndexError, TypeError, ValueError):
                    candidates = []

            if len(candidates) < needed:
                token_ids_cpu = getattr(input_batch, "token_ids_cpu", None)
                if token_ids_cpu is not None:
                    try:
                        values = token_ids_cpu[
                            req_index, len(committed):target_prefix_len
                        ]
                        if hasattr(values, "tolist"):
                            values = values.tolist()
                        if isinstance(values, (list, tuple)):
                            cpu_candidates = [
                                int(x) for x in values if int(x) >= 0
                            ]
                            if len(cpu_candidates) >= needed:
                                candidates = cpu_candidates
                    except (IndexError, TypeError, ValueError):
                        pass

            if len(candidates) >= needed:
                committed.extend(candidates[:needed])
                adopted = needed

        if len(committed) < target_prefix_len:
            raise RuntimeError(
                "length-only local Draft state is shorter than the requested "
                f"boundary for {self.request_id!r}: "
                f"have={len(committed)} need={target_prefix_len}"
            )

        self.committed_token_ids = committed[:target_prefix_len]
        self.prompt_token_ids = list(self.committed_token_ids)
        request.prompt_token_ids = list(self.committed_token_ids)
        request.num_prompt_tokens = target_prefix_len
        if getattr(request, "prompt_is_token_ids", None) is not None:
            request.prompt_is_token_ids = [True] * target_prefix_len

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

        if os.environ.get("PEARL_STAGE5_NANOPEARL_TRACE", "0") == "1":
            print(
                "[PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_V3] "
                f"request_id={self.request_id!r} "
                f"target_prefix_len={target_prefix_len} "
                f"adopted_local_tokens={adopted} "
                "action=update_lengths_only_keep_kv",
                flush=True,
            )
'''


OLD_LENGTH_ONLY_GUARD = '''                if item["length_only"]:
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
'''


NEW_LENGTH_ONLY_GUARD = '''                if item["length_only"]:
                    self._sync_model_runner_lengths_only(
                        request,
                        target_prefix_len,
                    )
                    request.is_prefill_chunk = False
                    request.spec_token_ids = []
                    request.num_output_placeholders = 0
'''


def transform(source: str) -> str:
    if MARKER in source:
        compile(source, "pearl_stage5_draft.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V2" not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py: length-only v2 is missing; apply v2 first; "
            "no files were changed"
        )
    source = replace_method(
        source,
        "_sync_model_runner_lengths_only",
        SYNC_METHOD_V3,
        "Draft length-only local-state sync",
    )
    source = replace_once(
        source,
        OLD_LENGTH_ONLY_GUARD,
        NEW_LENGTH_ONLY_GUARD,
        "Draft length-only boundary guard",
    )
    source = MARKER + "\n" + source
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


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
        prefix=f".{path.name}.pearl_length_only_v3.",
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
    target = repo / TARGET
    original = target.read_text(encoding="utf-8")
    updated = transform(original)

    print(f"target: {target}")
    print(f"state: {'post' if updated == original else 'pre'}")
    print("change: repair Draft-local length-only bookkeeping")
    if updated == original:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_length_only_v3.{timestamp()}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    copy_backup_file(target, backup_dir / TARGET)
    (backup_dir / "MANIFEST.sha256").write_text(
        f"{sha256(original.encode('utf-8'))}  {TARGET}\n",
        encoding="utf-8",
    )
    write_atomic(target, updated, target.stat().st_mode)
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


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
