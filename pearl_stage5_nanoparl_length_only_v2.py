#!/usr/bin/env python3
"""Broaden length-only eligibility to an already-prefetched bonus token.

v1 intentionally required replacement_token_id=None.  The Stage-5 verifier
can report a bonus/replacement token even when that token is already the first
token of Draft's optimistic look-ahead.  In that case no new token ID has to
cross the process boundary: the Draft persistent sequence already contains
the exact target prefix.

This v2 keeps the safety rule:

    current_prefix = pending_prefix + prefix_of(prefetched_draft)

If that rule fails, the existing full-prefix commit path is retained.
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
    "runtime": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V2",
    "worker": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V2",
    "draft": "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V2",
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


RUNTIME_ELIGIBILITY_V2 = '''    def _eligible_length_only_ids(
        self,
        current: tuple[DraftRequest, ...],
        explicit: dict[str, VerifyResult],
        pending_by_id: dict[str, tuple[DraftRequest, DraftResult]],
        pending_error: Exception | None,
    ) -> set[str]:
        """Return rows whose Target prefix is already in Draft KV state."""
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
            draft = old_result.draft_token_ids
            if result.finished or not result.all_accepted:
                continue
            if result.draft_len != len(draft):
                continue
            if request.gamma != old.gamma:
                continue
            if (
                len(request.prefix_token_ids) < len(old.prefix_token_ids)
                or request.prefix_token_ids[: len(old.prefix_token_ids)]
                != old.prefix_token_ids
            ):
                continue

            extra = request.prefix_token_ids[len(old.prefix_token_ids) :]
            if len(extra) > len(draft):
                continue
            if tuple(extra) != tuple(draft[: len(extra)]):
                continue

            # A replacement/bonus is safe only when the current Target
            # prefix already contains the same prefetched token.  If Target
            # has not appended it yet, extra is empty and no token transfer
            # is needed in this round.
            if (
                result.replacement_token_id is not None
                and extra
                and int(extra[0]) != int(result.replacement_token_id)
            ):
                continue
            eligible.add(request.request_id)

        if eligible:
            self._trace(
                "length_only_eligible "
                f"round={self.round_id} batch={len(eligible)} "
                f"rows={','.join(sorted(eligible))}"
            )
        return eligible
'''


def transform_runtime(source: str) -> str:
    marker = MARKERS["runtime"]
    if marker in source:
        compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_RUNTIME_V1" not in source:
        raise RuntimeError(
            "runtime: length-only v1 is missing; apply v1 first; "
            "no files were changed"
        )
    source = replace_method(
        source,
        "_eligible_length_only_ids",
        RUNTIME_ELIGIBILITY_V2,
        "runtime length-only eligibility",
    )
    source = marker + "\n" + source
    compile(source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")
    return source


def transform_worker(source: str) -> str:
    marker = MARKERS["worker"]
    if marker in source:
        compile(source, "pearl_stage5_worker.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_WORKER_V1" not in source:
        raise RuntimeError(
            "worker: length-only v1 is missing; apply v1 first; "
            "no files were changed"
        )
    source = replace_once(
        source,
        "                                    if (\n"
        "                                        finished\n"
        "                                        or accepted_len != draft_len\n"
        "                                        or replacement is not None\n"
        "                                    ):\n",
        "                                    if (\n"
        "                                        finished\n"
        "                                        or accepted_len != draft_len\n"
        "                                    ):\n",
        "worker length-only replacement validation",
    )
    source = marker + "\n" + source
    compile(source, "pearl_stage5_worker.py", "exec")
    return source


def transform_draft(source: str) -> str:
    marker = MARKERS["draft"]
    if marker in source:
        compile(source, "pearl_stage5_draft.py", "exec")
        return source
    if "# PEARL_STAGE5_NANOPEARL_LENGTH_ONLY_DRAFT_V1" not in source:
        raise RuntimeError(
            "draft: length-only v1 is missing; apply v1 first; "
            "no files were changed"
        )
    source = replace_once(
        source,
        "                if (\n"
        "                    finished\n"
        "                    or accepted_len != draft_len\n"
        "                    or replacement is not None\n"
        "                ):\n",
        "                if (\n"
        "                    finished\n"
        "                    or accepted_len != draft_len\n"
        "                ):\n",
        "draft length-only replacement validation",
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
        prefix=f".{path.name}.pearl_length_only_v2.",
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

    print("change: allow already-prefetched bonus in length-only nano-PEARL")
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
        / f"{repo.name}.pearl_stage5_nanoparl_length_only_v2.{timestamp()}"
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
