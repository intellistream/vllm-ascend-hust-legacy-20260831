#!/usr/bin/env python3
"""Fix stale cross-request Draft outputs in the Stage-5 batch bridge.

The v1 batch bridge buffered Draft outputs belonging to another request.  A
later Target verification changes that request's prefix, so those buffered
tokens are stale.  Consuming them also skips ``EngineCore.get_output()`` and
leaves the request in WAITING state.  v2 keeps Target verification batched but
drives the currently selected Draft request afresh on every token.
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


MARKER_V1 = "# PEARL_STAGE5_BATCH_GT1_V1"
MARKER_V2 = "# PEARL_STAGE5_BATCH_GT1_V2"
TARGET = Path("pearl_stage5_draft.py")


STEP_METHOD = '''
    def _step_one(self) -> int:
        if self.request_id is None:
            raise RuntimeError("Draft request has not been created")

        # Do not consume a token buffered while another request was active.
        # The Target prefix may have changed since that token was produced.
        # Driving get_output() here forces the current request through the
        # scheduler after every persistent requeue, so it cannot remain in
        # WAITING state merely because another batch row produced output.
        while True:
            outputs = self.core_client.get_output()
            for output in outputs.outputs:
                output_request_id = str(output.request_id)
                if output_request_id != self.request_id:
                    continue
                new_token_ids = getattr(output, "new_token_ids", None) or []
                if new_token_ids:
                    return int(new_token_ids[-1])
                if output.finished:
                    raise RuntimeError(
                        "Draft request finished before returning a token: "
                        f"{output.finish_reason}"
                    )

            if self.request_id not in self.core.scheduler.requests:
                raise RuntimeError("Draft request disappeared while decoding")
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one anchor, found {count}; no files were changed"
        )
    return source.replace(old, new, 1)


def replace_method(source: str, method_name: str, next_method: str) -> str:
    start = source.find(f"    def {method_name}")
    end = source.find(f"    def {next_method}", start + 1)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"{method_name}: method boundary not found; no files were changed"
        )
    return source[:start] + STEP_METHOD.rstrip() + "\n\n" + source[end:]


def transform(source: str) -> str:
    if MARKER_V2 in source:
        raise RuntimeError("batch v2 marker already exists")
    if MARKER_V1 not in source:
        raise RuntimeError(
            "pearl_stage5_draft.py is not in batch v1 state; no files were changed"
        )

    source = MARKER_V2 + "\n" + source
    source = replace_once(
        source,
        "        if prefix_token_ids != self.committed_token_ids:\n",
        "        if prefix_token_ids != self.committed_token_ids:\n"
        "            # Any queued output was generated from the old Target prefix.\n"
        "            self._pending_tokens.pop(self.request_id, None)\n",
        "clear stale batch token anchor",
    )
    source = replace_method(source, "_step_one(", "propose(")
    compile(source, "pearl_stage5_draft.py", "exec")
    return source


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_batch_gt1_v2.{stamp}"
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
    if MARKER_V2 in source:
        print(f"target: {target}")
        print("state: post")
        print("already patched: no files were changed and no backup was created")
        return
    transformed = transform(source)
    print(f"target: {target}")
    print("state: pre-v2")
    print("change: discard stale cross-request Draft outputs in batch mode")
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
        "marker": MARKER_V2,
        "original_sha256": sha256_bytes(original),
        "original_size": len(original),
        "target": str(target),
        "backup_file": str(backup_file),
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
