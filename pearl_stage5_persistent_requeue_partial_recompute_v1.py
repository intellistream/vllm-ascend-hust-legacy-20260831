#!/usr/bin/env python3
"""Add an opt-in partial-tail recompute path to Stage-5 persistent requeue.

The validated fallback resets the Draft request whenever the common prefix
ends inside a KV block.  That is correct but discards all preceding blocks.
This patch adds a diagnostic path which keeps the complete blocks before the
partial tail and schedules the tail for recomputation from the aligned
boundary.  Prefixes shorter than one complete block still use the existing
fresh-request fallback.

Enable only for the comparison run with::

    PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE=1

The normal default remains unchanged.  Every real modification creates a new
full-file backup; ``--dry-run`` creates neither a backup nor a source change.
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


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE_V1"
REQUIRED_MARKERS = (
    "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK_V1",
    "PEARL_STAGE5_PERSISTENT_REQUEUE_FRESH_BLOCK_PROBE_V5",
)

FALLBACK_PREFIX = (
    "        if os.environ.get(\n"
        "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK\", "
)
FALLBACK_SUFFIX = (
    "\n        ) == \"1\":\n"
)

DROP_CALL_OLD = (
    "        if os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK\", \"0\"\n"
    "        ) == \"1\":\n"
)

DROP_CALL_NEW = (
    "        if (\n"
    "            os.environ.get(\n"
    "                \"PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK\",\n"
    "                \"0\",\n"
    "            ) == \"1\"\n"
    "            or partial_recompute_has_full_block\n"
    "        ):\n"
)

ASSIGN_OLD = (
    "        elif os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            request.num_computed_tokens = 0\n"
    "        else:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)

ASSIGN_NEW = (
    "        elif os.environ.get(\n"
    "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_FULL_RECOMPUTE\", \"0\"\n"
    "        ) == \"1\":\n"
    "            request.num_computed_tokens = 0\n"
    "        elif partial_recompute_has_full_block:\n"
    "            request.num_computed_tokens = (\n"
    "                reusable_tokens // partial_recompute_block_size\n"
    "            ) * partial_recompute_block_size\n"
    "            if os.environ.get(\n"
    "                \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
    "            ) == \"1\":\n"
    "                print(\n"
    "                    \"[PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE_V1] \"\n"
    "                    f\"request={request.request_id!r} \"\n"
    "                    f\"common_len={common_len} \"\n"
    "                    f\"reusable_tokens={reusable_tokens} \"\n"
    "                    f\"recompute_from={request.num_computed_tokens} \"\n"
    "                    f\"block_size={partial_recompute_block_size}\",\n"
    "                    flush=True,\n"
    "                )\n"
    "        else:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "        self._sync_model_runner_state(prefix_token_ids, request)\n"
)


def make_fallback_new(default_value: str) -> str:
    return (
        f"        # {MARKER}\n"
        "        partial_recompute = os.environ.get(\n"
        "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE\",\n"
        "            \"0\",\n"
        "        ) == \"1\"\n"
        "        reusable_tokens = max(0, common_len - 1)\n"
        "        partial_recompute_block_size = 0\n"
        "        if partial_recompute:\n"
        "            try:\n"
        "                partial_recompute_block_size = int(\n"
        "                    getattr(scheduler, \"block_size\")\n"
        "                )\n"
        "            except (AttributeError, TypeError, ValueError):\n"
        "                partial_recompute_block_size = 0\n"
        "        partial_recompute_has_full_block = (\n"
        "            partial_recompute\n"
        "            and partial_recompute_block_size > 0\n"
        "            and reusable_tokens >= partial_recompute_block_size\n"
        "        )\n"
        "\n"
        "        if os.environ.get(\n"
        "            \"PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_FALLBACK\", "
        f"\"{default_value}\"\n"
        "        ) == \"1\" and not partial_recompute_has_full_block:\n"
    )


def fallback_old(default_value: str) -> str:
    return FALLBACK_PREFIX + f'"{default_value}"' + FALLBACK_SUFFIX


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def inspect_state(source: str) -> tuple[str, str | None]:
    if MARKER in source:
        if DROP_CALL_NEW in source and ASSIGN_NEW in source:
            return "post", None
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )

    missing = [marker for marker in REQUIRED_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            f"{TARGET_REL}: required marker(s) missing: {missing}; "
            "no files were changed"
        )

    fallback_matches = []
    for default_value in ("0", "1"):
        anchor = fallback_old(default_value)
        if source.count(anchor):
            fallback_matches.append((default_value, source.count(anchor)))
    if len(fallback_matches) != 1 or fallback_matches[0][1] != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one fallback condition with default 0 or 1, "
            f"found {fallback_matches}; no files were changed"
        )
    if source.count(DROP_CALL_OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one partial-block call anchor, found "
            f"{source.count(DROP_CALL_OLD)}; no files were changed"
        )
    if source.count(ASSIGN_OLD) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one computed-token assignment anchor, found "
            f"{source.count(ASSIGN_OLD)}; no files were changed"
        )
    return "pre", fallback_matches[0][0]


def make_backup_dir(repo: Path, explicit: str | None) -> Path:
    backup_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_persistent_requeue_partial_recompute_v1."
        f"{timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
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


def apply_patch(repo: Path, backup_dir_arg: str | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state, fallback_default = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print(
        "change: retain complete KV blocks and recompute only the partial tail "
        "(opt-in)"
    )

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return
    assert fallback_default is not None

    patched = original.replace(
        fallback_old(fallback_default), make_fallback_new(fallback_default), 1
    )
    patched = patched.replace(DROP_CALL_OLD, DROP_CALL_NEW, 1)
    patched = patched.replace(ASSIGN_OLD, ASSIGN_NEW, 1)
    if (
        patched == original
        or MARKER not in patched
        or DROP_CALL_NEW not in patched
        or ASSIGN_NEW not in patched
    ):
        raise RuntimeError("internal replacement failed; no files were changed")
    compile(patched, str(target), "exec")

    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = make_backup_dir(repo, backup_dir_arg)
    backup_file = backup_dir / target.name
    shutil.copy2(target, backup_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_script": Path(__file__).name,
        "repo": str(repo),
        "target": str(target),
        "backup_file": str(backup_file),
        "original_sha256": sha256_bytes(original_bytes),
        "original_size": len(original_bytes),
        "marker": MARKER,
        "mode": "opt-in-aligned-full-block-reuse-with-partial-tail-recompute",
        "enable_env": "PEARL_STAGE5_PERSISTENT_REQUEUE_PARTIAL_RECOMPUTE=1",
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        write_atomic(target, patched, target.stat().st_mode)
    except Exception:
        shutil.copy2(backup_file, target)
        raise
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
