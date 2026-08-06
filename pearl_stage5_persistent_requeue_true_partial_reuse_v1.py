#!/usr/bin/env python3
"""Enable an opt-in true partial-tail reuse path for Stage-5 Draft.

The current Stage-5 diagnostic intentionally drops a partial tail block and
recomputes from the previous full-block boundary.  This patch adds a separate
opt-in path that keeps the request-owned tail block and resumes from the exact
common-prefix position.

This is deliberately gated until the runtime trace proves that the fork's
scheduler and model runner safely handle an unaligned computed-token count:

    PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE=1

The patch targets the already-installed partial-recompute v1 state.  It does
not change the default behavior.
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

import pearl_stage5_persistent_requeue_partial_recompute_v1 as base


TARGET_REL = Path("pearl_stage5_draft.py")
MARKER = "# PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_BLOCK_REUSE_V1"
ENV_NAME = "PEARL_STAGE5_PERSISTENT_REQUEUE_TRUE_PARTIAL_REUSE"

TRUE_FLAG = (
    "        true_partial_reuse = os.environ.get(\n"
    f"            {ENV_NAME!r}, \"0\"\n"
    "        ) == \"1\" and reusable_tokens > 0\n"
)

TRACE_BRANCH = (
    "        elif true_partial_reuse:\n"
    "            request.num_computed_tokens = max(0, common_len - 1)\n"
    "            if os.environ.get(\n"
    "                \"PEARL_STAGE5_PERSISTENT_REQUEUE_TRACE\", \"0\"\n"
    "            ) == \"1\":\n"
    "                print(\n"
    "                    \"[PEARL_STAGE5_TRUE_PARTIAL_BLOCK_REUSE_V1] \"\n"
    "                    f\"request={request.request_id!r} \"\n"
    "                    f\"common_len={common_len} \"\n"
    "                    f\"reusable_tokens={reusable_tokens} \"\n"
    "                    f\"num_computed_tokens={request.num_computed_tokens} \"\n"
    "                    \"action=retain_partial_tail\",\n"
    "                    flush=True,\n"
    "                )\n"
)

# The v1 drop guard is changed so the opt-in true-reuse path owns the partial
# block instead of releasing it.
DROP_TRUE = base.DROP_CALL_NEW.replace(
    "            or partial_recompute_has_full_block\n"
    "        ):\n",
    "            or partial_recompute_has_full_block\n"
    "        ) and not true_partial_reuse:\n",
)

# Insert the exact-position branch before the aligned recompute branch.
ASSIGN_TRUE = base.ASSIGN_NEW.replace(
    "        elif partial_recompute_has_full_block:\n",
    TRACE_BRANCH + "        elif partial_recompute_has_full_block:\n",
    1,
)


def _fallback_block(default_value: str) -> str:
    return base.make_fallback_new(default_value)


def _true_fallback_block(default_value: str) -> str:
    old = _fallback_block(default_value)
    marker_line = f"        # {base.MARKER}\n"
    if old.count(marker_line) != 1:
        raise RuntimeError(
            "partial-recompute fallback block has drifted; "
            "no files were changed"
        )

    new = old.replace(
        marker_line,
        f"        {MARKER}\n" + marker_line,
        1,
    )
    reusable_line = "        reusable_tokens = max(0, common_len - 1)\n"
    if new.count(reusable_line) != 1:
        raise RuntimeError(
            "reusable-token anchor is missing; no files were changed"
        )
    new = new.replace(reusable_line, reusable_line + TRUE_FLAG, 1)
    new = new.replace(
        ") == \"1\" and not partial_recompute_has_full_block:\n",
        ") == \"1\" and not partial_recompute_has_full_block "
        "and not true_partial_reuse:\n",
        1,
    )
    return new


def _find_fallback(source: str) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    for default_value in ("0", "1"):
        block = _fallback_block(default_value)
        count = source.count(block)
        if count:
            matches.append((default_value, block))
            if count != 1:
                raise RuntimeError(
                    f"{TARGET_REL}: fallback block occurs {count} times; "
                    "no files were changed"
                )
    if len(matches) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one installed partial-recompute fallback "
            f"block, found {len(matches)}; no files were changed"
        )
    return matches[0]


def inspect_state(source: str) -> str:
    if MARKER in source:
        if (
            source.count(MARKER) == 1
            and DROP_TRUE in source
            and ASSIGN_TRUE in source
            and any(
                source.count(_true_fallback_block(value)) == 1
                for value in ("0", "1")
            )
        ):
            return "post"
        raise RuntimeError(
            f"{TARGET_REL}: {MARKER} is present but the complete patch is "
            "missing; no files were changed"
        )

    if base.MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: the partial-recompute v1 patch is not installed; "
            "apply pearl_stage5_persistent_requeue_partial_recompute_v1.py "
            "first; no files were changed"
        )
    if base.DROP_CALL_NEW not in source or base.ASSIGN_NEW not in source:
        raise RuntimeError(
            f"{TARGET_REL}: partial-recompute v1 anchors are incomplete; "
            "no files were changed"
        )
    _find_fallback(source)
    return "pre"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_backup_dir(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        backup_dir = explicit.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_true_partial_block_reuse_v1."
            f"{stamp}"
        )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: "
            f"{backup_dir}"
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


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print(
        "change: opt-in true partial-block reuse; retain tail and resume "
        "from exact common-prefix position"
    )

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return

    fallback_default, fallback_old = _find_fallback(original)
    fallback_new = _true_fallback_block(fallback_default)

    patched = original.replace(fallback_old, fallback_new, 1)
    patched = patched.replace(base.DROP_CALL_NEW, DROP_TRUE, 1)
    patched = patched.replace(base.ASSIGN_NEW, ASSIGN_TRUE, 1)
    if (
        patched == original
        or MARKER not in patched
        or DROP_TRUE not in patched
        or ASSIGN_TRUE not in patched
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
        "mode": "opt-in-true-partial-block-reuse",
        "enable_env": f"{ENV_NAME}=1",
        "default_behavior": "unchanged",
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
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
