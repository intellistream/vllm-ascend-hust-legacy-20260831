#!/usr/bin/env python3
"""Robust v2 installer for the nano-PEARL physical KV ownership trace.

v1 required an exact indentation/location for the in-place branch comment.
Different Stage-5 patch chains can move that comment while keeping the
accepted_len/runner-sync logic identical.  v2 therefore anchors on the code
statements themselves and remains observability-only.

The v1 installer must already exist in the same repository because v2 reuses
its non-mutating snapshot helper and safe target-only backup utilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pearl_stage5_nanoparl_kv_ownership_trace_v1 as base


TARGET_REL = base.TARGET_REL
INPLACE_MARKER = base.INPLACE_MARKER
TRACE_MARKER = base.TRACE_MARKER
HELPER = base.HELPER


def _method_span(source: str) -> tuple[int, int]:
    start = source.find("    def _requeue_request_preserve_kv(")
    if start < 0:
        raise RuntimeError(
            f"{TARGET_REL}: cannot find _requeue_request_preserve_kv(); "
            "no files were changed"
        )
    if source.find("    def _requeue_request_preserve_kv(", start + 1) >= 0:
        raise RuntimeError(
            f"{TARGET_REL}: found multiple _requeue_request_preserve_kv() "
            "methods; no files were changed"
        )
    next_method = source.find("\n    def ", start + 5)
    end = len(source) if next_method < 0 else next_method + 1
    return start, end


def transform(source: str) -> str:
    if TRACE_MARKER in source:
        compile(source, str(TARGET_REL), "exec")
        return source

    if INPLACE_MARKER not in source:
        raise RuntimeError(
            f"{TARGET_REL}: in-place rollback patch is not present; "
            "apply pearl_stage5_nanoparl_inplace_rollback_v1.py first; "
            "no files were changed"
        )

    method_start, method_end = _method_span(source)
    method = source[method_start:method_end]
    lines = method.splitlines(keepends=True)

    accepted_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "accepted_len = common_len"
    ]
    if len(accepted_indices) != 1:
        raise RuntimeError(
            f"{TARGET_REL}: expected one accepted_len anchor, found "
            f"{len(accepted_indices)}; no files were changed"
        )
    accepted_index = accepted_indices[0]
    accepted_indent = lines[accepted_index][
        : len(lines[accepted_index]) - len(lines[accepted_index].lstrip())
    ]

    sync_indices = [
        index
        for index, line in enumerate(lines)
        if index > accepted_index
        and line.strip()
        == "self._sync_model_runner_state(prefix_token_ids, request)"
    ]
    if not sync_indices:
        raise RuntimeError(
            f"{TARGET_REL}: no runner-state sync after accepted_len anchor; "
            "no files were changed"
        )
    sync_index = sync_indices[0]
    sync_indent = lines[sync_index][
        : len(lines[sync_index]) - len(lines[sync_index].lstrip())
    ]

    before_lines = [
        accepted_indent + "kv_before = self._nanoparl_kv_ownership_snapshot(\n",
        accepted_indent + "    request.request_id\n",
        accepted_indent + ")\n",
    ]
    lines[accepted_index:accepted_index] = before_lines
    sync_index += len(before_lines)

    after_lines = [
        lines[sync_index],
        sync_indent + "kv_after = self._nanoparl_kv_ownership_snapshot(\n",
        sync_indent + "    request.request_id\n",
        sync_indent + ")\n",
        sync_indent + "if (\n",
        sync_indent + "    os.environ.get(\"PEARL_STAGE5_NANOPEARL_KV_TRACE\", \"0\")\n",
        sync_indent + "    == \"1\"\n",
        sync_indent + "    or os.environ.get(\"PEARL_STAGE5_NANOPEARL_TRACE\", \"0\")\n",
        sync_indent + "    == \"1\"\n",
        sync_indent + "):\n",
        sync_indent + "    print(\n",
        sync_indent + "        \"[PEARL_STAGE5_NANOPEARL_KV_OWNERSHIP_TRACE_V1] \"\n",
        sync_indent + "        + repr({\n",
        sync_indent + "            \"request\": request.request_id,\n",
        sync_indent + "            \"accepted_len\": accepted_len,\n",
        sync_indent + "            \"valid_len\": valid_len,\n",
        sync_indent + "            \"before\": kv_before,\n",
        sync_indent + "            \"after\": kv_after,\n",
        sync_indent + "            \"same_manager_block_ids\": (\n",
        sync_indent + "                kv_before[\"manager_block_ids\"]\n",
        sync_indent + "                == kv_after[\"manager_block_ids\"]\n",
        sync_indent + "            ),\n",
        sync_indent + "            \"same_runner_block_ids\": (\n",
        sync_indent + "                kv_before[\"runner_block_ids\"]\n",
        sync_indent + "                == kv_after[\"runner_block_ids\"]\n",
        sync_indent + "            ),\n",
        sync_indent + "        }),\n",
        sync_indent + "        flush=True,\n",
        sync_indent + "    )\n",
    ]
    lines[sync_index : sync_index + 1] = after_lines

    updated_method = "".join(lines)
    transformed = (
        source[:method_start]
        + HELPER
        + updated_method
        + source[method_end:]
    )
    compile(transformed, str(TARGET_REL), "exec")
    return transformed


def apply_patch(repo: Path, backup_dir_arg: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    transformed = transform(original)

    print(f"target: {target}")
    print("state: post-inplace rollback")
    print("change: robust non-mutating physical KV ownership trace v2")
    if transformed == original:
        print("already patched: no files were changed and no backup was created")
        return
    if dry_run:
        print("dry-run: no files were changed and no backup was created")
        return

    backup_dir = (
        backup_dir_arg.expanduser().resolve()
        if backup_dir_arg is not None
        else repo.parent
        / f"{repo.name}.pearl_stage5_nanoparl_kv_ownership_trace_v2."
        f"{base._timestamp()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True)
    import shutil

    shutil.copy2(target, backup_dir / TARGET_REL.name)
    print(f"backup: {backup_dir}")
    base._write_atomic(target, transformed, target.stat().st_mode)
    print(f"patched: {target}")
    print(f"source_sha256_before: {base._sha256(original_bytes)[:12]}")
    print(
        "source_sha256_after: "
        f"{base._sha256(transformed.encode('utf-8'))[:12]}"
    )


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
