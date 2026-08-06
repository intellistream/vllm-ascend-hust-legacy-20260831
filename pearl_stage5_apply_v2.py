#!/usr/bin/env python3
"""Safely apply the nano-PEARL Stage-5 Ascend changes.

This is a standalone v2 applicator. It uses stable source anchors, saves full
pre-change copies of both target files, and restores files if writing fails.

Examples:
    python pearl_stage5_apply_v2.py --repo /root/data/vllm-ascend-hust --dry-run
    python pearl_stage5_apply_v2.py --repo /root/data/vllm-ascend-hust
    python pearl_stage5_apply_v2.py --repo . --restore-from /path/to/backup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SPEC_PATH = Path("vllm_ascend/spec_decode/__init__.py")
RUNNER_PATH = Path("vllm_ascend/worker/model_runner_v1.py")
TARGETS = (SPEC_PATH, RUNNER_PATH)

SPEC_MARKER = '    elif method == "custom_class":'
SPEC_ANCHOR = '    elif method == "extract_hidden_states":'
SPEC_INSERT = """    elif method == "custom_class":
        # Keep the common vLLM custom proposer contract on Ascend.  The
        # proposer is intentionally model-free; it can obtain draft IDs from
        # an external Draft worker (for example, nano-PEARL over AF_UNIX).
        from vllm.v1.spec_decode.custom_class_proposer import (
            create_custom_proposer,
        )

        return create_custom_proposer(vllm_config)
"""

RUNNER_MARKER = (
    '        elif self.speculative_config.method == "custom_class":'
)
RUNNER_ANCHOR = (
    "        elif isinstance(self.drafter, AscendMedusaProposer):"
)
RUNNER_INSERT = """        elif self.speculative_config.method == "custom_class":
            # The common custom proposer API is CPU-side and is called after
            # bookkeeping.  It returns one list of draft token IDs per active
            # request.  No target hidden states or draft-model forward pass is
            # needed for an external nano-PEARL Draft worker.
            assert isinstance(valid_sampled_token_ids, list), (
                "sampled_token_ids should be a python list for custom_class"
            )
            assert self.drafter is not None
            draft_token_ids = self.drafter.propose(
                valid_sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=None,
            )
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def transform_spec(text: str) -> tuple[str, str]:
    if SPEC_MARKER in text:
        return text, "already-patched"

    anchor = text.find(SPEC_ANCHOR)
    if anchor < 0:
        raise RuntimeError(
            f"{SPEC_PATH}: stable anchor not found; no files were changed"
        )

    dispatch_else = text.find("\n    else:", anchor)
    if dispatch_else < 0:
        raise RuntimeError(
            f"{SPEC_PATH}: dispatch else branch not found; no files were changed"
        )

    insert_at = dispatch_else + 1
    return text[:insert_at] + SPEC_INSERT + text[insert_at:], "to-patch"


def transform_runner(text: str) -> tuple[str, str]:
    if RUNNER_MARKER in text:
        return text, "already-patched"

    anchor = text.find(RUNNER_ANCHOR)
    ngram = text.find(
        '        elif self.speculative_config.method in ("ngram", "suffix"):'
    )
    if anchor < 0 or ngram < 0 or ngram > anchor:
        raise RuntimeError(
            f"{RUNNER_PATH}: stable anchors not found; no files were changed"
        )

    return text[:anchor] + RUNNER_INSERT + text[anchor:], "to-patch"


def read_states(repo: Path) -> tuple[dict[Path, str], dict[Path, str]]:
    texts: dict[Path, str] = {}
    states: dict[Path, str] = {}

    for relative_path in TARGETS:
        path = repo / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"target file does not exist: {path}")

        text = path.read_text(encoding="utf-8")
        if relative_path == SPEC_PATH:
            _, state = transform_spec(text)
        else:
            _, state = transform_runner(text)
        texts[relative_path] = text
        states[relative_path] = state

    return texts, states


def create_backup(
    repo: Path,
    backup_arg: Path | None,
    states: dict[Path, str],
) -> Path:
    if backup_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_backup_v2.{timestamp}"
        )
    else:
        backup_dir = backup_arg.expanduser().resolve()

    if backup_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing backup: {backup_dir}"
        )

    backup_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "repo": str(repo),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_branch": git_info(repo, ["branch", "--show-current"]),
        "git_commit": git_info(repo, ["rev-parse", "HEAD"]),
        "states_before_apply": {str(k): v for k, v in states.items()},
        "files": {},
    }

    try:
        for relative_path in TARGETS:
            source = repo / relative_path
            destination = backup_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["files"][str(relative_path)] = {
                "sha256": file_sha256(source),
                "size": source.stat().st_size,
            }

        (backup_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(backup_dir)
        raise

    return backup_dir


def restore(repo: Path, backup_dir: Path) -> None:
    backup_dir = backup_dir.expanduser().resolve()
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"backup directory does not exist: {backup_dir}")

    for relative_path in TARGETS:
        source = backup_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"missing backup file: {source}")

    for relative_path in TARGETS:
        shutil.copy2(
            backup_dir / relative_path,
            repo / relative_path,
        )
        print(f"restored: {relative_path}")


def apply(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    original_text, states = read_states(repo)

    print("current state:")
    for relative_path in TARGETS:
        print(f"  {relative_path}: {states[relative_path]}")

    if all(state == "already-patched" for state in states.values()):
        print("Stage-5 changes are already applied; no files changed.")
        return

    if dry_run:
        print("dry-run passed; no files changed.")
        return

    backup_dir = create_backup(repo, backup_arg, states)

    try:
        for relative_path in TARGETS:
            if states[relative_path] == "already-patched":
                continue

            if relative_path == SPEC_PATH:
                new_text, _ = transform_spec(original_text[relative_path])
            else:
                new_text, _ = transform_runner(original_text[relative_path])

            (repo / relative_path).write_text(new_text, encoding="utf-8")
    except Exception:
        for relative_path, text in original_text.items():
            (repo / relative_path).write_text(text, encoding="utf-8")
        print(
            "apply failed; all target files restored in place",
            file=sys.stderr,
        )
        raise

    print(f"backup saved to: {backup_dir}")
    print("Stage-5 changes applied successfully.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="vllm-ascend-hust repository root",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="new backup directory; it must not already exist",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restore-from",
        type=Path,
        default=None,
        help="restore both target files from a previous backup",
    )
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    if args.restore_from is not None:
        restore(repo, args.restore_from)
    else:
        apply(repo, args.backup_dir, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

