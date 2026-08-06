#!/usr/bin/env python3
"""Backup and apply the nano-PEARL Stage-5 Ascend integration.

The script changes only two vLLM-Ascend files. It first verifies that both
files are still in the expected pre-patch state, saves their complete current
contents to a new timestamped backup directory, and then applies the edits.
If writing either file fails, both files are restored from that backup.

Usage:
    python apply_pearl_stage5_patch.py --repo /root/data/vllm-ascend-hust
    python apply_pearl_stage5_patch.py --repo . --dry-run
    python apply_pearl_stage5_patch.py --repo . --restore-from /path/to/backup
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


TARGETS = {
    Path("vllm_ascend/spec_decode/__init__.py"): (
        """    elif method == "extract_hidden_states":
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
""",
        """    elif method == "extract_hidden_states":
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
    elif method == "custom_class":
        # Keep the common vLLM custom proposer contract on Ascend.  The
        # proposer is intentionally model-free; it can obtain draft IDs from
        # an external Draft worker (for example, nano-PEARL over AF_UNIX).
        from vllm.v1.spec_decode.custom_class_proposer import (
            create_custom_proposer,
        )

        return create_custom_proposer(vllm_config)
""",
    ),
    Path("vllm_ascend/worker/model_runner_v1.py"): (
        """        elif self.speculative_config.method in ("ngram", "suffix"):
            draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
""",
        """        elif self.speculative_config.method in ("ngram", "suffix"):
            draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
        elif self.speculative_config.method == "custom_class":
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
""",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, args: list[str]) -> str | None:
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


def make_backup(repo: Path, backup_arg: Path | None) -> Path:
    if backup_arg is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_backup.{stamp}"
    else:
        backup_dir = backup_arg.expanduser().resolve()

    if backup_dir.exists():
        raise FileExistsError(
            f"backup path already exists; refusing to overwrite: {backup_dir}"
        )

    backup_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "repo": str(repo),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_branch": git_value(repo, ["branch", "--show-current"]),
        "git_commit": git_value(repo, ["rev-parse", "HEAD"]),
        "files": {},
    }

    try:
        for relative_path in TARGETS:
            source = repo / relative_path
            destination = backup_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["files"][str(relative_path)] = {
                "sha256": sha256(source),
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


def restore_backup(repo: Path, backup_dir: Path) -> None:
    backup_dir = backup_dir.expanduser().resolve()
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"backup directory does not exist: {backup_dir}")

    for relative_path in TARGETS:
        source = backup_dir / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"missing backup file: {source}")

    for relative_path in TARGETS:
        source = backup_dir / relative_path
        destination = repo / relative_path
        shutil.copy2(source, destination)
        print(f"restored {relative_path} from {backup_dir}")


def inspect_state(repo: Path) -> dict[Path, str]:
    state: dict[Path, str] = {}
    for relative_path, (old, new) in TARGETS.items():
        path = repo / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"target file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        has_new = new in text
        has_old = old in text
        # The post-patch block intentionally contains the original block as
        # its prefix, so check the new block first.
        if has_new:
            state[relative_path] = "patched"
        elif has_old:
            state[relative_path] = "original"
        else:
            raise RuntimeError(
                f"{path} does not match the expected pre-patch or post-patch code; "
                "no files were changed"
            )
    return state


def apply_patch(repo: Path, backup_arg: Path | None, dry_run: bool) -> None:
    state = inspect_state(repo)
    states = set(state.values())
    if states == {"patched"}:
        print("Stage-5 Ascend patch is already applied; no changes made.")
        return
    if states != {"original"}:
        raise RuntimeError(
            "the two target files are in different states; refusing a partial "
            f"patch: {state}"
        )

    if dry_run:
        print("dry-run: both target files match the expected pre-patch state")
        for relative_path in TARGETS:
            print(f"would modify {relative_path}")
        return

    backup_dir = make_backup(repo, backup_arg)
    original_text: dict[Path, str] = {}
    try:
        for relative_path, (old, new) in TARGETS.items():
            path = repo / relative_path
            text = path.read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise RuntimeError(
                    f"expected exactly one replacement block in {path}, "
                    f"found {text.count(old)}"
                )
            original_text[relative_path] = text
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
    except Exception:
        for relative_path, text in original_text.items():
            (repo / relative_path).write_text(text, encoding="utf-8")
        print(
            "apply failed; already-written files were restored in place",
            file=sys.stderr,
        )
        raise

    print(f"backup saved to: {backup_dir}")
    for relative_path in TARGETS:
        print(f"modified: {relative_path}")
    print("apply completed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="vllm-ascend-hust repository root; default: current directory",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="explicit new backup directory; it must not already exist",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restore-from",
        type=Path,
        default=None,
        help="restore the two files from a previous backup directory",
    )
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    if args.restore_from is not None:
        restore_backup(repo, args.restore_from)
    else:
        apply_patch(repo, args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())