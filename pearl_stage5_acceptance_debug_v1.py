#!/usr/bin/env python3
"""Safely add Stage-5 acceptance logging to vLLM's rejection sampler.

The debug branch is enabled only when PEARL_ACCEPTANCE_DEBUG=1, so the
default execution path is unchanged.  It prints per-step and cumulative:

    draft_len, accepted_len, acceptance rate, mean acceptance length

The rejection sampler output contains accepted draft tokens plus one recovered
or bonus token.  Therefore accepted draft tokens are computed as
``min(draft_len, valid_output_tokens - 1)``.

The target file is in the vllm core repository, not vllm-ascend-hust:
    /root/data/vllm-hust/vllm/v1/sample/rejection_sampler.py

Examples:
    python pearl_stage5_acceptance_debug_v1.py \
        --repo /root/data/vllm-hust --dry-run
    python pearl_stage5_acceptance_debug_v1.py \
        --repo /root/data/vllm-hust
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET = Path("vllm/v1/sample/rejection_sampler.py")
PATCH_MARKER = "# PEARL_ACCEPTANCE_DEBUG_V1"
CALL_ANCHOR = "        output_token_ids = rejection_sample("
CALL_END_ANCHOR = "        logprobs_tensors = None"

DEBUG_INSERT = '''        # PEARL_ACCEPTANCE_DEBUG_V1
        # ``output_token_ids`` contains accepted draft tokens plus one
        # recovered/bonus token.  The latter is not a draft token.
        if os.environ.get("PEARL_ACCEPTANCE_DEBUG", "0") == "1":
            valid_counts = (
                (output_token_ids != PLACEHOLDER_TOKEN_ID)
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            draft_lens = [int(x) for x in metadata.num_draft_tokens]
            accepted_lens = [
                max(0, min(draft_len, int(valid_count) - 1))
                for draft_len, valid_count in zip(draft_lens, valid_counts)
            ]
            debug_rows = [
                (index, draft_len, accepted_len)
                for index, (draft_len, accepted_len) in enumerate(
                    zip(draft_lens, accepted_lens)
                )
                if draft_len > 0
            ]
            if debug_rows:
                round_draft = sum(row[1] for row in debug_rows)
                round_accepted = sum(row[2] for row in debug_rows)
                self._pearl_debug_rounds = (
                    getattr(self, "_pearl_debug_rounds", 0) + len(debug_rows)
                )
                self._pearl_debug_draft_tokens = (
                    getattr(self, "_pearl_debug_draft_tokens", 0) + round_draft
                )
                self._pearl_debug_accepted_tokens = (
                    getattr(self, "_pearl_debug_accepted_tokens", 0)
                    + round_accepted
                )
                total_draft = self._pearl_debug_draft_tokens
                total_accepted = self._pearl_debug_accepted_tokens
                total_rounds = self._pearl_debug_rounds
                round_rate = (
                    100.0 * round_accepted / round_draft
                    if round_draft > 0
                    else float("nan")
                )
                total_rate = (
                    100.0 * total_accepted / total_draft
                    if total_draft > 0
                    else float("nan")
                )
                mean_acceptance_length = (
                    1.0 + total_accepted / total_rounds
                    if total_rounds > 0
                    else float("nan")
                )
                print(
                    "[PEARL_ACCEPTANCE_DEBUG] "
                    f"rows={debug_rows} "
                    f"round_draft={round_draft} "
                    f"round_accepted={round_accepted} "
                    f"round_rate={round_rate:.2f}% "
                    f"total_draft={total_draft} "
                    f"total_accepted={total_accepted} "
                    f"total_rate={total_rate:.2f}% "
                    f"mean_acceptance_length={mean_acceptance_length:.3f}",
                    flush=True,
                )
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-hust"),
        help="vllm core repository root",
    )
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore-from", type=Path, default=None)
    return parser


def transform(text: str) -> tuple[str, str]:
    if PATCH_MARKER in text:
        return text, "already-patched"

    call_start = text.find(CALL_ANCHOR)
    call_end = text.find(CALL_END_ANCHOR, call_start)
    if call_start < 0 or call_end < 0:
        raise RuntimeError(
            f"{TARGET} does not contain the expected rejection_sample anchors; "
            "no files were changed"
        )

    if "\nimport os\n" not in text:
        future_line = "from __future__ import annotations\n"
        future_end = text.find(future_line)
        if future_end < 0:
            raise RuntimeError(
                f"{TARGET}: future-import anchor not found; no files were changed"
            )
        future_end += len(future_line)
        text = text[:future_end] + "\nimport os\n" + text[future_end:]
        call_start = text.find(CALL_ANCHOR)
        call_end = text.find(CALL_END_ANCHOR, call_start)

    return text[:call_end] + DEBUG_INSERT + text[call_end:], "to-patch"


def save_backup(target: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_file = backup_dir / TARGET.name
    shutil.copy2(target, backup_file)
    manifest = {
        "target": str(target),
        "backup_file": str(backup_file),
        "sha256_before": sha256(target),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def restore(repo: Path, backup_dir: Path) -> None:
    source = backup_dir / TARGET.name
    if not source.is_file():
        raise FileNotFoundError(f"backup file does not exist: {source}")
    target = repo / TARGET
    shutil.copy2(source, target)
    print(f"restored: {target}")
    print(f"from:    {source}")


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(f"target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")
    updated, state = transform(original)
    print(f"current state: {TARGET}: {state}")

    if state == "already-patched":
        print("no changes needed")
        return
    if dry_run:
        print("dry-run passed; no files changed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if backup_dir is None:
        backup_dir = repo.parent / f"{repo.name}.pearl_stage5_acceptance_backup_v1.{timestamp}"
    save_backup(target, backup_dir)

    temporary = target.with_name(target.name + ".pearl_stage5_tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", newline="")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(backup_dir / TARGET.name, target)
        raise

    print(f"backup saved to: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve()
    if args.restore_from is not None:
        restore(repo, args.restore_from.resolve())
    else:
        apply_patch(
            repo,
            args.backup_dir.resolve() if args.backup_dir else None,
            args.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
