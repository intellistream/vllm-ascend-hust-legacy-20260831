#!/usr/bin/env python3
"""Install a narrow raw-vs-parsed speculative-sampling trace.

The trace is disabled by default and is enabled at runtime with
PEARL_STAGE5_SAMPLE_OUTPUT_TRACE=1. It records only the first request row,
its shape, and a short prefix of token IDs at two points:

1. immediately after AscendRejectionSampler returns;
2. immediately after vLLM's parse_output() filters the sampler result.

This script edits only the specified model-runner file. A real patch first
copies the complete original file into a fresh timestamped backup directory.
--dry-run never creates a backup and never writes files.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


MARKER = "# PEARL_STAGE5_SAMPLE_OUTPUT_TRACE_V1"
TARGET_RELATIVE = Path("vllm_ascend/worker/model_runner_v1.py")

SAMPLER_ANCHOR = '''        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
'''

SAMPLER_REPLACEMENT = '''        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        # PEARL_STAGE5_SAMPLE_OUTPUT_TRACE_V1
        if __import__("os").environ.get("PEARL_STAGE5_SAMPLE_OUTPUT_TRACE") == "1":
            _raw_ids = sampler_output.sampled_token_ids.detach().cpu().tolist()
            _raw_row = _raw_ids[0] if _raw_ids else []
            print(
                "[PEARL_STAGE5_SAMPLE_OUTPUT_TRACE] raw "
                f"shape={tuple(sampler_output.sampled_token_ids.shape)} "
                f"draft_counts={getattr(spec_decode_metadata, 'num_draft_tokens', None)} "
                f"row0={_raw_row[:16]}",
                flush=True,
            )
        return sampler_output
'''

PARSE_ANCHOR = '''                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
'''

PARSE_REPLACEMENT = '''                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
            # PEARL_STAGE5_SAMPLE_OUTPUT_TRACE_V1
            if __import__("os").environ.get("PEARL_STAGE5_SAMPLE_OUTPUT_TRACE") == "1":
                _parsed_raw_ids = sampled_token_ids.detach().cpu().tolist()
                _parsed_raw_row = _parsed_raw_ids[0] if _parsed_raw_ids else []
                _parsed_row = (
                    valid_sampled_token_ids[0]
                    if valid_sampled_token_ids
                    else []
                )
                print(
                    "[PEARL_STAGE5_SAMPLE_OUTPUT_TRACE] parsed "
                    f"raw_shape={tuple(sampled_token_ids.shape)} "
                    f"raw_row0={_parsed_raw_row[:16]} "
                    f"parsed_len={len(_parsed_row)} "
                    f"parsed_row0={_parsed_row[:16]}",
                    flush=True,
                )
        else:
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def backup_path(repo: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = repo.parent / f"{repo.name}.pearl_stage5_sample_output_trace_v1.{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def inspect_state(text: str) -> str:
    if text.count(MARKER) >= 2:
        return "post"
    if MARKER in text:
        raise RuntimeError("trace marker appears partially; no files were changed")
    if SAMPLER_ANCHOR not in text:
        raise RuntimeError("sampler anchor not found; no files were changed")
    if PARSE_ANCHOR not in text:
        raise RuntimeError("parse_output anchor not found; no files were changed")
    return "pre"


def patched_text(text: str) -> str:
    result = text.replace(SAMPLER_ANCHOR, SAMPLER_REPLACEMENT, 1)
    result = result.replace(PARSE_ANCHOR, PARSE_REPLACEMENT, 1)
    if result == text or result.count(MARKER) != 2:
        raise RuntimeError("patch anchors did not produce exactly two trace markers")
    return result


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    target = repo / TARGET_RELATIVE
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: raw sampler output + parse_output trace (runtime env-gated)")

    if state == "post":
        print("already patched: no files were changed and no backup was created")
        return 0
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    backup = backup_path(repo, args.backup_dir)
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    backup_target = backup / TARGET_RELATIVE
    backup_target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_target)
    print(f"backup: {backup}")

    target.write_text(patched_text(original), encoding="utf-8")
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
