#!/usr/bin/env python3
"""Submit Stage-5 HCCL prompts as one real Target batch.

The HCCL transport path had a batch-capable proposer and bridge, but its
coordinator still called ``target_client.request(generate)`` once per prompt.
That made a two-prompt test run as two sequential ``batch_size=1`` requests.
This patch changes only the HCCL prompt loop to use
``prompt_token_ids_batch``.  The existing RPC path is left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path("pearl_stage5_coordinator.py")
MARKER = "# PEARL_STAGE5_HCCL_BATCH_MAINFLOW_V1"


OLD = '''            prompts = args.prompts or ["The capital of France is"]
            for prompt_index, prompt in enumerate(prompts, start=1):
                prompt_ids = [
                    int(x)
                    for x in tokenizer.encode(prompt, add_special_tokens=True)
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids": prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                print(
                    f"[stage5-hccl] prompt {prompt_index}: {prompt!r}; "
                    f"output_ids={response.get('token_ids', [])}; "
                    f"text={response.get('text')!r}; "
                    f"elapsed_ms={response.get('elapsed_ms')}",
                    flush=True,
                )
'''


NEW = '''            prompts = args.prompts or ["The capital of France is"]
            if args.max_num_seqs < 1:
                raise ValueError("--max-num-seqs must be positive")
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            if args.batch_size > args.max_num_seqs:
                raise ValueError(
                    "--batch-size cannot exceed --max-num-seqs"
                )

            for batch_start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[
                    batch_start : batch_start + args.batch_size
                ]
                batch_prompt_ids = [
                    [
                        int(x)
                        for x in tokenizer.encode(
                            prompt, add_special_tokens=True
                        )
                    ]
                    for prompt in batch_prompts
                ]
                response = target_client.request(
                    {
                        "cmd": "generate",
                        "prompt_token_ids_batch": batch_prompt_ids,
                        "max_tokens": args.max_tokens,
                    }
                )
                results = response.get("results")
                if not isinstance(results, list):
                    raise RuntimeError(
                        "HCCL Target batch response is missing 'results'"
                    )
                if len(results) != len(batch_prompts):
                    raise RuntimeError(
                        "HCCL Target batch result count mismatch: "
                        f"expected={len(batch_prompts)} got={len(results)}"
                    )
                for offset, (prompt, result) in enumerate(
                    zip(batch_prompts, results)
                ):
                    output_ids = [
                        int(x) for x in result.get("token_ids", [])
                    ]
                    print(
                        f"[stage5-hccl] prompt "
                        f"{batch_start + offset + 1}: {prompt!r}; "
                        f"output_ids={output_ids}; "
                        f"text={result.get('text')!r}; "
                        f"elapsed_ms={response.get('elapsed_ms')}",
                        flush=True,
                    )
'''


def transform(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("HCCL batch main-flow patch is already applied")

    # Batch v1 normally installed this argument.  Keep the patch usable when
    # only --max-num-seqs was carried into the current coordinator.
    if 'parser.add_argument("--batch-size"' not in source:
        anchor = '    parser.add_argument("--max-num-seqs", type=int, default=1)\n'
        if source.count(anchor) != 1:
            raise RuntimeError(
                "--batch-size parser anchor not found; no files were changed"
            )
        source = source.replace(
            anchor,
            anchor
            + '    parser.add_argument("--batch-size", type=int, default=1)\n',
            1,
        )

    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(
            "HCCL sequential prompt loop: expected one anchor, "
            f"found {count}; no files were changed"
        )
    transformed = source.replace(OLD, NEW, 1)
    transformed = MARKER + "\n" + transformed
    compile(transformed, str(TARGET), "exec")
    return transformed


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    original = target.read_text(encoding="utf-8")
    print(f"target: {target}")
    print(f"state: {'post' if MARKER in original else 'pre'}")
    print("change: submit HCCL prompts as one real Target batch")
    if MARKER in original:
        print("already patched: no files changed")
        return

    transformed = transform(original)
    if dry_run:
        digest = hashlib.sha256(original.encode()).hexdigest()[:12]
        print(f"dry-run: no files changed; source_sha256={digest}")
        return

    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_hccl_batch_mainflow_v1.{stamp}"
        )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup_dir / target.name)
    target.write_text(transformed, encoding="utf-8")
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
