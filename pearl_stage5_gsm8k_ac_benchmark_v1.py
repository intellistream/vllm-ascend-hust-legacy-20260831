# PEARL_STAGE5_TARGET_ASYNC_BENCH_OPTIN_V1
# PEARL_STAGE5_BATCH_GT1_V1
#!/usr/bin/env python3
"""Measure PEARL Stage-5 acceptance on the same GSM8K prompts as baseline."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


AC_RE = re.compile(
    r"\[PEARL_ACCEPTANCE_DEBUG(?:_V[0-9]+)?\].*?"
    r"total_draft=(\d+).*?total_accepted=(\d+)"
)


def load_prompts(path: Path, limit: int) -> list[str]:
    frame = pd.read_parquet(path)
    prompts: list[str] = []
    for index in range(min(limit, len(frame))):
        row = frame.iloc[index]
        if "question" in frame.columns:
            prompt = row["question"]
        elif "prompt" in frame.columns:
            value = row["prompt"]
            if isinstance(value, list) and value:
                first = value[0]
                prompt = (
                    first.get("content", first)
                    if isinstance(first, dict)
                    else first
                )
            else:
                prompt = value
        else:
            prompt = str(row)
        prompts.append(str(prompt))
    if not prompts:
        raise RuntimeError(f"no prompts loaded from {path}")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="/data/datasets/gsm8k/test.parquet",
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--draft-device", default="6")
    parser.add_argument("--target-device", default="7")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--log-file",
        default="/tmp/pearl_stage5_gsm8k_ac_benchmark_v1.log",
    )
    parser.add_argument(
        "--coordinator",
        default="pearl_stage5_coordinator.py",
        help="Stage-5 coordinator script in the same directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0 or args.max_tokens <= 0 or args.gamma <= 0:
        raise ValueError("num-samples, max-tokens, and gamma must be positive")
    if args.max_num_seqs < 1 or args.batch_size < 1:
        raise ValueError("max-num-seqs and batch-size must be positive")
    if args.batch_size > args.max_num_seqs:
        raise ValueError("batch-size cannot exceed max-num-seqs")

    repo = Path(__file__).resolve().parent
    coordinator = repo / args.coordinator
    if not coordinator.is_file():
        raise FileNotFoundError(coordinator)
    prompts = load_prompts(Path(args.data_path), args.num_samples)
    log_path = Path(args.log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(coordinator),
        "--draft-model",
        "/data/shared-models/Qwen3-0.6B",
        "--target-model",
        "/data/shared-models/Qwen3-8B",
        "--draft-device",
        str(args.draft_device),
        "--target-device",
        str(args.target_device),
        "--max-model-len",
        str(args.max_model_len),
        "--gamma",
        str(args.gamma),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--batch-size",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
    ]
    for prompt in prompts:
        command.extend(["--prompt", prompt])

    env = os.environ.copy()
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    env.setdefault("PEARL_STAGE5_TARGET_ASYNC_SCHEDULING", "0")
    env["PEARL_STAGE5_PERSISTENT_REQUEUE"] = "1"
    env["PEARL_STAGE5_PERSISTENT_REQUEUE_DROP_PARTIAL_BLOCK"] = "1"
    env["PEARL_ACCEPTANCE_DEBUG"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print("===== PEARL GSM8K AC BENCHMARK =====", flush=True)
    print(f"data_path: {args.data_path}", flush=True)
    print(f"num_samples: {len(prompts)}", flush=True)
    print(f"max_tokens: {args.max_tokens}", flush=True)
    print(f"gamma: {args.gamma}", flush=True)
    print(f"max_num_seqs: {args.max_num_seqs}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"log_file: {log_path}", flush=True)
    print("===== CHILD PROCESS OUTPUT IS FILTERED; FULL OUTPUT IS IN LOG =====", flush=True)

    last_draft = None
    last_accepted = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if (
                "PEARL_ACCEPTANCE_DEBUG" in line
                or line.startswith("[stage5] prompt ")
                or line.startswith("[stage5] launch ")
                or "status': 'ready'" in line
                or "Traceback" in line
                or "ERROR" in line
            ):
                print(line, end="", flush=True)
            match = AC_RE.search(line)
            if match:
                last_draft = int(match.group(1))
                last_accepted = int(match.group(2))
        return_code = process.wait()

    print("\n===== PEARL AC SUMMARY =====")
    if last_draft is None or last_accepted is None:
        print("AC metrics not found; inspect:", log_path)
        return 1 if return_code == 0 else return_code

    ac_rate = last_accepted / max(last_draft, 1)
    print(f"draft tokens:    {last_draft}")
    print(f"accepted tokens: {last_accepted}")
    print(f"average AC rate: {ac_rate * 100:.2f}%")
    print(f"child exit code: {return_code}")
    print(f"full log:        {log_path}")

    # The current worker has a known shutdown-only traceback after it has
    # returned the result.  The subprocess exit code is the authoritative
    # status for this benchmark.
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
