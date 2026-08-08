# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native nano-PEARL target-only path with one engine load."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    return parser


def _load_prompts(prompt: str | None, gsm8k: str | None, max_samples: int) -> list[str]:
    if prompt is not None:
        return [prompt] * max_samples
    import pyarrow.parquet as pq

    rows = pq.read_table(gsm8k, columns=["question"]).to_pylist()
    if len(rows) < max_samples:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but batch size {max_samples} was requested.")
    return [str(row["question"]) for row in rows[:max_samples]]


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every target-only batch size must be positive.")
    max_batch_size = max(args.batch_sizes)

    from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams

    prompts = _load_prompts(args.prompt, args.gsm8k, max_batch_size)
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        max_num_batched_tokens=args.max_model_len * max_batch_size,
        max_num_seqs=max_batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        gamma=4,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    results = []
    with PEARLEngine(config) as engine:
        first_formatted_prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[0]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        first_prompt_token_ids = list(engine.tokenizer.encode(first_formatted_prompt))
        for batch_size in args.batch_sizes:
            for prompt in prompts[:batch_size]:
                engine.add_request(prompt, sampling_params)
            engine.AR_generate()

            for prompt in prompts[:batch_size]:
                engine.add_request(prompt, sampling_params)
            started = time.perf_counter()
            _, num_tokens, _, inference_elapsed = engine.AR_generate()
            e2e_elapsed = time.perf_counter() - started
            output_tokens = sum(num_tokens)
            first_metrics = engine.last_metrics[0]
            results.append(
                {
                    "batch_size": batch_size,
                    "output_tokens": output_tokens,
                    "inference_elapsed_seconds": inference_elapsed,
                    "e2e_elapsed_seconds": e2e_elapsed,
                    "inference_throughput_tokens_per_second": output_tokens / inference_elapsed,
                    "e2e_throughput_tokens_per_second": output_tokens / e2e_elapsed,
                    "prefill_elapsed_seconds": first_metrics["prefill_elapsed_seconds"],
                    "decode_elapsed_seconds": first_metrics["decode_elapsed_seconds"],
                    "first_output_token_ids": first_metrics["completion_token_ids"],
                }
            )

    print(
        json.dumps(
            {
                "backend": "nano-pearl-native-target-only",
                "draft_model": args.draft_model,
                "target_model": args.target_model,
                "draft_tensor_parallel_size": args.draft_tp_size,
                "target_tensor_parallel_size": args.target_tp_size,
                "max_tokens": args.max_tokens,
                "first_prompt_token_ids": first_prompt_token_ids,
                "results": results,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
