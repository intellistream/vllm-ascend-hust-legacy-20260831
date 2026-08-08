# SPDX-License-Identifier: Apache-2.0
"""Benchmark the production vLLM-Ascend target-only baseline."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--raw-prompts", action="store_true")
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
    if max(args.batch_sizes) > args.max_num_seqs:
        raise ValueError("The largest batch size exceeds max_num_seqs.")

    from vllm import LLM, SamplingParams, TokensPrompt

    prompts = _load_prompts(args.prompt, args.gsm8k, max(args.batch_sizes))
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    if args.raw_prompts:
        prompt_inputs = prompts
        first_prompt_token_ids = list(llm.get_tokenizer().encode(prompts[0]))
    else:
        tokenizer = llm.get_tokenizer()
        prompt_token_ids = [
            list(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            )
            for prompt in prompts
        ]
        prompt_inputs = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
        first_prompt_token_ids = prompt_token_ids[0]
    llm.generate(prompt_inputs[:1], sampling_params, use_tqdm=False)

    results = []
    for batch_size in args.batch_sizes:
        started = time.perf_counter()
        outputs = llm.generate(prompt_inputs[:batch_size], sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        results.append(
            {
                "batch_size": batch_size,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                "output_throughput_tokens_per_second": output_tokens / elapsed,
                "first_output_token_ids": list(outputs[0].outputs[0].token_ids),
            }
        )
    print(
        json.dumps(
            {
                "backend": "vllm-ascend-target-only",
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_tokens": args.max_tokens,
                "first_prompt_token_ids": first_prompt_token_ids,
                "results": results,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
