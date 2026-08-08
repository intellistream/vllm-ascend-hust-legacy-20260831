# SPDX-License-Identifier: Apache-2.0
"""Run the native Ascend nano-PEARL engine with its upstream-compatible API."""

from __future__ import annotations

import argparse

from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--mode", choices=("pearl", "target-ar", "bench"), default="pearl")
    parser.add_argument("--num-pearl-steps", type=int, default=100)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("prompt", nargs="+", help="One or more prompts to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        max_num_batched_tokens=max(args.max_model_len, args.max_model_len * args.max_num_seqs),
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enforce_eager=args.enforce_eager,
        gamma=args.gamma,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    with PEARLEngine(config) as engine:
        for prompt in args.prompt:
            engine.add_request(prompt, sampling_params)
        if args.mode == "pearl":
            outputs = engine.generate()
        elif args.mode == "target-ar":
            outputs = engine.AR_generate()
        else:
            outputs = engine.bench_generate(args.num_pearl_steps)
    print(outputs)


if __name__ == "__main__":
    main()
