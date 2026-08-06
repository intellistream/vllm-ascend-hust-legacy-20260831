#!/usr/bin/env python3
"""Target-only GSM8K baseline matching the Stage-5 batch-2 conditions."""

from __future__ import annotations

import pandas as pd
from vllm import LLM, SamplingParams


DATA_PATH = "/data/datasets/gsm8k/test.parquet"
MODEL = "/data/shared-models/Qwen3-8B"
NUM_SAMPLES = 2
MAX_TOKENS = 128
MAX_NUM_SEQS = 2


def load_prompts() -> list[str]:
    frame = pd.read_parquet(DATA_PATH)
    prompts: list[str] = []
    for index in range(min(NUM_SAMPLES, len(frame))):
        row = frame.iloc[index]
        if "question" in frame.columns:
            value = row["question"]
        elif "prompt" in frame.columns:
            value = row["prompt"]
            if isinstance(value, list) and value:
                first = value[0]
                value = (
                    first.get("content", first)
                    if isinstance(first, dict)
                    else first
                )
        else:
            value = str(row)
        prompts.append(str(value))
    if len(prompts) != NUM_SAMPLES:
        raise RuntimeError(f"expected {NUM_SAMPLES} prompts, got {len(prompts)}")
    return prompts


def main() -> None:
    prompts = load_prompts()
    print(
        f"baseline target={MODEL} batch={len(prompts)} "
        f"max_num_seqs={MAX_NUM_SEQS} max_tokens={MAX_TOKENS} "
        "async_scheduling=False",
        flush=True,
    )
    for index, prompt in enumerate(prompts, start=1):
        print(f"PROMPT {index}: {prompt!r}", flush=True)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        async_scheduling=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"output count mismatch: expected {len(prompts)}, "
            f"got {len(outputs)}"
        )
    for index, output in enumerate(outputs, start=1):
        result = output.outputs[0]
        print(f"OUTPUT {index}: {result.text!r}", flush=True)
        print(
            f"OUTPUT {index} TOKENS: {len(result.token_ids)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
