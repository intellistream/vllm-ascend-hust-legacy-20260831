#!/usr/bin/env python3
"""Target-only baseline for comparing Stage-5 output correctness.

Run this as a real Python file.  vLLM's spawn multiprocessing cannot start
reliably when the main module is provided through ``python -`` / stdin.
"""

from __future__ import annotations

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL = "/data/shared-models/Qwen3-8B"
PROMPTS = (
    "The capital of France is",
    "2+2=",
    "The largest planet in the solar system is",
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    llm = LLM(
        model=MODEL,
        max_model_len=1024,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
    )

    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params,
            use_tqdm=False,
        )
        result = outputs[0].outputs[0]
        print(f"\nPROMPT: {prompt}", flush=True)
        print(f"TEXT: {result.text}", flush=True)
        print(f"TOKEN_IDS: {list(result.token_ids)}", flush=True)


if __name__ == "__main__":
    main()
