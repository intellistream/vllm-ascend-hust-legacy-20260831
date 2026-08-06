#!/usr/bin/env python3
"""Qwen3 chat-template Target-only GSM8K baseline."""

from __future__ import annotations

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DATA_PATH = "/data/datasets/gsm8k/test.parquet"
MODEL = "/data/shared-models/Qwen3-8B"
NUM_SAMPLES = 2
MAX_TOKENS = 128
MAX_NUM_SEQS = 2


def load_questions() -> list[str]:
    frame = pd.read_parquet(DATA_PATH)
    questions: list[str] = []
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
        questions.append(str(value))
    if len(questions) != NUM_SAMPLES:
        raise RuntimeError(
            f"expected {NUM_SAMPLES} questions, got {len(questions)}"
        )
    return questions


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )
    questions = load_questions()
    prompt_token_ids: list[list[int]] = []
    for index, question in enumerate(questions, start=1):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = [int(token_id) for token_id in ids]
        prompt_token_ids.append(ids)
        print(f"PROMPT {index}: {question!r}", flush=True)
        print(f"PROMPT {index} TOKENS: {len(ids)}", flush=True)

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
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params,
        use_tqdm=False,
    )
    if len(outputs) != len(questions):
        raise RuntimeError(
            f"output count mismatch: expected {len(questions)}, "
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
