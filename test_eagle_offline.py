import os
import time
import pandas as pd

from vllm import LLM, SamplingParams


DATA_PATH = "/data/datasets/gsm8k/test.parquet"

# Target 模型
MODEL = "/data/shared-models/Qwen2.5-14B-Instruct"

# EAGLE draft 模型
DRAFT_MODEL = "/data/shared-models/Eagle-Qwen2.5-14B-Instruct"

NUM_SAMPLES = 200
MAX_TOKENS = 512

EAGLE_NUM_SPECULATIVE_TOKENS = int(
    os.environ.get("EAGLE_NUM_SPECULATIVE_TOKENS", "2")
)
EAGLE_DRAFT_ENFORCE_EAGER = (
    os.environ.get("EAGLE_DRAFT_ENFORCE_EAGER", "0") == "1"
)
EAGLE_DISABLE_PADDED_DRAFTER_BATCH = (
    os.environ.get("EAGLE_DISABLE_PADDED_DRAFTER_BATCH", "0") == "1"
)
EAGLE_TUNED_CUDAGRAPH = (
    os.environ.get("EAGLE_TUNED_CUDAGRAPH", "1") == "1"
)
EAGLE_LOG_STATS = os.environ.get("EAGLE_LOG_STATS", "0") == "1"
EAGLE_CUDAGRAPH_MODE = os.environ.get("EAGLE_CUDAGRAPH_MODE", "FULL")
EAGLE_MAX_NUM_BATCHED_TOKENS = int(
    os.environ.get("EAGLE_MAX_NUM_BATCHED_TOKENS", "8192")
)


def eagle_cudagraph_capture_sizes(num_speculative_tokens: int):
    query_len = num_speculative_tokens + 1
    request_capture_sizes = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
    ]
    return [query_len * batch_size for batch_size in request_capture_sizes]


def load_prompts():
    df = pd.read_parquet(DATA_PATH)

    prompts = []
    for i in range(min(NUM_SAMPLES, len(df))):
        row = df.iloc[i]

        if "question" in df.columns:
            prompt = row["question"]
        elif "prompt" in df.columns:
            p = row["prompt"]
            if isinstance(p, list):
                prompt = p[0]["content"]
            else:
                prompt = str(p)
        else:
            prompt = str(row)

        prompts.append(prompt)

    return prompts


def run_test(use_eagle: bool):
    prompts = load_prompts()

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    if use_eagle:
        print("\n===== LOAD EAGLE =====")
        eagle_kwargs = {
            "model": MODEL,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "disable_log_stats": not EAGLE_LOG_STATS,
            "max_num_seqs": 128,
            "max_num_batched_tokens": EAGLE_MAX_NUM_BATCHED_TOKENS,
            "speculative_config": {
                "method": "eagle",
                "model": DRAFT_MODEL,
                "draft_tensor_parallel_size": 1,
                "num_speculative_tokens": EAGLE_NUM_SPECULATIVE_TOKENS,
                "enforce_eager": EAGLE_DRAFT_ENFORCE_EAGER,
                "disable_padded_drafter_batch": (
                    EAGLE_DISABLE_PADDED_DRAFTER_BATCH
                ),
            },
        }
        if EAGLE_TUNED_CUDAGRAPH or EAGLE_CUDAGRAPH_MODE:
            compilation_config = {}
            if EAGLE_TUNED_CUDAGRAPH:
                compilation_config["cudagraph_capture_sizes"] = (
                    eagle_cudagraph_capture_sizes(
                        EAGLE_NUM_SPECULATIVE_TOKENS
                    )
                )
            if EAGLE_CUDAGRAPH_MODE:
                compilation_config["cudagraph_mode"] = EAGLE_CUDAGRAPH_MODE
            eagle_kwargs["compilation_config"] = compilation_config

        print(
            "EAGLE config:",
            f"gamma={EAGLE_NUM_SPECULATIVE_TOKENS}",
            f"draft_eager={EAGLE_DRAFT_ENFORCE_EAGER}",
            f"disable_padded_drafter_batch="
            f"{EAGLE_DISABLE_PADDED_DRAFTER_BATCH}",
            f"tuned_cudagraph={EAGLE_TUNED_CUDAGRAPH}",
            f"cudagraph_mode={EAGLE_CUDAGRAPH_MODE or 'default'}",
            f"max_num_batched_tokens={EAGLE_MAX_NUM_BATCHED_TOKENS}",
            f"log_stats={EAGLE_LOG_STATS}",
        )
        llm = LLM(**eagle_kwargs)
    else:
        print("\n===== LOAD BASELINE =====")
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=1,
            trust_remote_code=True,
            disable_log_stats=True,
            max_num_seqs=128,
        )

    print(f"\n===== RUN {len(prompts)} PROMPTS =====")

    st = time.time()
    outputs = llm.generate(prompts, sampling_params)
    ed = time.time()

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    elapsed = ed - st

    print("\n===== RESULT =====")
    print("mode:", "EAGLE" if use_eagle else "BASELINE")
    print("num_samples:", len(prompts))
    print("max_tokens:", MAX_TOKENS)
    print("time:", round(elapsed, 4), "s")
    print("output_tokens:", total_output_tokens)
    print("tok/s:", round(total_output_tokens / elapsed, 4))

    if use_eagle and EAGLE_LOG_STATS:
        print("\n===== SPECULATIVE METRICS =====")
        for metric in llm.llm_engine.get_metrics():
            metric_text = str(metric)
            if any(
                name in metric_text.lower()
                for name in ("spec", "draft", "accept")
            ):
                print(metric_text)

    print("\n===== SAMPLE OUTPUT =====")
    print(outputs[0].outputs[0].text[:500])

    del llm

    return total_output_tokens / elapsed


if __name__ == "__main__":
    eagle_tps = run_test(use_eagle=True)
    baseline_tps = run_test(use_eagle=False)

    print("\n===== FINAL SUMMARY =====")

    print(f"baseline tok/s: {baseline_tps:.4f}")
    print(f"eagle tok/s:    {eagle_tps:.4f}")

    speedup = eagle_tps / baseline_tps

    print(f"speedup:        {speedup:.4f}x")

    if speedup < 1:
        print(f"slowdown:       {1/speedup:.4f}x")
