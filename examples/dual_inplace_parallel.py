"""Demonstrates the Ascend split-batch (dual-stream) decode path.

Passes ``additional_config={"split_batch_config": ...}`` to run the
inplace-parallel (two NPU streams) decode path on an 80-request batch.
See docs/source/user_guide/feature_guide/split_batch.md for the full
configuration reference and admission conditions.
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "7")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Optional JSONL diagnostics for the split-batch path; disabled by default.
if os.environ.get("SPLIT_BATCH_DEBUG", "0") == "1":
    os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
    os.environ.setdefault(
        "VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE",
        "/tmp/vllm_ascend_inplace_parallel.jsonl",
    )

from vllm import LLM, SamplingParams


def main():
    base_topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "ancient history",
        "modern art",
        "human psychology",
        "economics",
        "philosophy",
    ]
    batch_size = 80
    raw_prompts = [
        f"Explain {base_topics[i % len(base_topics)]} (variant {i})"
        for i in range(batch_size)
    ]
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        additional_config={
            "split_batch_config": {
                "enabled": True,
                "mode": "inplace_parallel",
                "num_splits": 2,
                "enable_parallel_streams": True,
                "enable_inplace_lazy_capture": True,
                "inplace_split_planner_policy": "largest_lower",
                "inplace_offset_match_policy": "exact",
                "inplace_parallel_replay_policy": "full_graph_parallel",
                "inplace_offset_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
                "parallel_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            },
        },
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
        },
        gpu_memory_utilization=0.8,
    )

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )
        for text in raw_prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    if os.environ.get("VLLM_ASCEND_SPLIT_INPLACE_DEBUG") == "1":
        print(
            ">>> Diag log: "
            f"{os.environ.get('VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE', '(default)')} <<<"
        )

if __name__ == "__main__":
    main()
