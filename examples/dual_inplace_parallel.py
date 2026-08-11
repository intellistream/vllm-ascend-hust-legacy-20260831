import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "7"
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"] = "/tmp/vllm_ascend_inplace_parallel.jsonl"

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
    print(f">>> Diag log: {os.environ['VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE']} <<<")

if __name__ == "__main__":
    main()
