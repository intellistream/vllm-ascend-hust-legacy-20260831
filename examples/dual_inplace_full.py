import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "7"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"] = "/tmp/vllm_ascend_inplace_parallel.jsonl"

from vllm import LLM, SamplingParams

def main():
    # 1. 60个短请求：max_tokens=500 确保它们长时间停留在 Decode 阶段
    short_prompts = ["Hello, how are you?"] * 60
    
    # 2. 1个超长请求：长度约 1000+ tokens
    long_prompts = ["Explain the universe in extreme detail. " + ("word " * 1000)]
    
    raw_prompts = short_prompts + long_prompts
    
    sampling_params = SamplingParams(max_tokens=500, temperature=0.0)
    llm = LLM(
        model="/data/shared-models/Qwen2.5-14B-Instruct",
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
            },
        },
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        },
        gpu_memory_utilization=0.6,
        # 修正点：将 max_num_seqs 设为 64 (满足 <= max_num_batched_tokens 的要求)
        max_num_seqs=64,
        # 保持 max_num_batched_tokens=120，强制 60 Decode + 60 Prefill = 120 混合批次
        max_num_batched_tokens=120,
        enable_chunked_prefill=True,
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
    for output in outputs[:5]:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt[:50]!r}, Generated text: {generated_text!r}")
    print(f">>> Diag log: {os.environ['VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE']} <<<")

if __name__ == "__main__":
    main()