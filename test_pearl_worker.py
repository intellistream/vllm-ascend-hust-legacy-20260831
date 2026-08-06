import argparse
import os
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print(f"[{args.role}] visible device:",
          os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    print(f"[{args.role}] model:", args.model)

    start = time.time()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
    )

    print(f"[{args.role}] model loaded: {time.time() - start:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
    )

    outputs = llm.generate(
        ["请回答：1+1等于几？"],
        sampling_params,
        use_tqdm=False,
    )

    print(f"[{args.role}] output:")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()