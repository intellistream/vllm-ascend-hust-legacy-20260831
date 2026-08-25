#!/usr/bin/env python3
# isort: skip_file
"""Per-decode-step (steady-state) latency comparison: padding vs dual-stream split.

目标:
- 对同一个固定 decode batch, 测"单流 padding" vs "双流 split(exact)" 的
  **稳态每步前向 TPOT (time per output token)**.
- 正确做法: 逐 step 计时 (复刻 vllm.LLM._run_engine 骨架:
  `while engine.has_unfinished_requests(): engine.step()`),
  丢弃 prefill 步 + 首个 decode 步 (双流的 offset 图在此捕获/或作为 warmup),
  对剩余 replay 步求 median/mean —— 这才是稳态 TPOT.

背景 (为什么不能只测 total/max_tokens):
- dual-inplace 的 split1 精确尺寸 offset 图是 lazy capture, 在"形状变化"时才 re-capture
  (实测 bs=144: `inplace_capture_entry:2`, step_ids [1,7]; 跨 generate 复用)。
- `total/max_tokens` 会把 prefill + (偶尔)re-capture 摊进每步, 污染 pad/split 比值。
- 因此逐 step + 丢弃前导步, 给出干净稳态 TPOT。

与 correctness harness 区别: 只计时不做正确性断言; 复用其 LLM 构建与 split_batch_config 构造。

运行方式 (NPU, conda /root/miniconda3/envs/vllm-hust-dev):
  python examples/bench_split_batch_latency_npu.py \
    --model /data/shared-models/Qwen2.5-0.5B-Instruct \
    --batch-size 144 --max-tokens 32 --reps 2 \
    --capture-sizes 128,256 --enable-parallel-streams \
    --offset-match-policy exact --run split --output-dir <dir>

输出: <output-dir>/latency_bs<N>_<run>.json
"""

import importlib.util
import json
import os
import statistics
import sys
import time

# Matches harness behavior.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "3")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
import torch_npu  # noqa: E402
from vllm import SamplingParams  # noqa: E402

HARNESS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_split_batch_correctness_npu.py",
)

FIXED_PROMPT = "The capital of France is"


def _load_harness():
    spec = importlib.util.spec_from_file_location("split_harness", HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_int_list(raw):
    if raw is None:
        return None
    values = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _build_llm_args(h, *, model, capture_sizes, cudagraph_mode,
                    max_num_seqs, max_model_len, gpu_memory_utilization,
                    enforce_eager):
    args = {
        "model": model,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "compilation_config": {"cudagraph_mode": cudagraph_mode},
    }
    if capture_sizes is not None:
        h._apply_capture_sizes(args, capture_sizes)
    if max_num_seqs is not None:
        args["max_num_seqs"] = max_num_seqs
    return args


def _time_generate(llm, prompts, sampling):
    """Measure one llm.generate() start-to-end with device-wide sync (robust)."""
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sampling)
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    return time.perf_counter() - t0


def _run_decode_tpot(h, *, llm_args, prompts, additional_config,
                     warmup, reps, max_tokens):
    """Return per-decode-token wall ms via T(1) vs T(K) subtraction.

      T(1) = prefill + 1*decode      (max_tokens=1)
      T(K) = prefill + K*decode      (max_tokens=K)
      decode_ms = (median(TK) - median(T1)) / (K-1) * 1000

    Because the prompts are identical, prefill is identical in both runs (even if
    prefill is chunked into multiple engine steps), so subtraction removes prefill
    entirely -> pure per-token decode time. The offset split graph is captured by
    the warmup generate, so no recapture pollutes T(1)/T(K). Uses a single LLM
    instance so cudagraph/offset caches are shared.
    """
    from vllm import SamplingParams
    sK = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                        top_p=1.0, ignore_eos=True)
    s1 = SamplingParams(max_tokens=1, temperature=0.0,
                        top_p=1.0, ignore_eos=True)

    llm = h._build_llm_from_args(llm_args, additional_config=additional_config)
    # Warmup: untimed generate captures split offset graph + warms graphs.
    for _ in range(max(warmup, 0)):
        _time_generate(llm, prompts, sK)

    t1_samples = [_time_generate(llm, prompts, s1) for _ in range(reps)]
    tk_samples = [_time_generate(llm, prompts, sK) for _ in range(reps)]

    try:
        h._cleanup_llm(llm)
    except Exception:
        pass

    t1_med = statistics.median(t1_samples)
    tk_med = statistics.median(tk_samples)
    decode_ms = ((tk_med - t1_med) / (max_tokens - 1)) * 1000.0 if max_tokens > 1 else None
    return {
        "t1_samples": [round(x, 4) for x in t1_samples],
        "tk_samples": [round(x, 4) for x in tk_samples],
        "t1_median_s": round(t1_med, 4),
        "tk_median_s": round(tk_med, 4),
        "decode_per_token_ms": decode_ms,
    }


def parse_cudagraph_mode(raw):
    try:
        cfg = json.loads(str(raw))
        return cfg.get("cudagraph_mode", "FULL_DECODE_ONLY")
    except Exception:
        return str(raw)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/shared-models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed generate() runs before timed reps (warms graphs + captures offset)")
    p.add_argument("--capture-sizes", default="4,8")
    p.add_argument("--parallel-capture-sizes", default=None)
    p.add_argument("--compilation-config",
                   default='{"cudagraph_mode":"FULL_DECODE_ONLY"}')
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--split-mode", default="inplace_parallel")
    p.add_argument("--enable-parallel-streams", action="store_true")
    p.add_argument("--min-batch-size-for-split", type=int, default=4)
    p.add_argument("--force-split", action="store_true")
    p.add_argument("--offset-match-policy", default="exact",
                   choices=["exact", "bucket"])
    p.add_argument("--run", default="both", choices=["both", "pad", "split"])
    p.add_argument("--output-dir",
                   default="/workspace/现阶段情况核实/task-b-logs/latency")
    args = p.parse_args()

    capture_sizes = _parse_int_list(args.capture_sizes)
    parallel_capture_sizes = _parse_int_list(args.parallel_capture_sizes)
    h = _load_harness()

    max_num_seqs = args.max_num_seqs
    if max_num_seqs is None:
        req = int(args.batch_size)
        if capture_sizes:
            req = max(req, max(int(x) for x in capture_sizes))
        max_num_seqs = req

    llm_args = _build_llm_args(
        h, model=args.model, capture_sizes=capture_sizes,
        cudagraph_mode=parse_cudagraph_mode(args.compilation_config),
        max_num_seqs=max_num_seqs, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    prompts = h._generate_fixed_prompts(
        batch_size=args.batch_size, prompt=FIXED_PROMPT)

    def make_cfg(enabled: bool):
        cfg = h._build_split_additional_config(
            enabled=enabled,
            split_mode=args.split_mode,
            num_splits=2,
            enable_parallel_streams=args.enable_parallel_streams,
            min_batch_size_for_split=args.min_batch_size_for_split,
            parallel_capture_sizes=parallel_capture_sizes,
            force_split=args.force_split,
            inplace_split_planner_policy="largest_lower",
            inplace_parallel_replay_policy="full_graph_parallel",
            macro_graph_config=None,
            pa_shape_list=None,
        )
        cfg["split_batch_config"]["inplace_offset_match_policy"] = (
            args.offset_match_policy)
        return cfg

    results = {}
    runs = []
    if args.run in ("both", "pad"):
        runs.append("pad")
    if args.run in ("both", "split"):
        runs.append("split")

    os.makedirs(args.output_dir, exist_ok=True)

    for mode in runs:
        enabled = (mode == "split")
        cfg = make_cfg(enabled)
        res = _run_decode_tpot(
            h, llm_args=llm_args, prompts=prompts,
            additional_config=cfg, warmup=args.warmup, reps=args.reps,
            max_tokens=args.max_tokens,
        )
        results[mode] = {
            "reps": args.reps,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            **res,
        }
        print(f"[{mode}] decode_per_token={res['decode_per_token_ms']:.3f}ms "
              f"T1={res['t1_median_s']}s TK={res['tk_median_s']}s", flush=True)

    pad_med = results["pad"]["decode_per_token_ms"] if "pad" in results else None
    split_med = results["split"]["decode_per_token_ms"] if "split" in results else None
    if pad_med is not None and split_med is not None:
        results["cross"] = {
            "pad_decode_ms": pad_med,
            "split_decode_ms": split_med,
            "split_is_faster": split_med < pad_med,
            "speedup": round(pad_med / split_med, 4) if split_med else None,
        }
        print(f"[cross] pad={pad_med:.3f} split={split_med:.3f} "
              f"speedup={results['cross']['speedup']}", flush=True)

    out = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "capture_sizes": capture_sizes,
        "offset_match_policy": args.offset_match_policy,
        "split_mode": args.split_mode,
        "enable_parallel_streams": args.enable_parallel_streams,
        "warmup": args.warmup,
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "split_batch_config": make_cfg(True)["split_batch_config"],
        "results": results,
    }
    out_path = os.path.join(
        args.output_dir, f"latency_bs{args.batch_size}_{'_'.join(runs)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
