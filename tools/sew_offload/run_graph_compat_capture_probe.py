# SPDX-License-Identifier: Apache-2.0
"""Probe: does MoE offload capture under ACLGraph once the D2H decision is hoisted?

Milestone 1 (capture-pass) verification for the graph-compatible MoE offload
primitives. This harness runs ONE configuration per process (vLLM V1 EngineCore
is a subprocess; env must be set before import). The decisive experiment is the
pair:

  GRAPH_COMPATIBLE=0 + ACLGraph  -> expect capture FAILURE (107027/107030,
                                    synchronized memcpy forbidden in capture)
  GRAPH_COMPATIBLE=1 + ACLGraph  -> expect capture SUCCESS (capture-safe path
                                    bypasses torch.unique(...).cpu())

Offload budget is taken from VLLM_ASCEND_MOE_OFFLOAD_GB (set before launch, same
as --ascend-moe-offload-gb). The script prints machine-greppable markers:
  LOAD_OK / GENERATE_OK / OUTPUT_TOKENS / CASE_FAILED.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--model", default="/data/shared-models/Qwen3-30B-A3B")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="if >0, request top-k logprobs per position and print LOGPROBS lines",
    )
    parser.add_argument("--prompt", default="Briefly explain mixture-of-experts models.")
    parser.add_argument(
        "--latency",
        action="store_true",
        help="after the correctness generate, measure TTFT/TPOT/throughput via the "
        "differential method (T(1 token) ~= TTFT; TPOT = (T(N)-T(1))/(N-1))",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=5,
        help="number of timed repeats per measurement point (median is reported)",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="default False = ACLGraph capture enabled (the case under test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import torch/vllm only after env is in place (set by the caller).
    import torch
    import torch_npu  # noqa: F401

    # CRITICAL: offline LLM() cannot trigger the autoconfig monkeypatch on its
    # own. AscendPlatform.pre_register_and_update() (which calls adapt_patch ->
    # imports patch.platform, installing the EngineArgs.create_engine_config
    # autoconfig wrapper) runs as the FIRST line of the *unpatched*
    # create_engine_config (arg_utils.py:1634), so it only arms "next call" and
    # offline has no next call. The CLI/api_server path arms it during arg
    # parsing (arg_utils.py:2446) instead. We import the patch module up front so
    # the wrapper is installed BEFORE LLM() builds the engine config -> autoconfig
    # fires -> vLLM PrefetchOffloader is wired (the device-residency mechanism the
    # validated --ascend-moe-offload-gb command relies on). Without this, only the
    # SEW fixed-slot path activates, w13/w2 stay CPU-resident, and high-fanout
    # prefill (FULL_WEIGHT_PATH) crashes with "weight is on cpu".
    import vllm_ascend.patch.platform  # noqa: F401

    from vllm import LLM, SamplingParams

    print(f"CASE {args.case_name}", flush=True)
    print(f"model {args.model}", flush=True)
    print(f"ASCEND_RT_VISIBLE_DEVICES {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    print(f"VLLM_ASCEND_MOE_OFFLOAD_GB {os.environ.get('VLLM_ASCEND_MOE_OFFLOAD_GB')}", flush=True)
    print(
        "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE "
        f"{os.environ.get('VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE')}",
        flush=True,
    )
    print(f"enforce_eager {args.enforce_eager}", flush=True)
    print(
        f"npu_available {torch.npu.is_available()} device_count {torch.npu.device_count()}",
        flush=True,
    )

    # Self-check: confirm the autoconfig monkeypatch is armed BEFORE LLM() builds
    # the engine config. If this is False the run is invalid (PrefetchOffloader
    # would not be wired and the result would repeat the false-negative crash).
    from vllm.engine.arg_utils import EngineArgs

    autoconfig_armed = bool(
        getattr(EngineArgs.create_engine_config, "_ascend_moe_offload_autoconfig_patch", False)
    )
    print(f"AUTOCONFIG_PATCH_ARMED {autoconfig_armed}", flush=True)
    if not autoconfig_armed:
        print(
            "CASE_FAILED RuntimeError: autoconfig monkeypatch not armed; "
            "PrefetchOffloader would not be wired",
            flush=True,
        )
        raise RuntimeError("autoconfig monkeypatch not armed before LLM()")

    t0 = time.time()
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_mb * 1024 * 1024,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_expert_parallel=False,
            seed=0,
        )
        # LOAD_OK here means init + capture_model (when ACLGraph) BOTH passed:
        # capture happens during engine init for the V1 worker.
        print(f"LOAD_OK seconds={time.time() - t0:.3f}", flush=True)

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            logprobs=args.logprobs if args.logprobs > 0 else None,
        )
        gen_t0 = time.time()
        outputs = llm.generate([args.prompt], params)
        print(f"GENERATE_OK seconds={time.time() - gen_t0:.3f}", flush=True)
        for output in outputs:
            token_ids = list(output.outputs[0].token_ids)
            print(f"OUTPUT_TOKENS {token_ids}", flush=True)
            print("OUTPUT_TEXT " + output.outputs[0].text.replace("\n", "\\n"), flush=True)
            # Per-position top-k logprobs: lets us measure the top-1/top-2 gap at
            # the first divergence step (numerical near-tie vs a decisive logic flip).
            if args.logprobs > 0 and output.outputs[0].logprobs is not None:
                for pos, lp in enumerate(output.outputs[0].logprobs):
                    ranked = sorted(lp.items(), key=lambda kv: kv[1].rank)
                    items = [
                        f"{tid}:{v.logprob:.5f}(r{v.rank})" for tid, v in ranked
                    ]
                    print(f"LOGPROBS pos={pos} chosen={token_ids[pos]} " + " ".join(items), flush=True)

        # P0 latency: differential TTFT/TPOT. Offline V1 generate() is blocking and
        # does not stream, and RequestOutput.metrics is often unpopulated under V1,
        # so we derive the split from two wall-clock points instead of internal
        # timers: a max_tokens=1 generate measures prefill + 1 decode (~TTFT), and a
        # max_tokens=N generate measures the whole sequence; TPOT is their per-token
        # difference. A warmup generate first absorbs lazy graph-replay / allocator
        # costs so they do not contaminate the first timed point. Same prompt +
        # temperature=0 => deterministic, so repeats differ only by runtime jitter.
        if args.latency:
            import statistics as _stats

            def _timed(max_tokens: int) -> float:
                p = SamplingParams(max_tokens=max_tokens, temperature=0.0)
                t = time.perf_counter()
                llm.generate([args.prompt], p)
                return time.perf_counter() - t

            _timed(args.max_tokens)  # warmup (discarded)
            n = args.max_tokens
            reps = max(1, args.latency_repeats)
            t1 = _stats.median(_timed(1) for _ in range(reps))
            tn = _stats.median(_timed(n) for _ in range(reps))
            ttft_ms = t1 * 1000.0
            tpot_ms = ((tn - t1) / (n - 1)) * 1000.0 if n > 1 else float("nan")
            decode_tps = (n - 1) / (tn - t1) if (n > 1 and tn > t1) else float("nan")
            print(
                f"LATENCY case={args.case_name} n_tokens={n} reps={reps} "
                f"TTFT_MS={ttft_ms:.2f} TPOT_MS={tpot_ms:.3f} "
                f"DECODE_TPS={decode_tps:.2f} T1_S={t1:.4f} TN_S={tn:.4f}",
                flush=True,
            )
    except BaseException as exc:  # noqa: BLE001 - we want the full failure surfaced
        print(f"CASE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
