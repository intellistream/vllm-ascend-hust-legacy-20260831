#!/usr/bin/env python3
# isort: skip_file
"""Dual-stream link smoke: verify the inplace_parallel split-batch path can
run end-to-end (offline generation + OpenAI serve) without crashing.

This is a *link* check, not a correctness comparison: it asserts the run
succeeds and produces non-empty output. Use
``examples/test_split_batch_correctness_npu.py`` for split on/off output
consistency comparison.

Exit code: 0 on PASS, 1 on FAIL — suitable for automatic self-check after
code changes.

Usage (run from the repo root):
  python tests/run_dual_stream_link_smoke.py                # offline + serve
  python tests/run_dual_stream_link_smoke.py --mode offline
  python tests/run_dual_stream_link_smoke.py --mode serve
  python tests/run_dual_stream_link_smoke.py --model <path> --no-split
  python tests/run_dual_stream_link_smoke.py --eager        # skip cudagraph
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from a subdirectory while still importing repo-local helpers.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("VLLM_TARGET_DEVICE", "ascend")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DEFAULT = "/data/shared-models/Qwen2.5-0.5B-Instruct"
PROMPTS = ["你好，请介绍一下你自己。", "What is the capital of France?"]


def build_additional_config(split_enabled: bool, force_split: bool = False,
                            split_mode: str = "inplace_parallel",
                            unified_capture_sizes: list[int] | None = None) -> dict:
    cfg = {
        "split_batch_config": {
            "enabled": split_enabled,
            "mode": split_mode,
            "num_splits": 2,
            "enable_parallel_streams": True,
            "min_batch_size_for_split": 4,
        }
    }
    if split_mode == "inplace_parallel":
        cfg["split_batch_config"]["enable_inplace_lazy_capture"] = True
        cfg["split_batch_config"][
            "inplace_parallel_replay_policy"] = "full_graph_parallel"
    if force_split:
        cfg["split_batch_config"]["force_split"] = True
    # P11 unified-row-graph (requires the env switch too):
    # ``VLLM_ASCEND_DUAL_UNIFIED_GEMM=1`` is exported by main() when this
    # option is provided.
    if unified_capture_sizes:
        cfg["split_batch_config"]["unified_capture_sizes"] = list(
            unified_capture_sizes)
    return cfg


def build_compilation_config(cudagraph: bool) -> dict | None:
    if not cudagraph:
        return None
    return {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }


def run_offline(model: str, prompts: list[str],
                split_enabled: bool, cudagraph: bool,
                force_split: bool = False,
                split_mode: str = "inplace_parallel",
                unified_capture_sizes: list[int] | None = None) -> None:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        enforce_eager=not cudagraph,
        additional_config=build_additional_config(
            split_enabled, force_split, split_mode, unified_capture_sizes),
    )
    if cudagraph:
        kwargs["compilation_config"] = build_compilation_config(cudagraph)

    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(max_tokens=16, temperature=0),
        )
    finally:
        del llm

    for out in outputs:
        text = out.outputs[0].text
        if not text.strip():
            raise AssertionError(f"empty generation for prompt: {out.prompt!r}")
    print(f"[offline] PASS split_enabled={split_enabled} "
          f"cudagraph={cudagraph} outputs={len(outputs)}")


def run_serve(model: str, split_enabled: bool, cudagraph: bool,
              force_split: bool = False,
              split_mode: str = "inplace_parallel") -> None:
    import openai
    from vllm.utils.network_utils import get_open_port

    from tests.e2e.conftest import RemoteOpenAIServer

    compilation_config = build_compilation_config(cudagraph)
    port = get_open_port()
    server_args = [
        "--max_model_len", "1024",
        "--gpu_memory_utilization", "0.8",
        "--port", str(port),
        "--additional-config",
        json.dumps(build_additional_config(
            split_enabled, force_split, split_mode)),
    ]
    if compilation_config is not None:
        server_args += ["--compilation-config", json.dumps(compilation_config)]

    with RemoteOpenAIServer(model, server_args, server_port=port,
                            auto_port=False, max_wait_seconds=600) as server:
        health = server.url_for("health")
        assert health.endswith("/health"), f"unexpected health url: {health}"

        client = openai.OpenAI(base_url=server.url_for("v1"), api_key="empty")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPTS[0]}],
            max_tokens=16,
            temperature=0,
        )
        text = resp.choices[0].message.content
        if not text or not text.strip():
            raise AssertionError("serve returned empty completion")
    print(f"[serve] PASS split_enabled={split_enabled} "
          f"cudagraph={cudagraph} reply={text.strip()[:32]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual-stream link smoke (offline + serve).")
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help=f"model path (default: {MODEL_DEFAULT})")
    parser.add_argument("--mode", choices=("offline", "serve", "both"),
                        default="both")
    parser.add_argument("--batch-size", type=int, default=len(PROMPTS),
                        help="number of offline prompts (must be >= "
                             "min_batch_size_for_split to trigger split)")
    parser.add_argument("--no-split", action="store_true",
                        help="run without split_batch_config (baseline)")
    parser.add_argument("--force-split", action="store_true",
                        help="force split even on exact graph size hit")
    parser.add_argument("--split-mode", choices=("inplace_parallel", "dual_pad"),
                        default="inplace_parallel",
                        help="dual-stream execution mode (default: "
                             "inplace_parallel; dual_pad requires --batch-size "
                             ">= 5 to actually split with default capture sizes)")
    parser.add_argument("--eager", action="store_true",
                        help="skip cudagraph (enforce eager)")
    parser.add_argument("--unified-capture-sizes", type=str, default=None,
                        help="P11 DUAL_UNIFIED_GEMM: comma-separated exact "
                             "sizes (e.g. '6'); exports VLLM_ASCEND_DUAL_"
                             "UNIFIED_GEMM=1 and injects unified_capture_"
                             "sizes into split_batch_config. Requires a "
                             "would-be dual_pad decision hitting one of the "
                             "declared totals.")
    args = parser.parse_args()

    unified_sizes: list[int] | None = None
    if args.unified_capture_sizes:
        os.environ["VLLM_ASCEND_DUAL_UNIFIED_GEMM"] = "1"
        unified_sizes = [
            int(s) for s in args.unified_capture_sizes.split(",") if s.strip()]
        print(f"[unified] VLLM_ASCEND_DUAL_UNIFIED_GEMM=1, "
              f"unified_capture_sizes={unified_sizes}")

    split_enabled = not args.no_split
    cudagraph = not args.eager
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.batch_size)]

    failures = []
    if args.mode in ("offline", "both"):
        try:
            run_offline(args.model, prompts, split_enabled, cudagraph,
                        force_split=args.force_split,
                        split_mode=args.split_mode,
                        unified_capture_sizes=unified_sizes)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(("offline", exc))
    if args.mode in ("serve", "both"):
        try:
            run_serve(args.model, split_enabled, cudagraph,
                      force_split=args.force_split,
                      split_mode=args.split_mode)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(("serve", exc))

    if failures:
        for mode, exc in failures:
            print(f"[{mode}] FAIL: {exc!r}", file=sys.stderr)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
