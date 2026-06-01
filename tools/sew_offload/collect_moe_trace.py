# SPDX-License-Identifier: Apache-2.0
"""Collect routed MoE expert traces for SEW-Offload.

This script enables MVP-B trace-only mode, runs a small vLLM workload, and
exports the in-memory MoE trace collector as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import yaml
from vllm import LLM, SamplingParams

from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime, reset_moe_offload_runtime


DEFAULT_CONFIG = "docs/sew-offload/benchmark_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--prepare-smoke-manifest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-requests-per-bucket", type=int, default=1)
    parser.add_argument("--buckets", default="short_chat")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bucket_target_prompt_tokens(bucket: dict[str, Any], index: int) -> int:
    prompt_tokens = bucket["prompt_tokens"]
    if prompt_tokens == "mixed":
        mixed_targets = [192, 768, 3072, 384]
        return mixed_targets[index % len(mixed_targets)]
    low, high = prompt_tokens
    if high <= low:
        return int(low)
    ratio = (index % 5) / 4
    return int(round(low + (high - low) * ratio))


def prepare_synthetic_smoke_manifest(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    requests_per_bucket: int,
    buckets: set[str] | None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    seed_text = (
        "This is a SEW-Offload trace collection request for mixture of experts "
        "inference on Ascend NPU."
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        for bucket in config["workload_buckets"]:
            name = bucket["name"]
            if buckets is not None and name not in buckets:
                continue
            count = min(requests_per_bucket, int(bucket["num_requests"]))
            for i in range(count):
                target_tokens = _bucket_target_prompt_tokens(bucket, i)
                repeat_count = max(1, target_tokens // 16)
                prompt = " ".join([seed_text] * repeat_count)
                record = {
                    "request_id": f"{name}_{i:04d}",
                    "bucket": name,
                    "prompt": prompt,
                    "target_prompt_tokens": target_tokens,
                    "max_output_tokens": int(bucket["output_tokens"]),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": int(config["dataset"]["seed"]),
                    "dataset": "synthetic_smoke",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(
    manifest_path: Path,
    buckets: set[str] | None,
    max_requests: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if buckets is not None and record.get("bucket") not in buckets:
                continue
            requests.append(record)
            if max_requests and len(requests) >= max_requests:
                break
    if not requests:
        raise ValueError(f"no requests selected from manifest: {manifest_path}")
    return requests


def collect_trace(args: argparse.Namespace, config: dict[str, Any], requests: list[dict[str, Any]]) -> int:
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] = "1"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] = "1"
    reset_moe_offload_runtime()

    model_path = args.model or config["model"]["path"]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(config["model"]["tensor_parallel_size"]),
        trust_remote_code=False,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_mb * 1024 * 1024,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_expert_parallel=False,
        seed=int(config["dataset"]["seed"]),
        disable_log_stats=False,
    )
    sampling_params = [
        SamplingParams(
            max_tokens=int(req["max_output_tokens"]),
            temperature=float(req.get("temperature", 0.0)),
            top_p=float(req.get("top_p", 1.0)),
            ignore_eos=args.ignore_eos,
        )
        for req in requests
    ]
    llm.generate([req["prompt"] for req in requests], sampling_params, use_tqdm=False)
    return get_moe_offload_runtime().export_trace(args.output)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selected_buckets = csv_set(args.buckets) or None
    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prepare_smoke_manifest:
        prepare_synthetic_smoke_manifest(
            config=config,
            manifest_path=manifest_path,
            requests_per_bucket=args.smoke_requests_per_bucket,
            buckets=selected_buckets,
        )
        if args.prepare_only:
            print(f"PREPARE_OK manifest={manifest_path}", flush=True)
            return

    try:
        requests = load_manifest(manifest_path, selected_buckets, args.max_requests)
        num_records = collect_trace(args, config, requests)
        summary = {
            "status": "ok",
            "output": str(output_path),
            "manifest": str(manifest_path),
            "num_requests": len(requests),
            "num_trace_records": num_records,
        }
        print("TRACE_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        print(f"TRACE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
