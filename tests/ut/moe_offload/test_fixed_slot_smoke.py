#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.sew_offload.run_fixed_slot_smoke import (
    configure_sew_offload_env,
    load_manifest,
    load_inline_prompts_jsonl,
    make_inline_request,
    override_request_max_output_tokens,
    prepare_synthetic_smoke_manifest,
    run_fixed_slot_smoke,
    run_smoke,
)


def _config():
    return {
        "model": {
            "path": "/models/qwen3-moe",
            "tensor_parallel_size": 1,
        },
        "dataset": {"seed": 20260529},
        "workload_buckets": [
            {
                "name": "short_chat",
                "num_requests": 2,
                "prompt_tokens": [128, 256],
                "output_tokens": 16,
            }
        ],
    }


def test_prepare_synthetic_smoke_manifest_is_reusable_for_fixed_slot_smoke(tmp_path):
    manifest_path = tmp_path / "requests.jsonl"

    prepare_synthetic_smoke_manifest(
        config=_config(),
        manifest_path=manifest_path,
        requests_per_bucket=1,
        buckets={"short_chat"},
    )

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["request_id"] == "short_chat_0000"
    assert records[0]["dataset"] == "synthetic_smoke"
    assert records[0]["max_output_tokens"] == 16


def test_make_inline_request_builds_minimal_correctness_request():
    request = make_inline_request(prompt="hello", max_output_tokens=1)

    assert request == {
        "request_id": "inline_0000",
        "bucket": "inline",
        "prompt": "hello",
        "max_output_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "dataset": "inline_smoke",
    }


def test_load_inline_prompts_jsonl_builds_multiple_correctness_requests(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        json.dumps({"request_id": "p0", "prompt": "hello", "max_output_tokens": 3}) + "\n"
        + json.dumps({"request_id": "p1", "prompt": "explain moe"}) + "\n",
        encoding="utf-8",
    )

    requests = load_inline_prompts_jsonl(prompts_path, default_max_output_tokens=5)

    assert requests == [
        {
            "request_id": "p0",
            "bucket": "inline",
            "prompt": "hello",
            "max_output_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "dataset": "inline_smoke",
        },
        {
            "request_id": "p1",
            "bucket": "inline",
            "prompt": "explain moe",
            "max_output_tokens": 5,
            "temperature": 0.0,
            "top_p": 1.0,
            "dataset": "inline_smoke",
        },
    ]


def test_override_request_max_output_tokens_keeps_original_requests_immutable():
    requests = [
        {"request_id": "r0", "prompt": "a", "max_output_tokens": 128},
        {"request_id": "r1", "prompt": "b", "max_output_tokens": 64},
    ]

    overridden = override_request_max_output_tokens(requests, max_output_tokens=8)

    assert [req["max_output_tokens"] for req in overridden] == [8, 8]
    assert [req["max_output_tokens"] for req in requests] == [128, 64]


def test_run_fixed_slot_smoke_sets_env_and_writes_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "smoke"
    manifest_path = tmp_path / "requests.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "request_id": "short_chat_0000",
                "bucket": "short_chat",
                "prompt": "hello",
                "max_output_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        model=None,
        output_dir=str(output_dir),
        manifest=str(manifest_path),
        buckets="short_chat",
        max_requests=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_memory_mb=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        num_slots=2,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
        with_native_offload_backend=True,
    )
    generated = MagicMock()
    generated.request_id = "short_chat_0000"
    generated.outputs = [MagicMock(token_ids=[1, 2, 3, 4])]
    generated.metrics = MagicMock(
        first_token_latency=0.25,
        first_token_ts=10.0,
        last_token_ts=10.6,
    )

    with (
        patch("tools.sew_offload.run_fixed_slot_smoke.reset_moe_offload_runtime") as mock_reset,
        patch("tools.sew_offload.run_fixed_slot_smoke.LLM") as mock_llm_cls,
    ):
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = [generated]

        summary = run_fixed_slot_smoke(args, _config(), load_manifest(manifest_path, {"short_chat"}, 1))

    mock_reset.assert_called_once()
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] == "0"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "2"
    assert mock_llm_cls.call_args.kwargs["model"] == "/models/qwen3-moe"
    assert mock_llm_cls.call_args.kwargs["enable_expert_parallel"] is False
    assert mock_llm_cls.call_args.kwargs["offload_backend"] == "prefetch"
    assert mock_llm_cls.call_args.kwargs["offload_params"] == {"experts"}
    assert summary["status"] == "ok"
    assert summary["num_slots"] == 2
    assert summary["completed"] == 1
    assert summary["total_output_tokens"] == 4
    assert summary["ttft_ms"]["mean"] == 250.0
    assert summary["tpot_ms"]["mean"] == pytest.approx(200.0)
    summary_path = output_dir / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_configure_sew_offload_env_supports_default_and_trace_only_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "0")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "8")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_POLICY", "lru")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS", "16")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH", str(tmp_path / "stale_trace.jsonl"))
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", "0,1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD", "2")

    configure_sew_offload_env("no_offload", num_slots=8)
    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_POLICY" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD" not in os.environ

    configure_sew_offload_env("trace_only", num_slots=8)
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "0"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH"].endswith("moe_offload_trace.jsonl")

    configure_sew_offload_env(
        "fixed_slot_sync",
        num_slots=8,
        resident_layer_ids="1,2,3",
        release_original_expert_weights=True,
        layered_runtime=True,
        fanout_threshold=4,
    )
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"] == "1,2,3"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"] == "4"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH"].endswith("moe_offload_profile.jsonl")


def test_configure_sew_offload_env_supports_compute_bucket_fast_path(tmp_path, monkeypatch):
    plan_path = tmp_path / "sew_moe_p1_plan.json"
    monkeypatch.setenv("VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH", str(plan_path))
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "99")

    configure_sew_offload_env("compute_bucket_fast_path", num_slots=0)

    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY" not in os.environ
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "0"
    assert "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH" not in os.environ
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH"].endswith("moe_offload_profile.jsonl")

    configure_sew_offload_env(
        "compute_bucket_fast_path",
        num_slots=0,
        compute_bucket_plan_path=str(plan_path),
    )
    assert os.environ["VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH"] == str(plan_path)


def test_run_no_offload_smoke_omits_native_offload_kwargs_and_writes_outputs(tmp_path):
    output_dir = tmp_path / "smoke"
    args = argparse.Namespace(
        mode="no_offload",
        model=None,
        output_dir=str(output_dir),
        manifest=str(tmp_path / "requests.jsonl"),
        buckets="short_chat",
        max_requests=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_memory_mb=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        num_slots=2,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
    )
    request = {
        "request_id": "short_chat_0000",
        "bucket": "short_chat",
        "prompt": "hello",
        "max_output_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    generated = MagicMock()
    generated.request_id = "short_chat_0000"
    generated.outputs = [MagicMock(token_ids=[7, 8], text="hi")]

    with (
        patch("tools.sew_offload.run_fixed_slot_smoke.reset_moe_offload_runtime") as mock_reset,
        patch("tools.sew_offload.run_fixed_slot_smoke.LLM") as mock_llm_cls,
    ):
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = [generated]

        summary = run_smoke(args, _config(), [request])

    mock_reset.assert_called_once()
    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" not in os.environ
    assert "offload_backend" not in mock_llm_cls.call_args.kwargs
    assert "offload_params" not in mock_llm_cls.call_args.kwargs
    assert summary["mode"] == "no_offload"
    outputs = [json.loads(line) for line in (output_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert outputs == [
        {
            "request_id": "short_chat_0000",
            "output_text": "hi",
            "output_token_ids": [7, 8],
            "output_tokens": 2,
        }
    ]


def test_run_compute_bucket_fast_path_smoke_sets_plan_path_and_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "smoke"
    plan_path = tmp_path / "sew_moe_p1_plan.json"
    args = argparse.Namespace(
        mode="compute_bucket_fast_path",
        model=None,
        output_dir=str(output_dir),
        manifest=str(tmp_path / "requests.jsonl"),
        buckets="short_chat",
        max_requests=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_memory_mb=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        num_slots=99,
        resident_layer_ids="",
        release_original_expert_weights=False,
        layered_runtime=False,
        fanout_threshold=0,
        compute_bucket_plan_path=str(plan_path),
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
    )
    request = {
        "request_id": "short_chat_0000",
        "bucket": "short_chat",
        "prompt": "hello",
        "max_output_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    generated = MagicMock()
    generated.request_id = "short_chat_0000"
    generated.outputs = [MagicMock(token_ids=[7, 8], text="hi")]

    with (
        patch("tools.sew_offload.run_fixed_slot_smoke.reset_moe_offload_runtime"),
        patch("tools.sew_offload.run_fixed_slot_smoke.LLM") as mock_llm_cls,
    ):
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = [generated]

        summary = run_smoke(args, _config(), [request])

    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" not in os.environ
    assert os.environ["VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH"] == str(plan_path)
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "0"
    assert summary["mode"] == "compute_bucket_fast_path"
    assert summary["num_slots"] == 0
    assert summary["compute_bucket_plan_path"] == str(plan_path)
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))[
        "compute_bucket_plan_path"
    ] == str(plan_path)


def test_run_smoke_writes_moe_offload_profile_to_summary(tmp_path):
    output_dir = tmp_path / "smoke"
    args = argparse.Namespace(
        mode="fixed_slot_sync",
        model=None,
        output_dir=str(output_dir),
        manifest=str(tmp_path / "requests.jsonl"),
        buckets="short_chat",
        max_requests=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_memory_mb=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        num_slots=2,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
    )
    request = {
        "request_id": "short_chat_0000",
        "bucket": "short_chat",
        "prompt": "hello",
        "max_output_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    generated = MagicMock()
    generated.request_id = "short_chat_0000"
    generated.outputs = [MagicMock(token_ids=[7], text="hi")]
    profile = {
        "events": [{"name": "register_layer_for_fixed_slots", "seconds": 0.1}],
        "total_seconds_by_event": {"register_layer_for_fixed_slots": 0.1},
        "memory_ledger": {"registered_layers": 1},
    }
    profile_jsonl = output_dir / "moe_offload_profile.jsonl"

    with (
        patch("tools.sew_offload.run_fixed_slot_smoke.reset_moe_offload_runtime"),
        patch("tools.sew_offload.run_fixed_slot_smoke.get_moe_offload_runtime", create=True) as mock_runtime,
        patch("tools.sew_offload.run_fixed_slot_smoke.LLM") as mock_llm_cls,
    ):
        mock_runtime.return_value.profiling_summary.return_value = profile
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [generated]

        def build_llm(**kwargs):
            del kwargs
            profile_jsonl.write_text(
                json.dumps({"name": "release_original_expert_weights", "seconds": 0.2}) + "\n",
                encoding="utf-8",
            )
            return mock_llm

        mock_llm_cls.side_effect = build_llm

        summary = run_smoke(args, _config(), [request])

    assert summary["moe_offload_profile"]["events"] == profile["events"]
    assert summary["moe_offload_profile_jsonl_events"] == [
        {"name": "release_original_expert_weights", "seconds": 0.2}
    ]
    summary_path = output_dir / "summary.json"
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["moe_offload_profile"]["events"] == profile["events"]
    assert written["moe_offload_profile_jsonl_events"] == [
        {"name": "release_original_expert_weights", "seconds": 0.2}
    ]
