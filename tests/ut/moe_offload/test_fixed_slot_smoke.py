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
    )
    generated = MagicMock()
    generated.request_id = "short_chat_0000"
    generated.outputs = [MagicMock(token_ids=[1, 2, 3, 4])]

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
    summary_path = output_dir / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_configure_sew_offload_env_supports_default_and_trace_only_modes(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "0")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "8")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_POLICY", "lru")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS", "16")

    configure_sew_offload_env("no_offload", num_slots=8)
    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_POLICY" not in os.environ
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS" not in os.environ

    configure_sew_offload_env("trace_only", num_slots=8)
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "0"


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
