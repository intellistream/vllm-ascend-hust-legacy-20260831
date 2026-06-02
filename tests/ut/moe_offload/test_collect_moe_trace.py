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

import json
from argparse import Namespace

import pytest
import torch

from tools.sew_offload.collect_moe_trace import (
    collect_trace,
    load_manifest,
    prepare_synthetic_smoke_manifest,
)
from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime


def test_prepare_synthetic_smoke_manifest_writes_selected_bucket(tmp_path):
    config = {
        "dataset": {"seed": 20260529},
        "workload_buckets": [
            {
                "name": "short_chat",
                "num_requests": 2,
                "prompt_tokens": [128, 256],
                "output_tokens": 16,
            },
            {
                "name": "decode_heavy",
                "num_requests": 2,
                "prompt_tokens": [128, 512],
                "output_tokens": 32,
            },
        ],
    }

    manifest_path = tmp_path / "requests.jsonl"
    prepare_synthetic_smoke_manifest(
        config=config,
        manifest_path=manifest_path,
        requests_per_bucket=1,
        buckets={"short_chat"},
    )

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["request_id"] == "short_chat_0000"
    assert records[0]["bucket"] == "short_chat"
    assert records[0]["max_output_tokens"] == 16
    assert records[0]["dataset"] == "synthetic_smoke"
    assert records[0]["prompt"]


def test_load_manifest_filters_buckets_and_limits_requests(tmp_path):
    manifest_path = tmp_path / "requests.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "a", "bucket": "short_chat"}),
                json.dumps({"request_id": "b", "bucket": "decode_heavy"}),
                json.dumps({"request_id": "c", "bucket": "short_chat"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_manifest(manifest_path, buckets={"short_chat"}, max_requests=1)

    assert records == [{"request_id": "a", "bucket": "short_chat"}]


def test_load_manifest_rejects_empty_selection(tmp_path):
    manifest_path = tmp_path / "requests.jsonl"
    manifest_path.write_text(json.dumps({"request_id": "a", "bucket": "short_chat"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no requests selected"):
        load_manifest(manifest_path, buckets={"missing"}, max_requests=0)


def test_collect_trace_uses_output_as_cross_process_trace_path(tmp_path, monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            assert prompts == ["hello"]
            assert len(sampling_params) == 1
            assert use_tqdm is False
            get_moe_offload_runtime().trace_routing(
                layer_id=0,
                topk_ids=torch.tensor([[1, 2]], dtype=torch.int32),
                topk_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
                num_experts=8,
                mode="decode",
            )
            return []

    output_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr("tools.sew_offload.collect_moe_trace.LLM", FakeLLM)

    args = Namespace(
        model="/tmp/fake-model",
        output=str(output_path),
        max_model_len=16,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        kv_cache_memory_mb=16,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        ignore_eos=True,
    )
    config = {
        "model": {"path": "/tmp/fake-model", "tensor_parallel_size": 1},
        "dataset": {"seed": 20260529},
    }
    requests = [
        {
            "prompt": "hello",
            "max_output_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    ]

    num_records = collect_trace(args, config, requests)

    assert num_records == 1
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["expert_token_counts"] == {
        "1": 1,
        "2": 1,
    }
    assert "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH" in __import__("os").environ
