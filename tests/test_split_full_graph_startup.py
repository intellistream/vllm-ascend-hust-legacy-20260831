#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

import json
import os

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@wait_until_npu_memory_free()
def test_split_full_graph_startup():
    """Verify that vllm serve with FULL graph + inplace_parallel split can
    complete profile/capture and reach /health readiness without crashing.

    This is a startup smoke-test: it only checks that the server becomes
    healthy, not inference correctness or performance.

    Covers the regression where ``forward_oot()`` reading
    ``_EXTRA_CTX.is_draft_model`` caused a Dynamo ``Unsupported`` error
    inside the split FULL-graph capture path.
    """
    model = os.environ.get("VLLM_TEST_MODEL", _DEFAULT_MODEL)
    port = get_open_port()

    compilation_config = json.dumps({
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
    })

    additional_config = json.dumps({
        "split_batch_config": {
            "enabled": True,
            "mode": "inplace_parallel",
            "enable_parallel_streams": True,
            "num_splits": 2,
            "parallel_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
        },
    })

    server_args = [
        "--max_model_len", "1024",
        "--gpu_memory_utilization", "0.9",
        "--port", str(port),
        "--compilation-config", compilation_config,
        "--additional-config", additional_config,
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        server_port=port,
        auto_port=False,
        max_wait_seconds=600,
    ) as server:
        resp = server.url_for("health")
        assert resp.endswith("/health")