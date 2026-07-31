# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import Mock, patch

from vllm.v1.worker.gpu import model_runner as vllm_model_runner

from vllm_ascend.worker.v2 import model_runner as ascend_model_runner


def test_graph_manager_wrapper_forwards_lora_capture_cases() -> None:
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager
    model_runner = object()
    graph_manager = object()
    factory = Mock(return_value=graph_manager)

    with (
        patch.object(ascend_model_runner, "ModelAclGraphManager", factory),
        ascend_model_runner.graph_manager_wrapper(model_runner),
    ):
        result = vllm_model_runner.ModelCudaGraphManager(
            "config",
            "device",
            "mode",
            1,
            lora_capture_cases=[0, 1],
        )

    assert result is graph_manager
    factory.assert_called_once_with(
        "config",
        "device",
        "mode",
        1,
        model_runner,
        lora_capture_cases=[0, 1],
    )
    assert vllm_model_runner.ModelCudaGraphManager is original_graph_manager
