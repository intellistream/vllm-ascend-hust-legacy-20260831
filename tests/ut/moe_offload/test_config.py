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

import os

from vllm_ascend.moe_offload.config import MoeOffloadConfig


def test_default_config_is_disabled(monkeypatch):
    for name in (
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY",
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS",
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY",
        "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES",
        "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = MoeOffloadConfig.from_env()

    assert cfg.enabled is False
    assert cfg.trace_only is False
    assert cfg.num_slots == 0
    assert cfg.policy == "deadline"
    assert cfg.max_phases == 2
    assert cfg.async_load is False
    assert cfg.trace_max_records == 4096


def test_env_config_parses_values(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "8")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_POLICY", "lru")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS", "16")

    cfg = MoeOffloadConfig.from_env()

    assert cfg.enabled is True
    assert cfg.trace_only is True
    assert cfg.num_slots == 8
    assert cfg.policy == "lru"
    assert cfg.max_phases == 1
    assert cfg.async_load is True
    assert cfg.trace_max_records == 16


def test_env_variables_are_registered(monkeypatch):
    import vllm_ascend.envs as envs_ascend

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "7")

    assert "VLLM_ASCEND_MOE_OFFLOAD_ENABLED" in envs_ascend.env_variables
    assert "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS" in envs_ascend.env_variables
    assert envs_ascend.VLLM_ASCEND_MOE_OFFLOAD_ENABLED is True
    assert envs_ascend.VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS == 7

    os.environ.pop("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", None)
    os.environ.pop("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", None)
