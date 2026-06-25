#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
from argparse import ArgumentParser

import pytest

from vllm_ascend.moe_offload.autoconfig import (
    MOE_OFFLOAD_GB_ENV,
    apply_moe_offload_defaults,
    derive_prefetch_defaults,
    register_moe_offload_cli_arg,
)


QWEN3_30B_A3B_CONFIG = {
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}

_AUTOCONFIG_ENV_VARS = (
    MOE_OFFLOAD_GB_ENV,
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY",
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS",
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES",
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME",
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD",
    "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS",
    "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE",
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE",
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM",
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL",
)


@pytest.fixture(autouse=True)
def clean_moe_offload_autoconfig_env():
    original_env = {
        env_name: os.environ[env_name]
        for env_name in _AUTOCONFIG_ENV_VARS
        if env_name in os.environ
    }
    for env_name in _AUTOCONFIG_ENV_VARS:
        os.environ.pop(env_name, None)
    yield
    for env_name in _AUTOCONFIG_ENV_VARS:
        os.environ.pop(env_name, None)
    os.environ.update(original_env)


def test_moe_offload_autoconfig_is_disabled_when_env_is_unset(monkeypatch):
    monkeypatch.delenv(MOE_OFFLOAD_GB_ENV, raising=False)
    engine_args = object()

    assert apply_moe_offload_defaults(engine_args) is False
    assert not hasattr(engine_args, "offload_backend")


def test_moe_offload_autoconfig_is_disabled_when_env_is_zero(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "0")
    engine_args = object()

    assert apply_moe_offload_defaults(engine_args) is False
    assert not hasattr(engine_args, "offload_group_size")


def test_moe_offload_autoconfig_sets_prefetch_and_moe_defaults(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", raising=False)
    engine_args = type("EngineArgsStub", (), {})()

    assert apply_moe_offload_defaults(engine_args) is True

    assert engine_args.offload_backend == "prefetch"
    assert engine_args.offload_group_size == 4
    assert engine_args.offload_num_in_group == 1
    assert engine_args.offload_prefetch_step == 1
    assert engine_args.offload_params == {"experts"}
    assert engine_args.cpu_offload_gb == 0
    assert engine_args.cpu_offload_params == set()
    assert engine_args._ascend_moe_offload_autoconfig_applied is True

    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] == "0"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "8"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD"] == "0"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"] == "8"


def test_moe_offload_autoconfig_sew_dataplane_arms_seam_and_b2(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", raising=False)
    engine_args = type("EngineArgsStub", (), {})()

    assert apply_moe_offload_defaults(engine_args) is True

    # SEW data plane: seam + B2 armed, PrefetchOffloader NOT wired.
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL"] == "1"
    assert not hasattr(engine_args, "offload_backend")
    assert engine_args._ascend_moe_offload_sew_dataplane is True
    # Shared defaults + resident layers still derived from the GB budget.
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "8"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"]


def test_moe_offload_autoconfig_default_dataplane_is_prefetch(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", raising=False)
    engine_args = type("EngineArgsStub", (), {})()

    assert apply_moe_offload_defaults(engine_args) is True
    # Default path unchanged: PrefetchOffloader wired, seam/B2 NOT auto-armed.
    assert engine_args.offload_backend == "prefetch"
    assert engine_args._ascend_moe_offload_sew_dataplane is False
    assert "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM" not in os.environ


def test_moe_offload_gb_derives_prefetch_amount_from_model_config():
    defaults = derive_prefetch_defaults(14, QWEN3_30B_A3B_CONFIG)
    assert defaults["offload_group_size"] == 4
    assert defaults["offload_num_in_group"] == 1
    assert defaults["estimated_offloaded_layers"] == 12
    assert defaults["offloaded_layer_ids"] == tuple(range(3, 48, 4))
    assert 3 not in defaults["resident_layer_ids"]
    assert 2 in defaults["resident_layer_ids"]
    assert 13 <= defaults["estimated_offloaded_gb"] <= 15


def test_larger_moe_offload_gb_derives_more_prefetch_layers():
    defaults = derive_prefetch_defaults(28, QWEN3_30B_A3B_CONFIG)

    assert defaults["offload_group_size"] == 4
    assert defaults["offload_num_in_group"] == 2
    assert defaults["estimated_offloaded_layers"] == 24
    assert defaults["offloaded_layer_ids"][:4] == (2, 3, 6, 7)
    assert 2 not in defaults["resident_layer_ids"]
    assert 1 in defaults["resident_layer_ids"]
    assert 26 <= defaults["estimated_offloaded_gb"] <= 29


def test_moe_offload_autoconfig_sets_resident_layers_from_gb(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", raising=False)
    engine_args = type("EngineArgsStub", (), {"_ascend_moe_offload_model_config": QWEN3_30B_A3B_CONFIG})()

    assert apply_moe_offload_defaults(engine_args) is True

    resident_layer_ids = {
        int(item)
        for item in os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"].split(",")
        if item
    }
    assert 3 not in resident_layer_ids
    assert 7 not in resident_layer_ids
    assert 2 in resident_layer_ids
    assert 4 in resident_layer_ids


def test_moe_offload_autoconfig_preserves_explicit_resident_layers(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", "0,1")
    engine_args = type("EngineArgsStub", (), {"_ascend_moe_offload_model_config": QWEN3_30B_A3B_CONFIG})()

    assert apply_moe_offload_defaults(engine_args) is True

    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"] == "0,1"


def test_moe_offload_autoconfig_preserves_explicit_values(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "12")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "offload_backend": "auto",
            "offload_group_size": 6,
            "offload_num_in_group": 2,
            "offload_prefetch_step": 3,
            "offload_params": {"experts.w2_weight"},
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert engine_args.offload_backend == "prefetch"
    assert engine_args.offload_group_size == 6
    assert engine_args.offload_num_in_group == 2
    assert engine_args.offload_prefetch_step == 3
    assert engine_args.offload_params == {"experts.w2_weight"}
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "12"


@pytest.mark.parametrize(
    "engine_args",
    [
        type("EngineArgsStub", (), {"cpu_offload_gb": 4})(),
        type("EngineArgsStub", (), {"offload_backend": "uva"})(),
    ],
)
def test_moe_offload_autoconfig_rejects_uva_conflicts(monkeypatch, engine_args):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")

    with pytest.raises(ValueError, match="cpu_offload_gb"):
        apply_moe_offload_defaults(engine_args)


def test_moe_offload_platform_patch_applies_defaults_before_engine_config(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", raising=False)
    from vllm_ascend.patch.platform import patch_moe_offload_autoconfig as patch_mod

    engine_args = type("EngineArgsStub", (), {})()

    def fake_create_engine_config(self, *args, **kwargs):
        return {
            "backend": self.offload_backend,
            "group_size": self.offload_group_size,
            "offload_params": self.offload_params,
        }

    monkeypatch.setattr(patch_mod, "_ORIGINAL_CREATE_ENGINE_CONFIG", fake_create_engine_config)

    result = patch_mod._patched_create_engine_config(engine_args)

    assert result == {
        "backend": "prefetch",
        "group_size": 4,
        "offload_params": {"experts"},
    }


@pytest.mark.parametrize("value", ["-1", "abc", ""])
def test_moe_offload_autoconfig_rejects_invalid_env(monkeypatch, value):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, value)

    with pytest.raises(ValueError, match=MOE_OFFLOAD_GB_ENV):
        apply_moe_offload_defaults(object())


def test_moe_offload_cli_arg_sets_autoconfig_env(monkeypatch):
    monkeypatch.delenv(MOE_OFFLOAD_GB_ENV, raising=False)
    parser = ArgumentParser()

    register_moe_offload_cli_arg(parser)
    args = parser.parse_args(["--ascend-moe-offload-gb", "14"])

    assert args.ascend_moe_offload_gb == 14
    assert os.environ[MOE_OFFLOAD_GB_ENV] == "14"


def test_moe_offload_cli_arg_is_idempotent():
    parser = ArgumentParser()

    register_moe_offload_cli_arg(parser)
    register_moe_offload_cli_arg(parser)

    assert "--ascend-moe-offload-gb" in parser._option_string_actions
