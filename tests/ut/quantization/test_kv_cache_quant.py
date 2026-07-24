#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for KV cache quantization handlers.

Tests cover the four KV cache quant methods (INT4, NVFP4, FP8_E4M3, FP4_E2M1)
as well as the utility functions in ``kv_cache_utils``.

These tests do NOT require NPU hardware.
"""

# ---------------------------------------------------------------------------
# IMPORTANT: Mock torch_npu BEFORE any vllm/vllm_ascend imports.
# The parent conftest (tests/ut/conftest.py) imports vllm_ascend which
# requires torch_npu.  We use --noconftest to skip the parent conftest
# and set up the mock here instead.
# ---------------------------------------------------------------------------
import importlib.machinery
import os
import sys
import types
from unittest.mock import MagicMock

os.environ.setdefault("VLLM_VERSION", "0.19.1rc1")


def _make_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


_mock_torch_npu = _make_mock_module("torch_npu", attrs={
    "is_available": MagicMock(return_value=False),
    "Event": MagicMock(),
    "Stream": MagicMock(),
    "current_stream": MagicMock(return_value=MagicMock()),
    "current_device": MagicMock(return_value=0),
    "synchronize": MagicMock(),
    "set_device": MagicMock(),
    "device_count": MagicMock(return_value=0),
    "mem_get_info": MagicMock(return_value=(0, 0)),
    "get_device_name": MagicMock(return_value="MockNPU"),
    "get_device_properties": MagicMock(),
    "manual_seed_all": MagicMock(),
    "reset_peak_memory_stats": MagicMock(),
    "max_memory_allocated": MagicMock(return_value=0),
    "empty_cache": MagicMock(),
    "is_current_stream_capturing": MagicMock(return_value=False),
    "set_compile_mode": MagicMock(),
})
sys.modules["torch_npu"] = _mock_torch_npu

import torch  # noqa: E402
torch.npu = _mock_torch_npu

# Mock torchaudio to avoid libcudart.so loading errors
sys.modules["torchaudio"] = _make_mock_module("torchaudio")

import pytest  # noqa: E402

from vllm_ascend.quantization.kv_cache_utils import (
    get_kv_cache_scheme,
    setup_kv_cache_quant,
)
from vllm_ascend.quantization.methods.kv_int4 import AscendKVCacheInt4Method
from vllm_ascend.quantization.methods.kv_nvfp4 import AscendKVCacheNVFP4Method
from vllm_ascend.quantization.methods.kv_fp8_e4m3 import AscendKVCacheFP8E4M3Method
from vllm_ascend.quantization.methods.kv_fp4_e2m1 import AscendKVCacheFP4E2M1Method

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_layer():
    """Return a bare ``nn.Module`` to serve as a dummy attention layer."""
    return torch.nn.Module()


# ===========================================================================
# AscendKVCacheInt4Method
# ===========================================================================


class TestAscendKVCacheInt4Method:

    def test_create_weights_sets_uint8_dtype(self):
        layer = _make_layer()
        scheme = AscendKVCacheInt4Method()
        scheme.create_weights(layer)

        assert layer.kv_cache_torch_dtype == torch.uint8
        assert hasattr(layer, "k_cache_scale")
        assert hasattr(layer, "v_cache_scale")
        assert isinstance(layer.k_cache_scale, torch.nn.Parameter)
        assert isinstance(layer.v_cache_scale, torch.nn.Parameter)

    def test_apply_raises_runtime_error(self):
        layer = _make_layer()
        scheme = AscendKVCacheInt4Method()
        dummy = torch.zeros(1)

        with pytest.raises(RuntimeError) as exc_info:
            scheme.apply(layer, dummy, dummy, dummy, None, None, None, None, None)

        assert "AscendKVCacheInt4Method.apply" in str(exc_info.value)
        assert "KV_INT4" in str(exc_info.value)

    def test_process_weights_after_loading_flattens_scales(self):
        layer = _make_layer()
        scheme = AscendKVCacheInt4Method()
        scheme.create_weights(layer)

        # Manually reshape scales to 2D to verify flattening
        layer.k_cache_scale.data = layer.k_cache_scale.data.view(1, 1)
        layer.v_cache_scale.data = layer.v_cache_scale.data.view(1, 1)

        scheme.process_weights_after_loading(layer)

        assert layer.k_cache_scale.data.dim() == 1
        assert layer.v_cache_scale.data.dim() == 1


# ===========================================================================
# AscendKVCacheNVFP4Method
# ===========================================================================


class TestAscendKVCacheNVFP4Method:

    def test_create_weights_sets_uint8_dtype(self):
        layer = _make_layer()
        scheme = AscendKVCacheNVFP4Method()
        scheme.create_weights(layer)

        assert layer.kv_cache_torch_dtype == torch.uint8

    def test_create_weights_does_not_add_scales(self):
        layer = _make_layer()
        scheme = AscendKVCacheNVFP4Method()
        scheme.create_weights(layer)

        assert not hasattr(layer, "k_cache_scale")
        assert not hasattr(layer, "v_cache_scale")

    def test_apply_raises_runtime_error(self):
        layer = _make_layer()
        scheme = AscendKVCacheNVFP4Method()
        dummy = torch.zeros(1)

        with pytest.raises(RuntimeError) as exc_info:
            scheme.apply(layer, dummy, dummy, dummy, None, None, None, None, None)

        assert "AscendKVCacheNVFP4Method.apply" in str(exc_info.value)
        assert "KV_NVFP4" in str(exc_info.value)

    def test_process_weights_after_loading_is_noop(self):
        layer = _make_layer()
        scheme = AscendKVCacheNVFP4Method()
        scheme.create_weights(layer)

        # Should not raise
        scheme.process_weights_after_loading(layer)


# ===========================================================================
# AscendKVCacheFP8E4M3Method
# ===========================================================================


class TestAscendKVCacheFP8E4M3Method:

    def test_create_weights_sets_fp8_dtype(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP8E4M3Method()
        scheme.create_weights(layer)

        assert layer.kv_cache_torch_dtype == torch.float8_e4m3fn
        assert hasattr(layer, "k_cache_scale")
        assert hasattr(layer, "v_cache_scale")
        assert isinstance(layer.k_cache_scale, torch.nn.Parameter)
        assert isinstance(layer.v_cache_scale, torch.nn.Parameter)

    def test_apply_raises_runtime_error(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP8E4M3Method()
        dummy = torch.zeros(1)

        with pytest.raises(RuntimeError) as exc_info:
            scheme.apply(layer, dummy, dummy, dummy, None, None, None, None, None)

        assert "AscendKVCacheFP8E4M3Method.apply" in str(exc_info.value)
        assert "KV_FP8_E4M3" in str(exc_info.value)

    def test_process_weights_after_loading_flattens_scales(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP8E4M3Method()
        scheme.create_weights(layer)

        layer.k_cache_scale.data = layer.k_cache_scale.data.view(1, 1)
        layer.v_cache_scale.data = layer.v_cache_scale.data.view(1, 1)

        scheme.process_weights_after_loading(layer)

        assert layer.k_cache_scale.data.dim() == 1
        assert layer.v_cache_scale.data.dim() == 1


# ===========================================================================
# AscendKVCacheFP4E2M1Method
# ===========================================================================


class TestAscendKVCacheFP4E2M1Method:

    def test_create_weights_sets_uint8_dtype(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP4E2M1Method()
        scheme.create_weights(layer)

        assert layer.kv_cache_torch_dtype == torch.uint8

    def test_create_weights_does_not_add_scales(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP4E2M1Method()
        scheme.create_weights(layer)

        assert not hasattr(layer, "k_cache_scale")
        assert not hasattr(layer, "v_cache_scale")

    def test_apply_raises_runtime_error(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP4E2M1Method()
        dummy = torch.zeros(1)

        with pytest.raises(RuntimeError) as exc_info:
            scheme.apply(layer, dummy, dummy, dummy, None, None, None, None, None)

        assert "AscendKVCacheFP4E2M1Method.apply" in str(exc_info.value)
        assert "KV_FP4_E2M1" in str(exc_info.value)

    def test_process_weights_after_loading_is_noop(self):
        layer = _make_layer()
        scheme = AscendKVCacheFP4E2M1Method()
        scheme.create_weights(layer)

        # Should not raise
        scheme.process_weights_after_loading(layer)


# ===========================================================================
# kv_cache_utils
# ===========================================================================


class TestKVCacheUtils:

    # -- get_kv_cache_scheme -------------------------------------------------

    @pytest.mark.parametrize("dtype,expected_cls", [
        ("int4", AscendKVCacheInt4Method),
        ("nvfp4", AscendKVCacheNVFP4Method),
        ("fp8_e4m3", AscendKVCacheFP8E4M3Method),
        ("fp4_e2m1", AscendKVCacheFP4E2M1Method),
    ])
    def test_get_kv_cache_scheme_returns_correct_type(self, dtype, expected_cls):
        """For each known dtype string, ``get_kv_cache_scheme`` returns a
        non-None instance of the correct handler class."""
        scheme = get_kv_cache_scheme(dtype)
        assert scheme is not None
        assert isinstance(scheme, expected_cls)

    @pytest.mark.parametrize("unknown_dtype", ["auto", "float16", "bfloat16", "foo", ""])
    def test_get_kv_cache_scheme_returns_none_for_unknown(self, unknown_dtype):
        """Unknown or unquantized dtype strings return ``None``."""
        assert get_kv_cache_scheme(unknown_dtype) is None

    # -- setup_kv_cache_quant ------------------------------------------------

    @pytest.mark.parametrize("dtype", ["int4", "nvfp4", "fp8_e4m3", "fp4_e2m1"])
    def test_setup_kv_cache_quant_sets_up_layer(self, dtype):
        """For quantized dtypes, the layer is set up with ``kv_cache_torch_dtype``."""
        layer = _make_layer()
        setup_kv_cache_quant(layer, dtype)
        assert hasattr(layer, "kv_cache_torch_dtype")

    def test_setup_kv_cache_quant_sets_expected_dtype_for_int4(self):
        layer = _make_layer()
        setup_kv_cache_quant(layer, "int4")
        assert layer.kv_cache_torch_dtype == torch.uint8

    def test_setup_kv_cache_quant_sets_expected_dtype_for_nvfp4(self):
        layer = _make_layer()
        setup_kv_cache_quant(layer, "nvfp4")
        assert layer.kv_cache_torch_dtype == torch.uint8

    def test_setup_kv_cache_quant_sets_expected_dtype_for_fp8_e4m3(self):
        layer = _make_layer()
        setup_kv_cache_quant(layer, "fp8_e4m3")
        assert layer.kv_cache_torch_dtype == torch.float8_e4m3fn

    def test_setup_kv_cache_quant_sets_expected_dtype_for_fp4_e2m1(self):
        layer = _make_layer()
        setup_kv_cache_quant(layer, "fp4_e2m1")
        assert layer.kv_cache_torch_dtype == torch.uint8

    @pytest.mark.parametrize("noop_dtype", ["auto", "float16", "bfloat16"])
    def test_setup_kv_cache_quant_noop_for_non_quantized(self, noop_dtype):
        """Non-quantized dtypes should be a no-op — the layer is not modified."""
        layer = _make_layer()
        setup_kv_cache_quant(layer, noop_dtype)
        assert not hasattr(layer, "kv_cache_torch_dtype")

    def test_setup_kv_cache_quant_noop_for_empty_string(self):
        layer = _make_layer()
        setup_kv_cache_quant(layer, "")
        assert not hasattr(layer, "kv_cache_torch_dtype")