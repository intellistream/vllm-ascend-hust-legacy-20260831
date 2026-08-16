# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE

try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
except ImportError:
    # Pre-RoutedExperts upstream (e.g. the CI vLLM at d886c26): only the
    # FusedMoE class exists there.
    RoutedExperts = None

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config

# Importing the patch module applies the monkey-patch on ExpertsInt8Config.
from vllm_ascend.patch.platform import patch_experts_int8  # noqa: F401
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_online import (
    AscendW8A8OnlineFusedMoEMethod,
)


@pytest.fixture(autouse=True)
def _patch_online_scheme_deps():
    mocks = [
        patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group",
            return_value=Mock(device_group=Mock()),
        ),
        patch("torch.distributed.get_rank", return_value=0),
        patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config",
            return_value=Mock(
                multistream_overlap_gate=False,
                eplb_config=Mock(dynamic_eplb=False),
                enable_fused_mc2=0,
            ),
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config",
            return_value=Mock(
                model_config=Mock(dtype=torch.float16, enforce_eager=True),
                compilation_config=Mock(mode=Mock()),
            ),
        ),
    ]
    for m in mocks:
        m.start()
    yield
    for m in mocks:
        m.stop()


def test_linear_layer_left_unquantized():
    layer = Mock(spec=LinearBase)
    method = ExpertsInt8Config().get_quant_method(layer, prefix="model.layers.0.mlp.gate_proj")
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.skipif(RoutedExperts is None, reason="RoutedExperts unavailable in this vLLM version")
def test_routed_experts_routes_to_ascend_online_int8():
    # Upstream main uses RoutedExperts as the MoE layer type. Build a real
    # instance (no Mock) so the patched get_quant_method must match it.
    layer = object.__new__(RoutedExperts)
    layer.moe_config = Mock()
    method = ExpertsInt8Config().get_quant_method(layer, prefix="model.layers.0.mlp")
    assert isinstance(method, AscendFusedMoEMethod)
    assert isinstance(method.quant_method, AscendW8A8OnlineFusedMoEMethod)


@pytest.mark.skipif(RoutedExperts is not None, reason="RoutedExperts exists; FusedMoE fallback not used")
def test_fused_moe_fallback_routes_to_ascend_online_int8():
    # Pre-RoutedExperts upstream: the FusedMoE class is the MoE layer type.
    layer = object.__new__(FusedMoE)
    layer.moe_config = Mock()
    method = ExpertsInt8Config().get_quant_method(layer, prefix="model.layers.0.mlp")
    assert isinstance(method, AscendFusedMoEMethod)
    assert isinstance(method.quant_method, AscendW8A8OnlineFusedMoEMethod)


def test_unsupported_layer_returns_none():
    # A plain object is neither LinearBase nor a MoE layer.
    layer = torch.nn.Module()
    method = ExpertsInt8Config().get_quant_method(layer, prefix="foo")
    assert method is None
