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

# Patch upstream experts_int8 to route MoE experts to the Ascend NPU
# INT8 cube unit.
#
# vLLM's ExpertsInt8Config selects a TRITON-only MoE backend
# (select_int8_moe_backend) that does not run on Ascend. On NPU we
# override get_quant_method so that --quantization experts_int8
# loads fp16/bf16 expert weights and quantizes them online to int8,
# executing via npu_grouped_matmul_swiglu_quant (the W8A8_DYNAMIC MoE
# path exposed by AscendW8A8OnlineFusedMoEMethod).
#
# Linear layers are left unquantized, matching upstream semantics.

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

from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_online import (
    AscendW8A8OnlineFusedMoEMethod,
)


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    # Upstream main exposes the MoE layer as RoutedExperts. In v0.23.0 it is
    # a type alias for the FusedMoE class; in newer versions it is an
    # independent class and FusedMoE became a factory function, so isinstance
    # must target RoutedExperts. Fall back to FusedMoE only for the
    # pre-RoutedExperts upstream commit used by CI (d886c26).
    if RoutedExperts is not None:
        return isinstance(layer, RoutedExperts)
    return isinstance(layer, FusedMoE)


def get_quant_method(self, layer: torch.nn.Module, prefix: str, tid2eid: int | None = None):
    """Route experts_int8 layers to Ascend NPU kernels.

    - LinearBase: left unquantized (matches upstream experts_int8).
    - RoutedExperts: online per-row int8 quantization on the NPU INT8 cube.
    """
    if isinstance(layer, LinearBase):
        return UnquantizedLinearMethod()
    if _is_fused_moe_layer(layer):
        scheme = AscendW8A8OnlineFusedMoEMethod()
        return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
    return None


ExpertsInt8Config.get_quant_method = get_quant_method
