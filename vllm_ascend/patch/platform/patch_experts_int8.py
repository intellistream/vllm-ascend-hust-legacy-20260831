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
from vllm.model_executor.layers.fused_moe import FusedMoE, MoERunner
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
from vllm.version import __version__ as VLLM_VERSION

from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_online import (
    AscendW8A8OnlineFusedMoEMethod,
)


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    # vLLM 0.23.0 exposes FusedMoE as a class; newer versions replace it with
    # the MoERunner factory function, so isinstance must target MoERunner.
    if VLLM_VERSION.startswith("0.23.0"):
        return isinstance(layer, FusedMoE)
    return isinstance(layer, MoERunner)


def get_quant_method(self, layer: torch.nn.Module, prefix: str):
    """Route experts_int8 layers to Ascend NPU kernels.

    - LinearBase: left unquantized (matches upstream experts_int8).
    - FusedMoE: online per-row int8 quantization on the NPU INT8 cube.
    """
    if isinstance(layer, LinearBase):
        return UnquantizedLinearMethod()
    if _is_fused_moe_layer(layer):
        scheme = AscendW8A8OnlineFusedMoEMethod()
        return AscendFusedMoEMethod(scheme, layer.moe_config)
    return None


ExpertsInt8Config.get_quant_method = get_quant_method
