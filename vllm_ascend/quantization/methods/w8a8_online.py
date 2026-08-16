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
"""W8A8 online int8 quantization for MoE experts on Ascend NPU.

This scheme corresponds to upstream vLLM's experts_int8 quantization:
expert weights are loaded as fp16/bf16 and quantized per-row to int8 during
process_weights_after_loading, then executed on the NPU INT8 cube unit
via npu_grouped_matmul_swiglu_quant (reusing the W8A8_DYNAMIC MoE path).
Linear layers are left unquantized, matching upstream experts_int8.
"""

from typing import Any

import torch
from vllm.model_executor.utils import replace_parameter

from .base import QuantType
from .registry import register_scheme
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


@register_scheme("W8A8_ONLINE", "moe")
class AscendW8A8OnlineFusedMoEMethod(AscendW8A8DynamicFusedMoEMethod):
    """Online per-row INT8 MoE quantization for Ascend NPU.

    Loads fp16/bf16 expert weights and quantizes them per-row to int8 during
    loading, mirroring upstream experts_int8 semantics. Forward
    computation reuses the W8A8_DYNAMIC MoE path which dispatches to the NPU
    INT8 cube unit (npu_grouped_matmul_swiglu_quant).
    """

    # Reuse the W8A8 INT8 cube path of the dynamic MoE scheme.
    quant_type: QuantType = QuantType.W8A8

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        # Load weights in floating point so they can be quantized online.
        # Upstream experts_int8 loads fp16/bf16 and quantizes per-row.
        param_dict: dict[str, Any] = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes,
            dtype=params_dtype,
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition,
            dtype=params_dtype,
        )
        return param_dict

    @staticmethod
    def _quantize_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row symmetric int8 quantization over the last dim.

        Args:
            weight: fp16/bf16 weight of shape [E, out_features, in_features].

        Returns:
            (int8_weight, weight_scale) where weight_scale has shape
            [E, out_features, 1] and the same dtype as weight.
        """
        vmax = torch.iinfo(torch.int8).max
        # Per output-channel symmetric scale over the input dim.
        scales = weight.abs().amax(dim=-1) / vmax
        # Guard against zero rows to avoid division by zero.
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        q = weight.div(scales.unsqueeze(-1)).round().clamp(-vmax, vmax).to(torch.int8)
        return q, scales.unsqueeze(-1).to(weight.dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Idempotency guard mirroring upstream Int8OnlineMoEMethod: vLLM may
        # call this more than once, and without the guard the already
        # quantized/transposed int8 weights would be re-quantized and
        # re-transposed, corrupting the layer.
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Online quantize fp16/bf16 expert weights to int8 per-row (symmetric),
        # matching upstream experts_int8 semantics.
        w13_q, w13_scale = self._quantize_per_row(layer.w13_weight.data)
        w2_q, w2_scale = self._quantize_per_row(layer.w2_weight.data)

        # Swap fp16 params for int8 params (preserves weight_loader attrs).
        replace_parameter(layer, "w13_weight", w13_q)
        replace_parameter(layer, "w2_weight", w2_q)

        # Fill the (empty) per-channel scale/offset params created by
        # get_dynamic_quant_param. Symmetric quantization -> offset = 0.
        layer.w13_weight_scale.data = w13_scale
        layer.w2_weight_scale.data = w2_scale
        layer.w13_weight_offset.data.zero_()
        layer.w2_weight_offset.data.zero_()

        # Delegate NZ format conversion + scale flattening to the parent
        # (W8A8_DYNAMIC) implementation, which transposes the int8 weights
        # to FRACTAL_NZ and prepares weight_scale_fp32 for the NPU kernel.
        super().process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True
