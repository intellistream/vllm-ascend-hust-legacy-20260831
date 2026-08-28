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
"""NVFP4 KV cache quantization for dense-attention models."""

import torch
from vllm.logger import init_logger

from .base import AscendAttentionScheme
from .registry import register_scheme

logger = init_logger(__name__)


@register_scheme("KV_NVFP4", "attention")
class AscendKVCacheNVFP4Method(AscendAttentionScheme):
    """NVFP4 KV cache quantization for dense-attention models.

    Uses packed fp4 data + fp8 block scales format.
    Each block of 16 fp4 elements shares a single fp8 scale factor.
    """

    def __init__(self, quant_description: dict | None = None, prefix: str | None = None):
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.uint8

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        err_msg = (
            "[vllm-ascend/KV_NVFP4] AscendKVCacheNVFP4Method.apply should "
            "not be called. NVFP4 KV cache quantization is handled by the "
            "attention backend."
        )
        raise RuntimeError(err_msg)
