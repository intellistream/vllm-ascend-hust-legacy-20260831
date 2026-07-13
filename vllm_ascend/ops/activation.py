#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import time

import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.activation import (
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)

from vllm_ascend import envs
from vllm_ascend.utils import get_weight_prefetch_method

_GATE_HALF_INPLACE_REWRITE = "gate_half_inplace"
_probe_hits = 0


def _enable_gate_half_inplace_rewrite(x: torch.Tensor) -> bool:
    if envs.VLLM_ASCEND_MLP_MATERIALIZATION_REWRITE.lower() != _GATE_HALF_INPLACE_REWRITE:
        return False
    if x.shape[-1] % 2 != 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return x.is_contiguous()


def _silu_and_mul_reuse_gate_half_inplace(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    out = F.silu(gate, inplace=True)
    out.mul_(up)
    return out


def _maybe_probe_silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor,
    rewrite_enabled: bool,
) -> None:
    probe_file = envs.VLLM_ASCEND_MLP_MATERIALIZATION_PROBE_FILE
    if not probe_file:
        return
    try:
        if torch.compiler.is_compiling():
            return
    except AttributeError:
        pass

    global _probe_hits
    limit = envs.VLLM_ASCEND_MLP_MATERIALIZATION_PROBE_LIMIT
    if _probe_hits >= limit:
        return
    _probe_hits += 1

    half = x.shape[-1] // 2
    event = {
        "event": "ascend_silu_and_mul_forward_oot",
        "hit_index": _probe_hits,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "rewrite_env": envs.VLLM_ASCEND_MLP_MATERIALIZATION_REWRITE,
        "rewrite_enabled": rewrite_enabled,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "contiguous": x.is_contiguous(),
        "input_data_ptr": int(x.data_ptr()),
        "gate_half_data_ptr": int(x[..., :half].data_ptr()),
        "output_data_ptr": int(out.data_ptr()),
        "output_aliases_gate_half": out.data_ptr() == x[..., :half].data_ptr(),
        "output_numel": int(out.numel()),
        "output_element_size": int(out.element_size()),
    }
    try:
        with open(probe_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        pass


class AscendQuickGELU(QuickGELU):
    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        rewrite_enabled = _enable_gate_half_inplace_rewrite(x)
        if rewrite_enabled:
            out = _silu_and_mul_reuse_gate_half_inplace(x)
        else:
            out = torch_npu.npu_swiglu(x)
        _maybe_probe_silu_and_mul(x, out, rewrite_enabled)
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


class AscendSiluAndMulWithClamp(SiluAndMulWithClamp):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        x = torch.cat([gate, up], dim=-1)
        out = torch_npu.npu_swiglu(x)
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


class AscendSwigluOAIAndMul:
    def swiglu_oai_forward(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
        class MinimalSwigluOAIAndMul:
            def __init__(self):
                self.alpha = alpha
                self.limit = limit

        layer = MinimalSwigluOAIAndMul()
        return SwigluOAIAndMul.forward_native(layer, x)


class AscendSwigluStepAndMul:
    def swiglustep_forward(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")

        class MinimalSwigluStepAndMul:
            def __init__(self):
                self.limit = limit

        layer = MinimalSwigluStepAndMul()
        return SwigluStepAndMul.forward_native(layer, x)
