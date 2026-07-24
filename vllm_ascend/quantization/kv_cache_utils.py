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
"""KV cache quantization utilities for Ascend NPU.

This module provides a lightweight dispatch mechanism for ``--kv-cache-dtype``
based KV cache quantization. It is *not* used for model-level quantization
schemes (e.g. C8/QuaRot), which have their own path through
:class:`AscendKVCacheMethod` in :mod:`vllm_ascend.quantization.method_adapters`.

Usage::

    from vllm_ascend.quantization.kv_cache_utils import get_kv_cache_scheme

    scheme = get_kv_cache_scheme("int4")
    if scheme is not None:
        scheme.create_weights(layer)
"""

import torch
from vllm.logger import init_logger

from .methods import get_scheme_class

logger = init_logger(__name__)

#: Mapping from ``cache_dtype`` strings to scheme registration keys.
#: Each entry ``"<cache_dtype>" → "<KV_<DTYPE>", "attention">`` so that
#: :func:`get_scheme_class` can look up the corresponding handler.
_CACHE_DTYPE_TO_SCHEME_KEY: dict[str, str] = {
    "int4": "KV_INT4",
    "nvfp4": "KV_NVFP4",
    "fp8_e4m3": "KV_FP8_E4M3",
    "fp4_e2m1": "KV_FP4_E2M1",
}


def get_kv_cache_scheme(cache_dtype: str):
    """Return the registered KV cache scheme for *cache_dtype*, or ``None``.

    Args:
        cache_dtype: The ``cache_dtype`` string from ``--kv-cache-dtype``
            (e.g. ``"int4"``, ``"nvfp4"``).

    Returns:
        An :class:`AscendAttentionScheme` subclass instance, or ``None``
        if no scheme is registered for the given dtype.
    """
    scheme_key = _CACHE_DTYPE_TO_SCHEME_KEY.get(cache_dtype)
    if scheme_key is None:
        return None
    scheme_cls = get_scheme_class(scheme_key, "attention")
    if scheme_cls is None:
        logger.warning(
            "No KV cache scheme registered for '--kv-cache-dtype %s' "
            "(scheme_key=%s). Falling back to unquantized path.",
            cache_dtype,
            scheme_key,
        )
        return None
    return scheme_cls()


def setup_kv_cache_quant(layer: torch.nn.Module, cache_dtype: str) -> None:
    """Set up KV cache quantization on *layer* for the given *cache_dtype*.

    This is called by the model runner after creating attention layers.
    If *cache_dtype* is not a quantized dtype known to this plugin, the
    call is a no-op.

    Args:
        layer: The attention layer (:class:`AttentionLayer` or subclass).
        cache_dtype: The ``cache_dtype`` string from ``--kv-cache-dtype``.
    """
    if not cache_dtype or cache_dtype in ("auto", "float16", "bfloat16"):
        return

    scheme = get_kv_cache_scheme(cache_dtype)
    if scheme is None:
        return

    scheme.create_weights(layer)
    logger.info(
        "KV cache quantization enabled: dtype=%s, scheme=%s",
        cache_dtype,
        type(scheme).__name__,
    )