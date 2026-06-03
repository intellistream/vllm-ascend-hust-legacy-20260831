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
#

from __future__ import annotations

from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger

from vllm_ascend.moe_offload.autoconfig import MOE_OFFLOAD_GB_ENV, apply_moe_offload_defaults

logger = init_logger(__name__)

_ORIGINAL_CREATE_ENGINE_CONFIG = EngineArgs.create_engine_config


def _patched_create_engine_config(self, *args, **kwargs):
    if apply_moe_offload_defaults(self):
        plan = getattr(self, "_ascend_moe_offload_autoconfig_plan", {})
        logger.info(
            "Enabled Ascend MoE offload autoconfig from %s. "
            "Using vLLM PrefetchOffloader; cpu_offload_gb/UVA remains disabled. "
            "Derived prefetch config: group_size=%s, num_in_group=%s, "
            "estimated_offloaded_layers=%s, estimated_offloaded_gb=%.2f.",
            MOE_OFFLOAD_GB_ENV,
            getattr(self, "offload_group_size", None),
            getattr(self, "offload_num_in_group", None),
            plan.get("estimated_offloaded_layers", 0),
            float(plan.get("estimated_offloaded_gb", 0.0)),
        )
    return _ORIGINAL_CREATE_ENGINE_CONFIG(self, *args, **kwargs)


if not getattr(EngineArgs.create_engine_config, "_ascend_moe_offload_autoconfig_patch", False):
    _patched_create_engine_config._ascend_moe_offload_autoconfig_patch = True
    EngineArgs.create_engine_config = _patched_create_engine_config
