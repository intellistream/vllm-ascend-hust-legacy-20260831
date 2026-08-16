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

import importlib.util
import logging
import os

import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa
from vllm_ascend import envs
from vllm_ascend.utils import is_310p

if not is_310p():
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa
import vllm_ascend.patch.platform.patch_minimax_m2_config  # noqa
import vllm_ascend.patch.platform.patch_minimax_usage_accounting  # noqa
logger = logging.getLogger(__name__)

if importlib.util.find_spec("vllm.tool_parsers.glm4_moe_tool_parser") is not None:
    import vllm_ascend.patch.platform.patch_glm_tool_call_parser  # noqa
else:
    logger.warning(
        "Skipping GLM tool-call parser patch because "
        "vllm.tool_parsers.glm4_moe_tool_parser is unavailable."
    )
import vllm_ascend.patch.platform.patch_sched_yield  # noqa
import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa

openai_serving = importlib.import_module("vllm.entrypoints.openai.engine.serving")
if hasattr(openai_serving.OpenAIServing, "_parse_tool_calls_from_content"):
    import vllm_ascend.patch.platform.patch_tool_choice_none_content  # noqa
else:
    logger.warning(
        "Skipping forced tool-choice none-content patch because "
        "OpenAIServing._parse_tool_calls_from_content is unavailable."
    )

if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
