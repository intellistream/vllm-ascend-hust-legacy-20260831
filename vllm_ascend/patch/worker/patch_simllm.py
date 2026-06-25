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
#

"""Sim-LLM worker patch entry point.

Auto-loaded by ``vllm_ascend/patch/worker/__init__.py`` at worker init time.
When ``VLLM_ASCEND_SIMLLM_ENABLED=1``, wraps ``NPUModelRunner.execute_model()``
with Sim-LLM preprocessing / KV reuse / postprocessing hooks.
When disabled the patch is a silent no-op.
"""

from __future__ import annotations

import logging
import os
import sys

from vllm_ascend.simllm.patch.patch_model_runner import apply_simllm_patch

logger = logging.getLogger(__name__)

SIMLLM_PATCH_MODULE = "vllm_ascend.simllm.patch.patch_model_runner"


def _method_label(method: object) -> str:
    module = getattr(method, "__module__", type(method).__module__)
    name = getattr(method, "__name__", type(method).__name__)
    return f"{module}.{name}"


def _is_simllm_method(method: object) -> bool:
    return getattr(method, "__module__", "") == SIMLLM_PATCH_MODULE


def describe_model_runner_patch_state(model_runner: object) -> str:
    """Return a compact description of active model-runner method bindings."""
    runner_cls = model_runner.__class__
    execute_model = getattr(model_runner, "execute_model", None)
    model_forward = getattr(model_runner, "_model_forward", None)
    return (
        f"runner={runner_cls.__module__}.{runner_cls.__name__} "
        f"execute_model={_method_label(execute_model)} "
        f"_model_forward={_method_label(model_forward)}"
    )


def log_simllm_patch_state(model_runner: object) -> None:
    """Log whether the instantiated runner is using Sim-LLM methods."""
    if os.getenv("VLLM_ASCEND_SIMLLM_ENABLED", "0") != "1":
        return

    execute_model = getattr(model_runner, "execute_model", None)
    model_forward = getattr(model_runner, "_model_forward", None)
    state = describe_model_runner_patch_state(model_runner)
    if _is_simllm_method(execute_model) and _is_simllm_method(model_forward):
        logger.warning("SimLLM worker patch state: %s", state)
    else:
        logger.warning("SimLLM requested but worker runner is not patched: %s", state)


def try_apply_simllm_patch() -> None:
    """Apply Sim-LLM only after ``NPUModelRunner`` is fully defined.

    ``vllm_ascend.patch.worker`` is imported while ``model_runner_v1`` is still
    loading.  Applying immediately would import ``NPUModelRunner`` from a
    partially initialized module and fail with a circular import.  The model
    runner calls this function again at the end of its module.
    """
    module = sys.modules.get("vllm_ascend.worker.model_runner_v1")
    model_runner_cls = getattr(module, "NPUModelRunner", None)
    if model_runner_cls is None:
        logger.debug("SimLLM patch deferred until NPUModelRunner is defined.")
        return

    apply_simllm_patch(model_runner_cls)


try_apply_simllm_patch()
