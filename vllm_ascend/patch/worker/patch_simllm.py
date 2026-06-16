#
# Copyright (c) 2025 vLLM-HUST. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
#
# Sim-LLM worker patch — applies Sim-LLM hooks to NPUModelRunner.
#
# This file is auto-imported by vllm_ascend/patch/worker/__init__.py
# during worker startup. When SIMLLM_ENABLED=1, it patches
# NPUModelRunner.execute_model() with Sim-LLM preprocessing,
# similarity identification, KV preparation, and postprocessing hooks.
#
# When SIMLLM_ENABLED=0, this import is a no-op (apply_simllm_patch
# returns early).

from vllm_ascend.simllm.patch.patch_model_runner import apply_simllm_patch

apply_simllm_patch()
