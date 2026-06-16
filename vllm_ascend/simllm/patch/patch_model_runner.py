# Sim-LLM model runner patch for NPUModelRunner.
#
# Follows the vllm_ascend/patch/worker/ convention (same pattern as
# patch_deepseek_mtp.py, patch_qwen3_next.py, etc.).
#
# Injected during worker __init__ via _apply_patches(). When SIMLLM_ENABLED=1,
# wraps NPUModelRunner.execute_model() with Sim-LLM preprocessing,
# similarity identification, KV preparation, and postprocessing hooks.
#
# Implemented in PLAN Phase 2 (Task 2.2).
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import logging

from vllm_ascend.envs import env_variables
from vllm_ascend.simllm.config import SimLLMConfig

logger = logging.getLogger(__name__)


def is_simllm_enabled() -> bool:
    """Check if Sim-LLM is enabled via SIMLLM_ENABLED env var."""
    return env_variables["SIMLLM_ENABLED"]()


def apply_simllm_patch() -> None:
    """Apply the Sim-LLM monkey-patch to NPUModelRunner.

    Called from vllm_ascend/patch/worker/__init__.py during worker startup.
    When SIMLLM_ENABLED=0, this function is a no-op.
    """
    if not is_simllm_enabled():
        logger.debug("Sim-LLM patch skipped (SIMLLM_ENABLED=0)")
        return

    config = SimLLMConfig.from_env()
    logger.info("Applying Sim-LLM patch with config=%s", config)

    # Phase 2 implementation:
    # from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    # _original_execute_model = NPUModelRunner.execute_model
    #
    # def _simllm_execute_model(self, scheduler_output, **kwargs):
    #     simllm_preprocess(self, scheduler_output, config)
    #     match_results = self._simllm_identifier.identify(...)
    #     simllm_prepare_kv(self, match_results, config)
    #     outputs = _original_execute_model(self, scheduler_output, **kwargs)
    #     simllm_postprocess(self, outputs, match_results)
    #     return outputs
    #
    # NPUModelRunner.execute_model = _simllm_execute_model

    logger.warning("Sim-LLM patch logic is not yet implemented (Phase 2)")
