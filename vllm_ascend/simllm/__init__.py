# Sim-LLM: Optimizing LLM Inference at the Edge through Inter-Task KV Reuse
#
# This sub-package lives inside vllm-ascend-hust (vllm_ascend namespace).
# It follows the same conventions as other vllm_ascend subsystems (EPLB,
# kv-transfer, utility scheduling): code in a dedicated sub-package,
# monkey-patches registered through vllm_ascend/patch/worker/.
#
# Licensed under the Apache License, Version 2.0.

from vllm_ascend.simllm.config import SimLLMConfig

__all__ = ["SimLLMConfig"]
