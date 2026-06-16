# Sim-LLM configuration dataclass.
#
# Holds all runtime configuration for the Sim-LLM KV reuse subsystem,
# populated from SIMLLM_* environment variables at init time.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_ascend.envs import env_variables


@dataclass
class SimLLMConfig:
    """Runtime configuration for the Sim-LLM KV reuse subsystem.

    All fields are populated from SIMLLM_* environment variables.
    See vllm_ascend/envs.py for documentation of each variable.
    """

    # Feature gate
    enabled: bool = field(default_factory=lambda: env_variables["SIMLLM_ENABLED"]())

    # Similarity detection
    cosine_threshold: float = field(default_factory=lambda: env_variables["SIMLLM_COSINE_THRESHOLD"]())
    lsh_num_bits: int = field(default_factory=lambda: env_variables["SIMLLM_LSH_NUM_BITS"]())
    lsh_batch_threshold: int = field(default_factory=lambda: env_variables["SIMLLM_LSH_BATCH_THRESHOLD"]())

    # KV cache management
    kv_cache_size: int = field(default_factory=lambda: env_variables["SIMLLM_KV_CACHE_SIZE"]())

    # Sandwich configuration
    sandwich_bottom: int = field(default_factory=lambda: env_variables["SIMLLM_SANDWICH_BOTTOM"]())
    sandwich_top: int = field(default_factory=lambda: env_variables["SIMLLM_SANDWICH_TOP"]())

    # Embedding extraction
    embedding_pooling: str = field(default_factory=lambda: env_variables["SIMLLM_EMBEDDING_POOLING"]())

    # Batch deferral
    deferral_ratio: float = field(default_factory=lambda: env_variables["SIMLLM_DEFERRAL_RATIO"]())

    @classmethod
    def from_env(cls) -> "SimLLMConfig":
        """Create a SimLLMConfig from current environment variables.

        This is the primary constructor. It reads all SIMLLM_* env vars
        and returns a fully populated config object.
        """
        return cls()

    def __repr__(self) -> str:
        lines = ["SimLLMConfig:"]
        for fld in self.__dataclass_fields__:
            lines.append(f"  {fld}={getattr(self, fld)}")
        return "\n".join(lines)
