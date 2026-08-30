# SPDX-License-Identifier: Apache-2.0
"""Architecture predicates shared by Ascend model compatibility paths."""

_DENSE_QWEN2_ROPE_ARCHITECTURE = "Qwen2ForCausalLM"
_QWEN2_ROPE_ARCHITECTURES = frozenset({_DENSE_QWEN2_ROPE_ARCHITECTURE, "SliceGPTQwen2ForCausalLM"})


def uses_qwen2_rope(architectures: list[str] | None) -> bool:
    """Return whether a model architecture uses Qwen2 RoPE handling."""
    return bool(_QWEN2_ROPE_ARCHITECTURES.intersection(architectures or ()))


def defaults_to_native_qwen2_rope(architectures: list[str] | None) -> bool:
    """Return whether native Qwen2 RoPE fallback is the safe default."""
    return _DENSE_QWEN2_ROPE_ARCHITECTURE in (architectures or ())
