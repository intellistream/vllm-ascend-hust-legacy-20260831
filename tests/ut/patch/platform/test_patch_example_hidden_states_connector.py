# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, MLAAttentionSpec

from vllm_ascend.patch.platform.patch_example_hidden_states_connector import (
    _find_hidden_state_group_id,
)


@dataclass
class _Group:
    kv_cache_spec: object
    layer_names: list[str]


def _hidden_state_spec(block_size: int = 16) -> HiddenStateCacheSpec:
    return HiddenStateCacheSpec(
        block_size=block_size,
        num_kv_heads=3,
        head_size=128,
        dtype="bfloat16",
        cache_dtype_str="auto",
    )


def _mla_spec(block_size: int = 16) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype="bfloat16",
        cache_dtype_str="auto",
    )


def test_find_hidden_state_group_uses_spec_type_not_first_group():
    config = SimpleNamespace(
        kv_cache_groups=[
            _Group(_mla_spec(), ["model.layers.0.self_attn"]),
            _Group(_hidden_state_spec(), ["draft.cache_only_layers.36"]),
        ]
    )

    assert _find_hidden_state_group_id(config) == 1


def test_find_hidden_state_group_supports_unique_name_fallback():
    config = SimpleNamespace(
        kv_cache_groups=[
            _Group(_mla_spec(), ["model.layers.0.self_attn"]),
            _Group(_mla_spec(), ["draft.cache_only_layers.36"]),
        ]
    )

    assert _find_hidden_state_group_id(config) == 1


def test_find_hidden_state_group_rejects_ambiguous_groups():
    config = SimpleNamespace(
        kv_cache_groups=[
            _Group(_mla_spec(), ["model.layers.0.self_attn"]),
            _Group(_mla_spec(), ["model.layers.1.self_attn"]),
        ]
    )

    with pytest.raises(ValueError, match="Could not uniquely identify"):
        _find_hidden_state_group_id(config)
