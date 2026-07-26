# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, MLAAttentionSpec

from vllm_ascend.patch.platform.patch_example_hidden_states_connector import (
    _find_hidden_state_group_id,
    _patch_hidden_state_block_tracking,
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


def _manager_instance():
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    class _Manager(SingleTypeKVCacheManager):
        def find_longest_cache_hit(self, *args, **kwargs):
            return []

        def get_num_common_prefix_blocks(self, *args, **kwargs):
            return 0

    return object.__new__(_Manager)


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


def test_hidden_state_manager_patch_tracks_new_blocks(monkeypatch):
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    class _Block:
        def __init__(self, block_id):
            self.block_id = block_id

    hidden_manager = _manager_instance()
    hidden_manager.kv_cache_spec = _hidden_state_spec()
    hidden_manager.new_block_ids = []
    hidden_manager.req_to_blocks = {"request": []}
    hidden_manager._null_block = _Block(-1)

    monkeypatch.delattr(
        SingleTypeKVCacheManager,
        "_ascend_hidden_state_block_tracking_patch",
        raising=False,
    )
    monkeypatch.setattr(
        SingleTypeKVCacheManager,
        "allocate_new_blocks",
        lambda self, *args, **kwargs: [_Block(3), _Block(5)],
    )
    monkeypatch.setattr(
        SingleTypeKVCacheManager,
        "allocate_new_computed_blocks",
        lambda self, *args, **kwargs: None,
    )

    _patch_hidden_state_block_tracking()

    blocks = hidden_manager.allocate_new_blocks("request", 2, 2)
    assert [block.block_id for block in blocks] == [3, 5]
    assert hidden_manager.new_block_ids == [3, 5]


def test_hidden_state_manager_patch_does_not_double_track_upstream_fix(monkeypatch):
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    class _Block:
        def __init__(self, block_id):
            self.block_id = block_id

    hidden_manager = _manager_instance()
    hidden_manager.kv_cache_spec = _hidden_state_spec()
    hidden_manager.new_block_ids = []
    hidden_manager.req_to_blocks = {"request": []}
    hidden_manager._null_block = _Block(-1)

    def _already_fixed(self, *args, **kwargs):
        blocks = [_Block(7)]
        self.new_block_ids.extend(block.block_id for block in blocks)
        return blocks

    monkeypatch.delattr(
        SingleTypeKVCacheManager,
        "_ascend_hidden_state_block_tracking_patch",
        raising=False,
    )
    monkeypatch.setattr(
        SingleTypeKVCacheManager,
        "allocate_new_blocks",
        _already_fixed,
    )
    monkeypatch.setattr(
        SingleTypeKVCacheManager,
        "allocate_new_computed_blocks",
        lambda self, *args, **kwargs: None,
    )

    _patch_hidden_state_block_tracking()
    hidden_manager.allocate_new_blocks("request", 1, 1)
    assert hidden_manager.new_block_ids == [7]


def test_hidden_state_manager_patch_accepts_refactored_current_core(monkeypatch):
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    monkeypatch.delattr(
        SingleTypeKVCacheManager,
        "_ascend_hidden_state_block_tracking_patch",
        raising=False,
    )
    monkeypatch.delattr(
        SingleTypeKVCacheManager,
        "allocate_new_computed_blocks",
        raising=False,
    )

    _patch_hidden_state_block_tracking()

    assert SingleTypeKVCacheManager._ascend_hidden_state_block_tracking_patch is True
