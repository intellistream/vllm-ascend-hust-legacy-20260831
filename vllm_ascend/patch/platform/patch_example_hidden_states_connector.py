# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compatibility fixes for the hidden-state extraction connector.

vLLM commit ebc6ef971 changed ``ExampleHiddenStatesConnector`` to defer
hidden-state extraction until a request finishes.  Before vLLM commit
5fd21eb0b, the scheduler-side connector kept the default KV-cache group index
zero because ``register_kv_caches`` only runs on workers.  Hidden states live
in a separate cache group, so request completion supplied block IDs from the
verifier cache and the connector saved zeros (or unrelated slots).

This feature-detected backport can be removed once every supported vLLM
revision includes vLLM #45849 and #46301.
"""

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec


def _find_hidden_state_group_id(kv_cache_config: Any | None) -> int:
    """Return the unique cache group used by ``extract_hidden_states``."""
    if kv_cache_config is None:
        return 0

    groups = kv_cache_config.kv_cache_groups
    group_ids = [
        group_id for group_id, group in enumerate(groups) if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
    ]
    if len(group_ids) == 1:
        return group_ids[0]

    # Some paired vLLM revisions lose the marker class while preserving the
    # dedicated cache-only layer name.  Keep this fallback narrow and unique.
    name_matches = [
        group_id
        for group_id, group in enumerate(groups)
        if any("cache_only_layers" in name for name in group.layer_names)
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    if not group_ids and not name_matches and len(groups) == 1:
        return 0

    raise ValueError(
        f"Could not uniquely identify the extract-hidden-states KV cache group among {len(groups)} groups."
    )


def _patch_hidden_state_block_tracking() -> None:
    """Backport vLLM #45849 block tracking without double-appending.

    Deferred extraction consumes the block IDs allocated since the previous
    scheduler step. Older paired cores only record attention-cache blocks, so
    the hidden-state group receives an empty list and the connector saves
    zeros. The before/after check also makes this wrapper safe on cores that
    already include the upstream fix.
    """
    manager_cls = SingleTypeKVCacheManager
    if getattr(manager_cls, "_ascend_hidden_state_block_tracking_patch", False):
        return

    original_allocate_new_blocks = manager_cls.allocate_new_blocks
    original_allocate_new_computed_blocks = getattr(manager_cls, "allocate_new_computed_blocks", None)

    def _patched_allocate_new_blocks(self: Any, *args: Any, **kwargs: Any) -> Any:
        tracked_before = len(self.new_block_ids)
        new_blocks = original_allocate_new_blocks(self, *args, **kwargs)
        if type(self.kv_cache_spec) is HiddenStateCacheSpec and len(self.new_block_ids) == tracked_before:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    if original_allocate_new_computed_blocks is not None:

        def _patched_allocate_new_computed_blocks(
            self: Any,
            request_id: str,
            new_computed_blocks: Any,
            num_local_computed_tokens: int,
            num_external_computed_tokens: int,
        ) -> Any:
            tracked_before = len(self.new_block_ids)
            req_blocks = self.req_to_blocks[request_id]
            existing_blocks = {id(block) for block in req_blocks}
            computed_blocks = {id(block) for block in new_computed_blocks}
            result = original_allocate_new_computed_blocks(
                self,
                request_id,
                new_computed_blocks,
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
            if (
                type(self.kv_cache_spec) is HiddenStateCacheSpec
                and num_external_computed_tokens > 0
                and len(self.new_block_ids) == tracked_before
            ):
                self.new_block_ids.extend(
                    block.block_id
                    for block in self.req_to_blocks[request_id]
                    if id(block) not in existing_blocks
                    and id(block) not in computed_blocks
                    and block is not self._null_block
                )
            return result

    manager_cls.allocate_new_blocks = _patched_allocate_new_blocks  # type: ignore[method-assign]
    if original_allocate_new_computed_blocks is not None:
        manager_cls.allocate_new_computed_blocks = (  # type: ignore[attr-defined]
            _patched_allocate_new_computed_blocks
        )
    manager_cls._ascend_hidden_state_block_tracking_patch = True  # type: ignore[attr-defined]


def _apply_patch() -> None:
    _patch_hidden_state_block_tracking()
    connector_cls = example_hidden_states_connector.ExampleHiddenStatesConnector

    # vLLM #46301 already provides the complete fix.
    if hasattr(connector_cls, "_find_cache_kv_group_id"):
        return

    original_init = connector_cls.__init__
    original_register = connector_cls.register_kv_caches

    def _patched_init(self, vllm_config, role, kv_cache_config):
        original_init(self, vllm_config, role, kv_cache_config)

        # The pre-ebc6ef connector saves during forward and has no deferred
        # group index.  Leave that implementation untouched.
        if not hasattr(self, "_hs_group_idx"):
            return

        group_id = _find_hidden_state_group_id(kv_cache_config)
        self._hs_group_idx = group_id
        if kv_cache_config is not None:
            self._block_size = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size

    def _patched_register_kv_caches(self, kv_caches):
        original_register(self, kv_caches)
        if not hasattr(self, "_hs_group_idx"):
            return

        # The delayed extractor computes slots with ``_block_size``.  Verify
        # that it matches the physical hidden-state cache view.
        kv_cache = getattr(self, "_kv_cache", None)
        if kv_cache is not None and self._block_size != kv_cache.shape[1]:
            raise ValueError(
                f"Hidden-states block-size mismatch: derived {self._block_size}, buffer has {kv_cache.shape[1]}."
            )

    connector_cls.__init__ = _patched_init
    connector_cls.register_kv_caches = _patched_register_kv_caches


_apply_patch()
