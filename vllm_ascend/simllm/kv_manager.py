# KV_Manager — Cached task store with LRU eviction and LSH bucket indexing.
#
# Stores processed tasks' metadata and top-layer KV. Operates at the
# task/semantic level, distinct from vLLM's token/block-level BlockManager.
#
# Implemented in PLAN Phase 1 (Task 1.1).
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CachedTask:
    """A task stored in the KV_Manager cache."""

    task_id: str
    embedding: torch.Tensor  # [1, D] pooled task embedding
    lsh_hash: int  # uint64 SimHash
    top_k: torch.Tensor  # [1, num_kv_heads, L_kv, head_dim]
    top_v: torch.Tensor  # [1, num_kv_heads, L_kv, head_dim]
    last_access_time: float  # monotonic timestamp for LRU
    seq_len: int  # original sequence length (for shape compat)


class KVManager:
    """LRU cache for task top-layer KV with LSH bucket indexing.

    Uses OrderedDict for O(1) LRU eviction and dict-of-lists for
    LSH bucket → task_id mapping.
    """

    def __init__(self, max_cache_size: int = 1024):
        self._max_cache_size: int = max_cache_size
        self._cache: OrderedDict[str, CachedTask] = OrderedDict()
        self._buckets: dict[int, list[str]] = {}

    def store(self, task: CachedTask) -> None:
        """Insert a task into cache. Evicts LRU if over capacity."""
        pass

    def lookup_by_hash(self, lsh_hash: int) -> list[CachedTask]:
        """Return all cached tasks in the given LSH bucket.

        Refreshes last_access_time for returned tasks (LRU refresh on access).
        """
        pass

    def get_kv(self, task_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Retrieve top-layer (K, V) for a cached task."""
        pass

    def evict_lru(self) -> None:
        """Evict the least-recently-used task. Called internally when full."""
        pass

    def size(self) -> int:
        """Return number of cached tasks."""
        return len(self._cache)

    def clear(self) -> None:
        """Reset all state (used for model reload / testing)."""
        pass
