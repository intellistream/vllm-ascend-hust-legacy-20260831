#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Low-overhead aggregate cache-hit profiling for SimLLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger("vllm.simllm.cache_hit_profiler")

PROFILE_LOG_PREFIX = "SIMLLM_CACHE_PROFILE "


@dataclass
class SimLLMCacheHitProfiler:
    """Accumulate cache-match and effective prefill-rewrite counters."""

    interval: int = 20
    batches: int = 0
    queries: int = 0
    matches: int = 0
    rewrites: int = 0
    covered_tokens: int = 0
    stores: int = 0
    cache_size: int = 0
    _next_emit_at: int = 20

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("profile interval must be greater than zero")
        self._next_emit_at = self.interval

    def record_batch(
        self,
        query_count: int,
        match_results: dict[int, Any],
        rewritten: int,
        covered_tokens: int,
        stored: int,
        cache_size: int,
    ) -> dict[str, int | float] | None:
        """Record one query-bearing batch and emit at the configured interval."""
        if query_count == 0:
            return None

        self.batches += 1
        self.queries += query_count
        self.matches += sum(1 for result in match_results.values() if result.matched)
        self.rewrites += rewritten
        self.covered_tokens += covered_tokens
        self.stores += stored
        self.cache_size = cache_size

        if self.queries < self._next_emit_at:
            return None
        while self._next_emit_at <= self.queries:
            self._next_emit_at += self.interval

        snapshot = self.snapshot()
        logger.info(
            "%s%s",
            PROFILE_LOG_PREFIX,
            json.dumps(snapshot, sort_keys=True),
        )
        return snapshot

    def snapshot(self) -> dict[str, int | float]:
        """Return cumulative counters and derived rates."""
        return {
            "batches": self.batches,
            "queries": self.queries,
            "matches": self.matches,
            "match_rate": self.matches / self.queries if self.queries else 0.0,
            "rewrites": self.rewrites,
            "rewrite_rate": self.rewrites / self.queries if self.queries else 0.0,
            "covered_tokens": self.covered_tokens,
            "avg_covered_tokens_per_rewrite": (self.covered_tokens / self.rewrites if self.rewrites else 0.0),
            "stores": self.stores,
            "cache_size": self.cache_size,
        }
