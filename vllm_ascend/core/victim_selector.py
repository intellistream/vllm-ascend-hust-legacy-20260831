# SPDX-License-Identifier: Apache-2.0
"""Victim selector protocol and plugin discovery for vLLM Ascend.

This module defines the lightweight protocol that scheduler preemption
victim selection plugins must implement, a no-op default selector (that
matches upstream vLLM behaviour), and a factory that discovers and loads
plugins via the ``vllm_ascend.victim_selector`` entry-point group.

Plugins (e.g. BidKV) are installed separately and auto-registered via::

    [project.entry-points."vllm_ascend.victim_selector"]
    bidkv = "vllm_ascend_bidkv:BidkvVictimSelector"
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VictimSelector(Protocol):
    """Protocol that victim selection plugins must implement.

    The protocol is intentionally minimal so that third-party plugins
    (e.g. BidKV) can be developed and released independently of
    vllm-ascend-hust.
    """

    @classmethod
    def from_vllm_config(cls, vllm_config) -> VictimSelector:
        """Factory: build a selector from a vLLM ``VllmConfig``."""
        ...

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        """Pick the request to preempt from *running*."""
        ...

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        """Emit observability / metrics log line (optional)."""
        ...

    def export_metrics(self) -> dict[str, Any]:
        """Export internal metrics as a flat dict (optional)."""
        ...


# ---------------------------------------------------------------------------
# No-op default (equivalent to upstream vLLM behaviour)
# ---------------------------------------------------------------------------


class NoOpVictimSelector:
    """Default victim selector — behaves identically to upstream vLLM.

    * FCFS: always picks the last request in ``running``.
    * PRIORITY: picks the request with the highest priority (ties broken
      by earliest arrival).
    """

    @classmethod
    def from_vllm_config(cls, vllm_config) -> NoOpVictimSelector:
        return cls()

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        if not running:
            raise ValueError("running is empty, cannot pick victim")
        if policy == SchedulingPolicy.PRIORITY:
            return max(
                running,
                key=lambda request: (request.priority, request.arrival_time),
            )
        return running[-1]

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        pass  # no-op: no BidKV metrics to emit

    def export_metrics(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def get_victim_selector(vllm_config) -> VictimSelector:
    """Discover and instantiate a victim selector.

    Tries to load a plugin registered under the
    ``vllm_ascend.victim_selector`` entry-point group.  Falls back to
    ``NoOpVictimSelector`` if no plugin is installed or loading fails.
    """
    # Allow explicit opt-out via additional_config
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if additional_config.get("victim_selector_plugin_disabled"):
        return NoOpVictimSelector()

    try:
        # Python 3.12+ preferred API; fall back for 3.10/3.11
        from importlib.metadata import EntryPoints, entry_points

        eps: EntryPoints = entry_points(group="vllm_ascend.victim_selector")
        for ep in eps:
            try:
                selector_cls = ep.load()
                if hasattr(selector_cls, "from_vllm_config"):
                    return selector_cls.from_vllm_config(vllm_config)
            except Exception:
                # One plugin failed — try the next (if any)
                continue
    except Exception:
        pass

    return NoOpVictimSelector()


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (used by patch_balance_schedule.py)
# ---------------------------------------------------------------------------


class UnifiedVictimSelector:
    """Backward-compatible wrapper around :func:`get_victim_selector`.

    Previously this was a concrete class; after the plugin refactor it
    delegates to the factory which discovers and instantiates the
    appropriate selector (plugin or no-op).
    """

    @classmethod
    def from_vllm_config(cls, vllm_config) -> VictimSelector:
        return get_victim_selector(vllm_config)


# ---------------------------------------------------------------------------
# Shared helpers (kept in vllm-ascend so schedulers don't depend on plugins)
# ---------------------------------------------------------------------------


def infer_kv_utilization_from_scheduler(scheduler) -> float | None:
    """Return current KV-cache utilization ratio [0, 1] from a scheduler.

    Used by schedulers to pass ``kv_utilization`` to ``pick_victim`` so
    that plugins (e.g. BidKV) can gate utility-based selection on KV
    pressure without coupling to scheduler internals.
    """
    try:
        block_pool = scheduler.kv_cache_manager.block_pool
        total_blocks = float(getattr(block_pool, "num_gpu_blocks", 0) or 0)
        if total_blocks <= 0:
            return None

        free_block_queue = getattr(block_pool, "free_block_queue", None)
        free_blocks = float(
            getattr(free_block_queue, "num_free_blocks", 0) or 0
        )
        used_ratio = (total_blocks - free_blocks) / total_blocks
        return min(max(used_ratio, 0.0), 1.0)
    except AttributeError:
        return None