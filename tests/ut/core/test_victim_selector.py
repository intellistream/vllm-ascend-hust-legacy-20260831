# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the victim selector protocol and no-op default.

BidKV-specific tests have been moved to the vllm-hust-ascend-bidkv plugin.
"""

from types import SimpleNamespace

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.victim_selector import (
    NoOpVictimSelector,
    VictimSelector,
    get_victim_selector,
)


def _make_request(
    request_id: str,
    *,
    priority: int = 0,
    arrival_time: float = 0.0,
):
    return SimpleNamespace(
        request_id=request_id,
        priority=priority,
        arrival_time=arrival_time,
    )


class TestNoOpVictimSelector:
    """Verify the default (no-plugin) victim selector matches upstream
    vLLM behaviour."""

    def test_fcfs_returns_tail(self):
        selector = NoOpVictimSelector()
        running = [
            _make_request("r1"),
            _make_request("r2"),
            _make_request("r3"),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r3"

    def test_priority_returns_highest_priority(self):
        selector = NoOpVictimSelector()
        running = [
            _make_request("r1", priority=1, arrival_time=1.0),
            _make_request("r2", priority=3, arrival_time=2.0),
            _make_request("r3", priority=2, arrival_time=3.0),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.PRIORITY)
        assert victim.request_id == "r2"

    def test_empty_running_raises(self):
        selector = NoOpVictimSelector()
        try:
            selector.pick_victim([], SchedulingPolicy.FCFS)
            assert False, "should have raised"
        except ValueError:
            pass


class TestGetVictimSelector:
    """Verify the plugin discovery factory."""

    def test_returns_noop_or_plugin(self):
        # When a plugin (e.g. BidKV) is installed, get_victim_selector
        # discovers and returns it.  Without a plugin installed it returns
        # NoOpVictimSelector.  Both satisfy the VictimSelector protocol.
        config = SimpleNamespace(additional_config={})
        selector = get_victim_selector(config)
        assert isinstance(selector, VictimSelector)

    def test_returns_noop_when_plugin_disabled(self):
        config = SimpleNamespace(
            additional_config={"victim_selector_plugin_disabled": True}
        )
        selector = get_victim_selector(config)
        assert isinstance(selector, NoOpVictimSelector)