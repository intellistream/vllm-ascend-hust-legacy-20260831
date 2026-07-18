from __future__ import annotations

import queue
from contextlib import nullcontext
from types import SimpleNamespace

from vllm_ascend.simple_kv_offload import copy_backend as copy_backend_module
from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend


class _RecordingStream:
    def __init__(self) -> None:
        self.waited_for = []

    def wait_event(self, event) -> None:
        self.waited_for.append(event)


def test_store_copy_waits_for_compute_done_event(monkeypatch):
    backend = NPUDmaCopyBackend()
    backend._device = "npu:7"
    backend._load_stream = _RecordingStream()
    backend._store_stream = _RecordingStream()
    backend._store_params = object()
    backend._queue = queue.SimpleQueue()

    copied = []
    recorded = []
    fake_event = SimpleNamespace(record=lambda stream: recorded.append(stream))
    fake_npu = SimpleNamespace(
        Event=lambda: fake_event,
        set_device=lambda device: None,
        stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(copy_backend_module.torch, "npu", fake_npu)
    monkeypatch.setattr(
        copy_backend_module,
        "copy_blocks",
        lambda src, dst, params: copied.append((src, dst, params)),
    )

    completion_events = []
    wait_event = object()
    backend.launch_copy(
        [1],
        [2],
        is_store=True,
        event_idx=3,
        events_list=completion_events,
        wait_event=wait_event,
    )
    backend._queue.put(None)
    backend._copy_loop()

    assert backend._store_stream.waited_for == [wait_event]
    assert copied == [([1], [2], backend._store_params)]
    assert recorded == [backend._store_stream]
    assert completion_events == [(3, fake_event)]
