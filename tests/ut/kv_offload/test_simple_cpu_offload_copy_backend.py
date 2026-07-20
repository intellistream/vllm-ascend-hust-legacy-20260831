"""验证 NPU 简单 CPU Offload 拷贝后端的事件依赖。

输入：模拟的 NPU 流、事件和拷贝参数。
输出：确认拷贝任务携带事件，并在拷贝前等待该事件。
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend


def _make_backend() -> NPUDmaCopyBackend:
    backend = NPUDmaCopyBackend()
    backend._device = object()
    backend._load_stream = MagicMock()
    backend._store_stream = MagicMock()
    backend._queue = queue.SimpleQueue()
    backend._load_params = object()
    backend._store_params = object()
    return backend


def test_launch_copy_queues_wait_event() -> None:
    backend = _make_backend()
    wait_event = object()
    events_list = []

    backend.launch_copy([1], [2], True, 3, events_list, wait_event)

    assert backend._queue.get() == (
        [1],
        [2],
        backend._store_params,
        True,
        3,
        events_list,
        wait_event,
    )


def test_store_waits_for_compute_event_before_copy() -> None:
    backend = _make_backend()
    backend._queue.put(([1], [2], object(), True, 3, events_list := [], wait_event := MagicMock()))
    backend._queue.put(None)

    order = []
    completion_event = MagicMock()
    backend._store_stream.wait_event.side_effect = lambda event: order.append(("wait", event))

    with (
        patch("vllm_ascend.simple_kv_offload.copy_backend.torch.npu.set_device"),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.copy_blocks",
            side_effect=lambda *_args: order.append(("copy", None)),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.Event",
            return_value=completion_event,
        ),
    ):
        backend._copy_loop()

    assert order == [("wait", wait_event), ("copy", None)]
    completion_event.record.assert_called_once_with(backend._store_stream)
    assert events_list == [(3, completion_event)]


@pytest.mark.parametrize("failure_point", ["wait", "copy", "record"])
def test_background_failures_are_reported(failure_point: str) -> None:
    backend = _make_backend()
    failure = RuntimeError(f"{failure_point} failed")
    wait_event = MagicMock()
    completion_event = MagicMock()
    if failure_point == "wait":
        backend._store_stream.wait_event.side_effect = failure
    if failure_point == "record":
        completion_event.record.side_effect = failure

    backend._queue.put(([1], [2], object(), True, 3, [], wait_event))
    backend._queue.put(None)
    copy_side_effect = failure if failure_point == "copy" else None
    with (
        patch("vllm_ascend.simple_kv_offload.copy_backend.torch.npu.set_device"),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.copy_blocks",
            side_effect=copy_side_effect,
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.Event",
            return_value=completion_event,
        ),
    ):
        backend._copy_loop()

    with pytest.raises(RuntimeError, match="copy thread failed") as exc_info:
        backend.check_health()
    assert exc_info.value.__cause__ is failure

    with pytest.raises(RuntimeError, match="copy thread failed"):
        backend.launch_copy([1], [2], True, 4, [])


def test_shutdown_drains_queued_and_inflight_copies() -> None:
    backend = _make_backend()
    copy_started = threading.Event()
    release_copy = threading.Event()
    completion_events = [MagicMock(), MagicMock()]
    recorded_events = []
    copy_calls = []

    def blocking_copy(*args) -> None:
        copy_calls.append(args)
        if len(copy_calls) == 1:
            copy_started.set()
            assert release_copy.wait(timeout=2.0)

    shutdown_errors = []
    with (
        patch("vllm_ascend.simple_kv_offload.copy_backend.torch.npu.set_device"),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.copy_blocks",
            side_effect=blocking_copy,
        ),
        patch(
            "vllm_ascend.simple_kv_offload.copy_backend.torch.npu.Event",
            side_effect=completion_events,
        ),
    ):
        backend._thread = threading.Thread(target=backend._copy_loop)
        backend._thread.start()
        backend.launch_copy([1], [2], True, 1, recorded_events)
        assert copy_started.wait(timeout=2.0)
        backend.launch_copy([3], [4], True, 2, recorded_events)

        def run_shutdown() -> None:
            try:
                backend.shutdown(timeout=2.0)
            except Exception as exc:
                shutdown_errors.append(exc)

        shutdown_thread = threading.Thread(target=run_shutdown)
        shutdown_thread.start()
        release_copy.set()
        shutdown_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()
    assert not shutdown_errors
    assert len(copy_calls) == 2
    assert recorded_events == [(1, completion_events[0]), (2, completion_events[1])]
    backend._load_stream.synchronize.assert_called_once_with()
    backend._store_stream.synchronize.assert_called_once_with()

    backend.shutdown()
    backend._load_stream.synchronize.assert_called_once_with()
    backend._store_stream.synchronize.assert_called_once_with()


def test_shutdown_rejects_new_submissions() -> None:
    backend = _make_backend()
    backend.shutdown()

    with pytest.raises(RuntimeError, match="is shut down"):
        backend.launch_copy([1], [2], True, 1, [])


def test_shutdown_reports_thread_timeout() -> None:
    backend = _make_backend()
    copy_thread = MagicMock()
    copy_thread.is_alive.return_value = True
    backend._thread = copy_thread

    with pytest.raises(RuntimeError, match="did not stop"):
        backend.shutdown(timeout=0.0)

    copy_thread.join.assert_called_once_with(timeout=0.0)
    backend._load_stream.synchronize.assert_not_called()
    backend._store_stream.synchronize.assert_not_called()


def test_shutdown_reports_background_error_after_stream_drain() -> None:
    backend = _make_backend()
    failure = RuntimeError("copy failed")
    backend._background_error = failure

    with pytest.raises(RuntimeError, match="copy thread failed") as exc_info:
        backend.shutdown()

    assert exc_info.value.__cause__ is failure
    backend._load_stream.synchronize.assert_called_once_with()
    backend._store_stream.synchronize.assert_called_once_with()

    with pytest.raises(RuntimeError, match="copy thread failed"):
        backend.shutdown()
    backend._load_stream.synchronize.assert_called_once_with()
    backend._store_stream.synchronize.assert_called_once_with()


def test_shutdown_reports_stream_synchronization_error() -> None:
    backend = _make_backend()
    failure = RuntimeError("stream sync failed")
    backend._load_stream.synchronize.side_effect = failure

    with pytest.raises(RuntimeError, match="synchronize") as exc_info:
        backend.shutdown()

    assert exc_info.value.__cause__ is failure
    backend._load_stream.synchronize.assert_called_once_with()
    backend._store_stream.synchronize.assert_called_once_with()

    with pytest.raises(RuntimeError, match="synchronize"):
        backend.shutdown()
    assert backend._load_stream.synchronize.call_count == 2
    assert backend._store_stream.synchronize.call_count == 2
