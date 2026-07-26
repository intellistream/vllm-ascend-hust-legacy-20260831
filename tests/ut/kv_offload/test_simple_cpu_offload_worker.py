"""验证 NPU Simple CPU Offload worker 的 store 事件传递。

配对的 vllm-hust 精确提交记录在 PR 描述中。
"""

import queue
import threading
from unittest.mock import MagicMock, patch

from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)
from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend
from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker


def test_npu_worker_store_records_and_forwards_compute_event() -> None:
    """Store submission follows the paired core's ordering contract."""
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    backend = MagicMock(spec=NPUDmaCopyBackend)
    worker._backend = backend
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        store_event=7,
        store_gpu_blocks=[3, 4],
        store_cpu_blocks=[11, 12],
    )
    worker._load_events = []
    worker._store_events = []
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._store_compute_done = None

    compute_event = MagicMock()
    compute_stream = object()
    with (
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.Event",
            return_value=compute_event,
        ),
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.cuda.current_stream",
            return_value=compute_stream,
        ),
    ):
        worker.get_finished(set())

    assert SimpleCPUOffloadNPUWorker.get_finished is SimpleCPUOffloadWorker.get_finished
    launch_kwargs = backend.launch_copy.call_args.kwargs
    if "wait_event" in launch_kwargs:
        compute_event.record.assert_called_once_with(compute_stream)
        assert launch_kwargs["wait_event"] is compute_event
    else:
        # The verified fixed core predates cross-stream store ordering and
        # submits the same transfer immediately.
        compute_event.record.assert_not_called()
    assert backend.launch_copy.call_args.args == ([3, 4], [11, 12])
    assert launch_kwargs["is_store"] is True
    assert launch_kwargs["event_idx"] == 7
    assert launch_kwargs["events_list"] is worker._store_events


def test_npu_worker_poll_checks_copy_thread_health() -> None:
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = MagicMock(spec=NPUDmaCopyBackend)
    worker._store_events = []
    worker._store_hwm = -1

    assert worker._poll_stream_events(is_store=True) == -1
    assert worker._backend.check_health.call_count == 2


def test_handle_preemptions_waits_for_background_copy_event() -> None:
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    backend = NPUDmaCopyBackend()
    backend._device = object()
    backend._load_stream = MagicMock()
    backend._store_stream = MagicMock()
    backend._queue = queue.SimpleQueue()
    backend._load_params = object()
    backend._store_params = object()
    worker._backend = backend
    worker._load_events = []
    worker._store_events = []
    worker._load_hwm = -1
    worker._store_hwm = -1

    copy_started = threading.Event()
    release_copy = threading.Event()
    copy_finished = threading.Event()
    flush_returned = threading.Event()
    completion_event = MagicMock()

    def blocking_copy(*_args) -> None:
        copy_started.set()
        assert release_copy.wait(timeout=2.0)
        copy_finished.set()

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
            return_value=completion_event,
        ),
    ):
        backend._thread = threading.Thread(target=backend._copy_loop)
        backend._thread.start()
        backend.launch_copy([1], [2], True, 7, worker._store_events)
        assert copy_started.wait(timeout=2.0)

        def handle_preemption() -> None:
            worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
            flush_returned.set()

        flush_thread = threading.Thread(target=handle_preemption)
        flush_thread.start()
        assert not flush_returned.wait(timeout=0.1)
        assert worker._store_events == []

        release_copy.set()
        flush_thread.join(timeout=2.0)
        backend.shutdown(timeout=2.0)

    assert not flush_thread.is_alive()
    assert copy_finished.is_set()
    assert flush_returned.is_set()
    completion_event.synchronize.assert_called_once_with()
    assert worker._store_hwm == 7
    assert worker._store_events == []


def test_npu_worker_shutdown_drains_backend() -> None:
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = MagicMock(spec=NPUDmaCopyBackend)

    worker.shutdown()

    worker._backend.shutdown.assert_called_once_with()


def test_connector_shutdown_routes_to_npu_worker() -> None:
    connector = AscendSimpleCPUOffloadConnector.__new__(AscendSimpleCPUOffloadConnector)
    connector.worker_handler = MagicMock(spec=SimpleCPUOffloadNPUWorker)

    connector.shutdown()

    connector.worker_handler.shutdown.assert_called_once_with()
