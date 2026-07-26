"""DMA copy backend for NPU<->CPU block transfers.

Mirrors :class:`vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend`
but routes batched memcpy through ``torch.ops._C_ascend.swap_blocks_batch``
and uses ``torch.npu`` streams/events.
"""

from __future__ import annotations

import queue
import threading

import torch

from vllm_ascend.simple_kv_offload.npu_mem_ops import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class NPUDmaCopyBackend:
    """``aclrtMemcpyBatchAsync`` copy backend running on a worker thread.

    Two pre-built ``BatchMemcpyParams`` are cached (load=H2D, store=D2H).
    Submitted jobs are dispatched in FIFO order to a single worker
    thread; each job issues its copies on a dedicated NPU stream and
    records an Event the main thread can poll without synchronizing
    the device.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.npu.Stream | None = None
        self._store_stream: torch.npu.Stream | None = None
        self._device: torch.device | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._idle_condition = threading.Condition(self._state_lock)
        self._shutdown_lock = threading.Lock()
        self._background_error: Exception | None = None
        self._outstanding_jobs: int = 0
        self._shutdown: bool = False
        self._shutdown_complete: bool = False

    def init(
        self,
        npu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.npu.Stream,
        store_stream: torch.npu.Stream,
    ) -> None:
        # Stores go NPU->CPU (D2H), loads go CPU->NPU (H2D).
        store_params = build_params(npu_caches, cpu_caches, DIRECTION_D2H)
        load_params = build_params(cpu_caches, npu_caches, DIRECTION_H2D)
        copy_queue = queue.SimpleQueue()
        copy_thread = threading.Thread(
            target=self._copy_loop,
            name="npu-kv-offload-copy",
            daemon=True,
        )

        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("NPU KV offload copy backend is shut down")
            if self._thread is not None:
                raise RuntimeError("NPU KV offload copy backend is already initialized")
            self._load_stream = load_stream
            self._store_stream = store_stream
            self._device = device
            self._store_params = store_params
            self._load_params = load_params
            self._queue = copy_queue
            self._thread = copy_thread
            copy_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.npu.Event]],
        wait_event: torch.npu.Event | None = None,
    ) -> None:
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("NPU KV offload copy backend is shut down")
            self._raise_if_failed_locked()
            params = self._store_params if is_store else self._load_params
            copy_queue = self._queue
            if params is None or copy_queue is None:
                raise RuntimeError("NPU KV offload copy backend is not initialized")
            self._outstanding_jobs += 1
            try:
                copy_queue.put(
                    (
                        src_blocks,
                        dst_blocks,
                        params,
                        is_store,
                        event_idx,
                        events_list,
                        wait_event,
                    )
                )
            except Exception:
                self._outstanding_jobs -= 1
                self._idle_condition.notify_all()
                raise

    def check_health(self) -> None:
        with self._state_lock:
            self._raise_if_failed_locked()

    def drain(self, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Wait until every copy accepted before this barrier is observable.

        Completion events are appended by the background thread only after it
        has issued the copy and recorded the event. Waiting on the accepted-job
        count closes the interval where the parent's event lists are still
        empty even though a queued or in-flight transfer is accessing blocks.
        The backend remains usable after this non-terminal barrier.
        """
        with self._idle_condition:
            drained = self._idle_condition.wait_for(
                lambda: self._outstanding_jobs == 0 or self._background_error is not None,
                timeout=timeout,
            )
            self._raise_if_failed_locked()
            if not drained:
                raise RuntimeError(f"NPU KV offload copy backend did not drain within {timeout:.1f} seconds")

    def shutdown(self, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> None:
        with self._shutdown_lock:
            with self._state_lock:
                if self._shutdown_complete:
                    self._raise_if_failed_locked()
                    return
                if not self._shutdown:
                    self._shutdown = True
                    if self._queue is not None:
                        self._queue.put(None)
                copy_thread = self._thread
                streams = (self._load_stream, self._store_stream)

            if copy_thread is not None:
                copy_thread.join(timeout=timeout)
                if copy_thread.is_alive():
                    raise RuntimeError(f"NPU KV offload copy thread did not stop within {timeout:.1f} seconds")

            sync_error: Exception | None = None
            for stream in streams:
                if stream is None:
                    continue
                try:
                    stream.synchronize()
                except Exception as exc:
                    if sync_error is None:
                        sync_error = exc

            with self._state_lock:
                background_error = self._background_error
                if sync_error is None:
                    self._shutdown_complete = True

            if background_error is not None:
                raise RuntimeError("NPU KV offload copy thread failed") from background_error
            if sync_error is not None:
                raise RuntimeError("Failed to synchronize NPU KV offload streams") from sync_error

    # ------------------------------------------------------------------
    # Worker thread main loop
    # ------------------------------------------------------------------
    def _copy_loop(self) -> None:
        try:
            assert self._device is not None
            assert self._queue is not None
            assert self._load_stream is not None
            assert self._store_stream is not None
            torch.npu.set_device(self._device)

            while True:
                item = self._queue.get()
                if item is None:
                    return
                (
                    src_blocks,
                    dst_blocks,
                    params,
                    is_store,
                    event_idx,
                    events_list,
                    wait_event,
                ) = item

                try:
                    stream = self._store_stream if is_store else self._load_stream
                    with torch.npu.stream(stream):
                        if wait_event is not None:
                            stream.wait_event(wait_event)
                        copy_blocks(src_blocks, dst_blocks, params)
                        event = torch.npu.Event()
                        event.record(stream)
                    events_list.append((event_idx, event))
                finally:
                    with self._idle_condition:
                        self._outstanding_jobs -= 1
                        self._idle_condition.notify_all()
        except Exception as exc:
            with self._idle_condition:
                if self._background_error is None:
                    self._background_error = exc
                self._idle_condition.notify_all()

    def _raise_if_failed_locked(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("NPU KV offload copy thread failed") from self._background_error
