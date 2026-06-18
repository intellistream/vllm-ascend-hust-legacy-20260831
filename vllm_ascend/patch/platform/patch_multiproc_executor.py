from __future__ import annotations

import os
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing.synchronize import Lock as LockType

import vllm.v1.executor.multiproc_executor
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)
from vllm_ascend import envs as envs_ascend

logger = init_logger(__name__)

_WORKER_DEVICE_INDEX_ENV = "VLLM_ASCEND_WORKER_DEVICE_INDEX"
_WORKER_PHYSICAL_DEVICE_ENV = "VLLM_ASCEND_WORKER_PHYSICAL_DEVICE"
_VISIBLE_DEVICE_ENVS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
)


def _split_visible_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_worker_visible_devices() -> str:
    for env_name in _VISIBLE_DEVICE_ENVS:
        original_visible_devices = os.environ.get(f"VLLM_ASCEND_ORIGINAL_{env_name}", "")
        if original_visible_devices:
            return original_visible_devices

    for env_name in _VISIBLE_DEVICE_ENVS:
        visible_devices = os.environ.get(env_name, "")
        if visible_devices:
            return visible_devices
    return ""


def _select_worker_visible_device(local_rank: int) -> str | None:
    visible_devices = _get_worker_visible_devices()
    if not visible_devices:
        logger.warning(
            "VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES is enabled, but no "
            "Ascend visible-device environment variable is set."
        )
        return None

    devices = _split_visible_devices(visible_devices)
    if local_rank < 0 or local_rank >= len(devices):
        logger.warning(
            "VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES is enabled, but "
            "local_rank=%s is outside visible devices %s.",
            local_rank,
            visible_devices,
        )
        return None

    return devices[local_rank]


def _apply_worker_visible_device(local_rank: int) -> bool:
    selected_device = _select_worker_visible_device(local_rank)
    if selected_device is None:
        return False

    for env_name in _VISIBLE_DEVICE_ENVS:
        original_value = os.environ.get(env_name)
        if original_value is not None:
            os.environ.setdefault(f"VLLM_ASCEND_ORIGINAL_{env_name}", original_value)
        os.environ[env_name] = selected_device

    os.environ[_WORKER_DEVICE_INDEX_ENV] = "0"
    os.environ[_WORKER_PHYSICAL_DEVICE_ENV] = selected_device
    os.environ["VLLM_ASCEND_WORKER_ORIGINAL_LOCAL_RANK"] = str(local_rank)
    logger.info(
        "Narrowed worker visible Ascend devices to %s for local_rank=%s.",
        selected_device,
        local_rank,
    )
    return True


def _narrow_worker_visible_devices(local_rank: int) -> None:
    if not envs_ascend.VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES:
        return

    _apply_worker_visible_device(local_rank)


@contextmanager
def _worker_visible_devices_env(local_rank: int) -> Iterator[None]:
    if not envs_ascend.VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES:
        yield
        return

    selected_device = _select_worker_visible_device(local_rank)
    if selected_device is None:
        yield
        return

    updates = {
        _WORKER_DEVICE_INDEX_ENV: "0",
        _WORKER_PHYSICAL_DEVICE_ENV: selected_device,
        "VLLM_ASCEND_WORKER_ORIGINAL_LOCAL_RANK": str(local_rank),
    }
    for env_name in _VISIBLE_DEVICE_ENVS:
        original_value = os.environ.get(env_name)
        if original_value is not None:
            updates[f"VLLM_ASCEND_ORIGINAL_{env_name}"] = original_value
        updates[env_name] = selected_device

    old_values = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


class AscendMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tensor_parallel_size, pp_parallel_size, pcp_parallel_size = self._get_parallel_sizes()
        assert self.world_size == tensor_parallel_size * pp_parallel_size * pcp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}) x prefill_context"
            f"_parallel_size ({pcp_parallel_size}). "
        )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=self.parallel_config.master_addr,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp

            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = [] if context.get_start_method() == "fork" else None

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = AscendWorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = AscendWorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[rank]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()
            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()
            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0


class AscendWorkerProc(WorkerProc):
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool = False,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=AscendWorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,
        )

        with _worker_visible_devices_env(local_rank):
            proc.start()
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def worker_main(*args, **kwargs):
        _narrow_worker_visible_devices(int(kwargs.get("local_rank", 0)))
        return WorkerProc.worker_main(*args, **kwargs)


vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
