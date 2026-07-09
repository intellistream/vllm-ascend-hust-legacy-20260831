# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.metadata import (
    MetadataServer,
    MetadataServerProc,
    MLAConfig,
)
import vllm_ascend.envs as ascend_envs
from vllm_ascend.kv_offload.host_gather_stats import record_host_gather_event
from vllm_ascend.kv_offload.staging_pool import (
    StagingPoolConfig,
    WorkerLocalStagingPool,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from vllm.model_executor.layers.attention import Attention, MLAAttention


def _profile_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_CPU_OFFLOAD_PROFILE", "0").lower() in {"1", "true", "yes", "on"}


def _indexed_copy_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_CPU_OFFLOAD_INDEX_COPY", "0").lower() in {"1", "true", "yes", "on"}


def _reverse_span_copy_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_CPU_OFFLOAD_REVERSE_SPAN_COPY", "0").lower() in {"1", "true", "yes", "on"}


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


BlockCopySpan = tuple[int, int, int]
DirectionalBlockCopySpan = tuple[int, int, int, int]


def _block_pair_locality(block_mapping: Sequence[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]]:
    """Return CPU/GPU adjacent-pair counts and a small mapping sample."""
    cpu_adjacent = 0
    gpu_adjacent = 0
    for (prev_cpu_id, prev_gpu_id), (cpu_block_id, gpu_block_id) in zip(
        block_mapping,
        block_mapping[1:],
    ):
        if cpu_block_id == prev_cpu_id + 1:
            cpu_adjacent += 1
        if gpu_block_id == prev_gpu_id + 1:
            gpu_adjacent += 1
    return cpu_adjacent, gpu_adjacent, list(block_mapping[:8])


def _coalesce_block_copy_spans(block_mapping: Sequence[tuple[int, int]]) -> list[BlockCopySpan]:
    """Coalesce adjacent CPU/GPU block pairs into contiguous copy spans."""
    if not block_mapping:
        return []

    spans: list[BlockCopySpan] = []
    cpu_start, gpu_start = block_mapping[0]
    prev_cpu_id, prev_gpu_id = cpu_start, gpu_start
    span_len = 1
    for cpu_block_id, gpu_block_id in block_mapping[1:]:
        if cpu_block_id == prev_cpu_id + 1 and gpu_block_id == prev_gpu_id + 1:
            span_len += 1
        else:
            spans.append((cpu_start, gpu_start, span_len))
            cpu_start, gpu_start = cpu_block_id, gpu_block_id
            span_len = 1
        prev_cpu_id, prev_gpu_id = cpu_block_id, gpu_block_id
    spans.append((cpu_start, gpu_start, span_len))
    return spans


def _coalesce_directional_block_copy_spans(
    block_mapping: Sequence[tuple[int, int]],
) -> list[DirectionalBlockCopySpan]:
    """Coalesce spans where CPU is ascending and GPU is ascending or descending."""
    if not block_mapping:
        return []

    spans: list[DirectionalBlockCopySpan] = []
    cpu_start, gpu_start = block_mapping[0]
    prev_cpu_id, prev_gpu_id = cpu_start, gpu_start
    span_len = 1
    direction = 0
    for cpu_block_id, gpu_block_id in block_mapping[1:]:
        next_direction = gpu_block_id - prev_gpu_id
        if cpu_block_id == prev_cpu_id + 1 and next_direction in (1, -1) and direction in (0, next_direction):
            span_len += 1
            direction = next_direction
        else:
            spans.append((cpu_start, gpu_start, span_len, direction or 1))
            cpu_start, gpu_start = cpu_block_id, gpu_block_id
            span_len = 1
            direction = 0
        prev_cpu_id, prev_gpu_id = cpu_block_id, gpu_block_id
    spans.append((cpu_start, gpu_start, span_len, direction or 1))
    return spans


@dataclass
class ReqMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_gpu_computed_tokens: int
    num_cpu_computed_tokens: int

    def update(self, other: "ReqMeta"):
        self.gpu_block_ids.extend(other.gpu_block_ids)
        self.cpu_block_ids.extend(other.cpu_block_ids)
        self.num_scheduled_tokens = other.num_scheduled_tokens
        self.num_computed_tokens = other.num_computed_tokens
        self.num_gpu_computed_tokens = other.num_gpu_computed_tokens
        # For scheduled_cached_reqs the scheduler reports num_computed_tokens
        # for both GPU and CPU fields. Keep the original CPU-hit prefix here:
        # it is the lower bound of blocks already materialized in CPU memory,
        # and save must write every later full block.
        self.num_cpu_computed_tokens = min(
            self.num_cpu_computed_tokens,
            other.num_cpu_computed_tokens,
        )


@dataclass
class CPUOffloadingConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, ReqMeta]
    finished_req_ids: set[str]


class CPUOffloadingConnector(KVConnectorBase_V1):
    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: Optional["KVCacheConfig"] = None
    ):
        self._connector_metadata = CPUOffloadingConnectorMetadata(requests={}, finished_req_ids=set())
        if not vllm_config.cache_config.enable_prefix_caching:
            self.connector_scheduler: CPUOffloadingConnectorScheduler | None = None
            self.connector_worker: CPUOffloadingConnectorWorker | None = None
        elif role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = CPUOffloadingConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = CPUOffloadingConnectorWorker(vllm_config)

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.connector_worker is not None:
            assert isinstance(connector_metadata, CPUOffloadingConnectorMetadata)
            self.connector_worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.clear_connector_metadata()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is not None:
            self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if self.connector_worker is not None:
            self.connector_worker.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.connector_worker is not None:
            self.connector_worker.wait_for_layer_load()

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(), None

    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if self.connector_scheduler is not None:
            return self.connector_scheduler.update_state_after_alloc(request)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.build_connector_meta(scheduler_output)
        return KVConnectorMetadata()

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        if self.connector_scheduler is not None:
            self.connector_scheduler.request_finished(request)
        return True, None

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


class CPUOffloadingConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig):
        logger.info("init CPUOffloadingConnectorScheduler")
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.use_mla = vllm_config.model_config.use_mla
        self.num_gpu_computed_tokens: dict[str, int] = {}
        self.num_cpu_computed_tokens: dict[str, int] = {}
        self.allocated_req_ids: set[str] = set()
        self.finished_req_ids: list[str] = []
        self.zmq_rpc_client = MetadataServer.ZMQRPCClient()
        self.zmq_rpc_client.call("post_init")
        if vllm_config.kv_transfer_config is not None:
            self.swap_in_threshold = vllm_config.kv_transfer_config.get_from_extra_config("swap_in_threshold", 0)
        else:
            self.swap_in_threshold = 0
        logger.info("swap_in_threshold: %s", self.swap_in_threshold)

    def get_num_new_matched_tokens(self, ori_request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        started = time.perf_counter()
        request = copy.deepcopy(ori_request)
        request.get_hash_new_full_blocks = None
        request._block_hasher = None
        num_cpu_computed_tokens, load_async = self.zmq_rpc_client.call("get_matched_num_and_touch", request)
        self.num_gpu_computed_tokens[request.request_id] = num_computed_tokens
        self.num_cpu_computed_tokens[request.request_id] = num_cpu_computed_tokens
        if _profile_enabled():
            logger.info(
                "[cpu-offload-profile] scheduler_match req=%s gpu_tokens=%s cpu_tokens=%s delta=%s wall_ms=%.3f",
                request.request_id,
                num_computed_tokens,
                num_cpu_computed_tokens,
                num_cpu_computed_tokens - num_computed_tokens,
                _elapsed_ms(started),
            )
        if num_cpu_computed_tokens - num_computed_tokens >= self.swap_in_threshold:
            return num_cpu_computed_tokens - num_computed_tokens, load_async
        else:
            return 0, load_async

    def update_state_after_alloc(self, request: "Request"):
        self.allocated_req_ids.add(request.request_id)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        started = time.perf_counter()
        num_tokens = {}
        # process scheduled_new_reqs
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            num_tokens[req_id] = req.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]

        # process scheduled_cached_reqs
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            num_tokens[req_id] = cached_reqs.num_computed_tokens[idx] + scheduler_output.num_scheduled_tokens[req_id]

        unallocated_req_ids = set(
            self.num_gpu_computed_tokens.keys() - self.allocated_req_ids - scheduler_output.num_scheduled_tokens.keys()
        )
        new_cpu_block_ids = self.zmq_rpc_client.call("allocate_slots", num_tokens, unallocated_req_ids)
        metadata = CPUOffloadingConnectorMetadata(
            requests={},
            finished_req_ids=set(self.finished_req_ids),
        )
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            gpu_block_ids = req.block_ids[0]
            metadata.requests[req_id] = ReqMeta(
                gpu_block_ids=[] if gpu_block_ids is None else gpu_block_ids,
                cpu_block_ids=new_cpu_block_ids.get(req_id, []),
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens[req_id],
                num_computed_tokens=req.num_computed_tokens,
                num_gpu_computed_tokens=self.num_gpu_computed_tokens[req_id],
                num_cpu_computed_tokens=self.num_cpu_computed_tokens[req_id],
            )

        for idx, req_id in enumerate(cached_reqs.req_ids):
            gpu_block_ids = cached_reqs.new_block_ids[idx]
            metadata.requests[req_id] = ReqMeta(
                gpu_block_ids=[] if gpu_block_ids is None else gpu_block_ids,
                cpu_block_ids=new_cpu_block_ids.get(req_id, []),
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens[req_id],
                num_computed_tokens=cached_reqs.num_computed_tokens[idx],
                num_gpu_computed_tokens=cached_reqs.num_computed_tokens[idx],
                num_cpu_computed_tokens=cached_reqs.num_computed_tokens[idx],
            )
        self.num_gpu_computed_tokens.clear()
        self.num_cpu_computed_tokens.clear()
        self.allocated_req_ids.clear()
        self.finished_req_ids.clear()
        if _profile_enabled():
            block_count = sum(len(req.cpu_block_ids) for req in metadata.requests.values())
            logger.info(
                "[cpu-offload-profile] scheduler_build_meta requests=%d finished=%d cpu_blocks=%d wall_ms=%.3f",
                len(metadata.requests),
                len(metadata.finished_req_ids),
                block_count,
                _elapsed_ms(started),
            )
        return metadata

    def request_finished(self, ori_request: "Request"):
        started = time.perf_counter()
        request = copy.deepcopy(ori_request)
        request.get_hash_new_full_blocks = None
        request._block_hasher = None
        self.finished_req_ids.append(request.request_id)
        # inform metadata server to record request, and free it after finish sending
        self.zmq_rpc_client.call("record_request_cache_and_free_slots", request)
        if _profile_enabled():
            logger.info(
                "[cpu-offload-profile] scheduler_request_finished req=%s wall_ms=%.3f",
                request.request_id,
                _elapsed_ms(started),
            )


class CPUOffloadingConnectorWorker:
    def __init__(self, vllm_config: VllmConfig):
        logger.info("init CPUOffloadingConnectorWorker")
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.pp_rank = get_pp_group().rank_in_group
        self.tp_group = get_tp_group()
        self.tp_rank = self.tp_group.rank_in_group
        self.tp_world_size = self.tp_group.world_size
        self.use_mla = vllm_config.model_config.use_mla

        self.requests: dict[str, ReqMeta] = {}
        self.load_stream = torch.npu.Stream()
        self.zmq_rpc_client = MetadataServer.ZMQRPCClient()
        self.load_block_mapping: list[tuple[int, int]] = []
        self.load_block_spans: list[BlockCopySpan] = []
        self.load_directional_block_spans: list[DirectionalBlockCopySpan] | None = None
        self.load_cpu_adjacent_pairs = 0
        self.load_gpu_adjacent_pairs = 0
        self.load_pair_sample: list[tuple[int, int]] = []
        self.save_input_queue: queue.Queue[tuple[str, ReqMeta]] = queue.Queue()
        self.save_output_queue: queue.Queue[str] = queue.Queue()
        self.done_sending_count: defaultdict[str, int] = defaultdict(int)
        self.use_indexed_copy = _indexed_copy_enabled()
        self.use_reverse_span_copy = _reverse_span_copy_enabled()
        self.use_host_gather = ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER
        self.use_staging_pool = ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_POOL
        self.staging_pool_config = StagingPoolConfig(
            slots=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_SLOTS,
            slab_blocks=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_SLAB_BLOCKS,
            pack_backend=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_PACK_BACKEND,
            pack_threads=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_PACK_THREADS,
            fused_kv=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_PACK_FUSED_KV,
            build_dir=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_PACK_BUILD_DIR,
            fallback_on_error=ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_FALLBACK_ON_ERROR,
        )
        self.profile_timing = ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_PROFILE_TIMING
        self.real_path_trace = ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_REAL_PATH_TRACE
        self.host_gather_op: Any | None = None
        self.staging_pool: WorkerLocalStagingPool | None = None
        self._host_gather_fallback_logged = False
        self._staging_pool_fallback_logged = False
        self._pending_load_profiles: list[dict[str, Any]] = []
        self._staging_src_ids_cpu: torch.Tensor | None = None
        self._staging_src_ids_npu: torch.Tensor | None = None
        self._staging_dst_ids_npu: torch.Tensor | None = None
        self._shutdown_done = False
        if self.use_host_gather or self.use_staging_pool:
            host_gather_lib = ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_LIB
            if host_gather_lib:
                torch.ops.load_library(host_gather_lib)
            else:
                try:
                    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
                except Exception as exc:
                    logger.warning(
                        "CPU offload host gather/staging is enabled, but "
                        "vllm_ascend_C could not be imported before checking "
                        "kv_cache_block_gather: %s",
                        exc,
                    )
            try:
                self.host_gather_op = torch.ops._C_ascend.kv_cache_block_gather
            except AttributeError:
                logger.warning(
                    "CPU offload host gather/staging is enabled, but "
                    "torch.ops._C_ascend.kv_cache_block_gather is not available. "
                    "Falling back to main span-copy."
                )
                self.use_host_gather = False
                self.use_staging_pool = False
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "init",
            host_gather_enabled=bool(ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER),
            host_gather_active=bool(self.use_host_gather and self.host_gather_op is not None),
            staging_pool_enabled=bool(ascend_envs.VLLM_ASCEND_CPU_OFFLOAD_STAGING_POOL),
            staging_pool_active=bool(self.use_staging_pool and self.host_gather_op is not None),
            staging_pool_slots=int(self.staging_pool_config.slots),
            staging_pool_slab_blocks=int(self.staging_pool_config.slab_blocks),
            staging_pack_backend=str(self.staging_pool_config.pack_backend),
            staging_pack_threads=int(self.staging_pool_config.pack_threads),
            staging_pack_fused_kv=bool(self.staging_pool_config.fused_kv),
            profile_timing=bool(self.profile_timing),
            real_path_trace=bool(self.real_path_trace),
            pp_rank=int(self.pp_rank),
            tp_rank=int(self.tp_rank),
            tp_world_size=int(self.tp_world_size),
            use_mla=bool(self.use_mla),
        )

        # start metadata server to init cpu_kv_cache_manager and handle rpc requests
        # all dp shared the same metadata server, only start the process on data_rank 0
        if vllm_config.parallel_config.data_parallel_rank == 0 and self.tp_rank == 0 and self.pp_rank == 0:
            config = VllmConfig()
            config.cache_config = vllm_config.cache_config
            config.model_config = vllm_config.model_config
            config.parallel_config = vllm_config.parallel_config
            config.scheduler_config = vllm_config.scheduler_config
            config.kv_transfer_config = vllm_config.kv_transfer_config
            self.init_metadata_server(config)
        self._wait_for_metadata_process_start()

    def init_metadata_server(self, vllm_config: VllmConfig):
        self.metadata_thread = threading.Thread(
            target=MetadataServerProc.run_metadata_server,
            args=(vllm_config,),
        )
        self.metadata_thread.daemon = True
        self.metadata_thread.start()

    def _wait_for_metadata_process_start(self):
        # TODO: wait for metadata server to start, add a rpc to check if ready
        while True:
            try:
                if self.zmq_rpc_client.call("ready"):
                    break
            except Exception as e:
                logger.info("wait for metadata server to start, error: %s", e)
                time.sleep(1)

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "shutdown_start",
            has_staging_pool=bool(self.staging_pool is not None),
        )
        if self.staging_pool is not None:
            try:
                self.staging_pool.close()
            except Exception:
                logger.debug("failed to close CPU offload staging pool", exc_info=True)
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "shutdown_done",
            has_staging_pool=bool(self.staging_pool is not None),
        )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def bind_connector_metadata(self, connector_metadata: CPUOffloadingConnectorMetadata) -> None:
        started = time.perf_counter()
        added_load_blocks = 0
        save_reqs = 0
        for req_id, req in connector_metadata.requests.items():
            if req_id in self.requests:
                self.requests[req_id].update(req)
                req = self.requests[req_id]
            else:
                self.requests[req_id] = req
            for i in range(req.num_gpu_computed_tokens // self.block_size, req.num_computed_tokens // self.block_size):
                self.load_block_mapping.append((req.cpu_block_ids[i], req.gpu_block_ids[i]))
                added_load_blocks += 1
        self._refresh_load_plan()
        for req_id in connector_metadata.finished_req_ids:
            if req_id in self.requests:
                self.save_input_queue.put((req_id, self.requests[req_id]))
                save_reqs += 1
        if _profile_enabled():
            logger.info(
                "[cpu-offload-profile] worker_bind requests=%d finished=%d load_blocks_added=%d pending_load_blocks=%d save_reqs=%d wall_ms=%.3f",
                len(connector_metadata.requests),
                len(connector_metadata.finished_req_ids),
                added_load_blocks,
                len(self.load_block_mapping),
                save_reqs,
                _elapsed_ms(started),
            )

    def clear_connector_metadata(self) -> None:
        self.load_block_mapping.clear()
        self._refresh_load_plan()

    def _refresh_load_plan(self) -> None:
        self.load_block_spans = _coalesce_block_copy_spans(self.load_block_mapping)
        self.load_directional_block_spans = (
            _coalesce_directional_block_copy_spans(self.load_block_mapping)
            if self.use_reverse_span_copy
            else None
        )
        (
            self.load_cpu_adjacent_pairs,
            self.load_gpu_adjacent_pairs,
            self.load_pair_sample,
        ) = _block_pair_locality(self.load_block_mapping)

    def _log_host_gather_fallback_once(self, reason: str) -> None:
        record_host_gather_event("cpu_offload_connector_worker", "fallback", reason=reason)
        if not self._host_gather_fallback_logged:
            logger.info("CPU offload mapped-host gather fallback: %s", reason)
            self._host_gather_fallback_logged = True

    def _log_staging_pool_fallback_once(self, reason: str) -> None:
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "staging_pool_fallback",
            reason=reason,
        )
        if not self._staging_pool_fallback_logged:
            logger.info("CPU offload staging pool fallback: %s", reason)
            self._staging_pool_fallback_logged = True

    def _load_mapping_profile(self) -> dict[str, Any]:
        if not self.load_block_mapping:
            return {
                "block_pairs": 0,
                "src_min": None,
                "src_max": None,
                "src_span_blocks": 0,
                "source_span_ratio": None,
                "source_run_count": 0,
                "source_dense": False,
                "source_sparse": False,
                "dst_min": None,
                "dst_max": None,
                "dst_span_blocks": 0,
                "dst_run_count": 0,
                "dst_dense": False,
            }

        def count_runs(values: list[int]) -> int:
            if not values:
                return 0
            runs = 1
            prev = values[0]
            for value in values[1:]:
                if value != prev + 1:
                    runs += 1
                prev = value
            return runs

        src_ids = [cpu_block_id for cpu_block_id, _ in self.load_block_mapping]
        dst_ids = [gpu_block_id for _, gpu_block_id in self.load_block_mapping]
        src_unique = sorted(set(src_ids))
        dst_unique = sorted(set(dst_ids))
        block_pairs = len(self.load_block_mapping)
        src_min = min(src_ids)
        src_max = max(src_ids)
        dst_min = min(dst_ids)
        dst_max = max(dst_ids)
        src_span_blocks = src_max - src_min + 1
        dst_span_blocks = dst_max - dst_min + 1
        source_span_ratio = float(src_span_blocks) / float(block_pairs) if block_pairs else None
        source_run_count = count_runs(src_unique)
        dst_run_count = count_runs(dst_unique)
        source_dense = len(src_unique) == src_span_blocks and len(src_unique) == block_pairs
        dst_dense = len(dst_unique) == dst_span_blocks and len(dst_unique) == block_pairs
        return {
            "block_pairs": int(block_pairs),
            "src_min": int(src_min),
            "src_max": int(src_max),
            "src_span_blocks": int(src_span_blocks),
            "source_span_ratio": source_span_ratio,
            "source_run_count": int(source_run_count),
            "source_dense": bool(source_dense),
            "source_sparse": bool(source_run_count > 1 or not source_dense),
            "dst_min": int(dst_min),
            "dst_max": int(dst_max),
            "dst_span_blocks": int(dst_span_blocks),
            "dst_run_count": int(dst_run_count),
            "dst_dense": bool(dst_dense),
        }

    def _can_gather_kv_part(
        self,
        cpu_layer_part: torch.Tensor,
        gpu_layer_part: torch.Tensor,
        *,
        log_fallback: Callable[[str], None],
    ) -> bool:
        if self.host_gather_op is None:
            log_fallback("mapped-host gather op is not available")
            return False
        if not cpu_layer_part.is_contiguous() or not gpu_layer_part.is_contiguous():
            log_fallback("cpu/gpu KV cache part is not contiguous")
            return False
        if cpu_layer_part.dtype != gpu_layer_part.dtype:
            log_fallback("cpu/gpu KV cache dtype mismatch")
            return False
        if cpu_layer_part.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            log_fallback(f"unsupported KV cache dtype {cpu_layer_part.dtype}")
            return False
        return True

    def _make_host_gather_indices(self, device: torch.device) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        src_min = min(cpu_block_id for cpu_block_id, _ in self.load_block_mapping)
        src_max = max(cpu_block_id for cpu_block_id, _ in self.load_block_mapping)
        src_block_ids = torch.tensor(
            [cpu_block_id - src_min for cpu_block_id, _ in self.load_block_mapping],
            dtype=torch.int32,
            device=device,
        )
        dst_block_ids = torch.tensor(
            [gpu_block_id for _, gpu_block_id in self.load_block_mapping],
            dtype=torch.int32,
            device=device,
        )
        return src_min, src_max, src_block_ids, dst_block_ids

    def _make_staging_indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_pairs = len(self.load_block_mapping)
        if self._staging_src_ids_cpu is None or int(self._staging_src_ids_cpu.numel()) != block_pairs:
            self._staging_src_ids_cpu = torch.tensor(
                [cpu_block_id for cpu_block_id, _ in self.load_block_mapping],
                dtype=torch.long,
                device="cpu",
            )
        if (
            self._staging_src_ids_npu is None
            or int(self._staging_src_ids_npu.numel()) != block_pairs
            or self._staging_src_ids_npu.device != device
        ):
            self._staging_src_ids_npu = torch.arange(block_pairs, dtype=torch.int32, device=device)
        if (
            self._staging_dst_ids_npu is None
            or int(self._staging_dst_ids_npu.numel()) != block_pairs
            or self._staging_dst_ids_npu.device != device
        ):
            self._staging_dst_ids_npu = torch.tensor(
                [gpu_block_id for _, gpu_block_id in self.load_block_mapping],
                dtype=torch.int32,
                device=device,
            )
        return self._staging_src_ids_cpu, self._staging_src_ids_npu, self._staging_dst_ids_npu

    def _ensure_staging_pool(self, cpu_parts: Sequence[torch.Tensor]) -> WorkerLocalStagingPool | None:
        if not self.use_staging_pool or self.host_gather_op is None:
            return None
        if self.staging_pool is not None:
            return self.staging_pool
        part_shapes = [list(part.shape) for part in cpu_parts]
        if not part_shapes:
            self._log_staging_pool_fallback_once("no CPU KV parts available")
            return None
        dtype = cpu_parts[0].dtype
        try:
            self.staging_pool = WorkerLocalStagingPool(
                config=self.staging_pool_config,
                part_shapes=part_shapes,
                dtype=dtype,
            )
            return self.staging_pool
        except Exception as exc:
            self._log_staging_pool_fallback_once(
                f"failed to initialize staging pool: {type(exc).__name__}: {exc}"
            )
            if not self.staging_pool_config.fallback_on_error:
                raise
            self.use_staging_pool = False
            return None

    def _can_use_staging_pool(
        self,
        cpu_parts: Sequence[torch.Tensor],
        gpu_parts: Sequence[torch.Tensor],
    ) -> bool:
        if not self.use_staging_pool or self.host_gather_op is None:
            return False
        if not self.load_block_mapping:
            return False
        if len(cpu_parts) != len(gpu_parts):
            self._log_staging_pool_fallback_once("CPU/GPU KV part count mismatch")
            return False
        if len(self.load_block_mapping) > int(self.staging_pool_config.slab_blocks):
            self._log_staging_pool_fallback_once("restore block count exceeds staging slab blocks")
            return False
        for cpu_part, gpu_part in zip(cpu_parts, gpu_parts):
            if not self._can_gather_kv_part(
                cpu_part,
                gpu_part,
                log_fallback=self._log_staging_pool_fallback_once,
            ):
                return False
        return True

    def _flush_pending_load_profiles(self, layer: int | None = None) -> None:
        remaining_profiles: list[dict[str, Any]] = []
        for profile in self._pending_load_profiles:
            if layer is not None and profile.get("layer") != layer:
                remaining_profiles.append(profile)
                continue
            start_event = profile.pop("start_event")
            end_event = profile.pop("end_event")
            try:
                device_elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception as exc:
                device_elapsed_ms = -1.0
                profile["timing_error"] = str(exc)
            record_host_gather_event(
                "cpu_offload_connector_worker",
                "load_part_timing",
                **profile,
                device_elapsed_ms=device_elapsed_ms,
            )
        self._pending_load_profiles = remaining_profiles

    def _load_kv_layer_staging_pool(
        self,
        *,
        layer: int,
        cpu_parts: Sequence[torch.Tensor],
        gpu_parts: Sequence[torch.Tensor],
        mapping_profile: dict[str, Any],
    ) -> bool:
        if not self._can_use_staging_pool(cpu_parts, gpu_parts):
            return False
        pool = self._ensure_staging_pool(cpu_parts)
        if pool is None:
            return False

        index_start_s = time.perf_counter()
        src_ids_cpu, staging_src_ids, dst_block_ids = self._make_staging_indices(gpu_parts[0].device)
        index_wall_ms = (time.perf_counter() - index_start_s) * 1000.0
        if self.profile_timing or self.real_path_trace:
            record_host_gather_event(
                "cpu_offload_connector_worker",
                "make_staging_indices_timing",
                layer=int(layer),
                wall_ms=float(index_wall_ms),
                **mapping_profile,
            )

        try:
            slot, slot_wait_ms = pool.acquire()
            pack_wall_ms = pool.pack(
                layer=layer,
                slot=slot,
                src_parts=cpu_parts,
                src_ids=src_ids_cpu,
            )
        except Exception as exc:
            self._log_staging_pool_fallback_once(f"staging pack failed: {type(exc).__name__}: {exc}")
            if not self.staging_pool_config.fallback_on_error:
                raise
            return False

        block_pairs = len(self.load_block_mapping)
        for staging_part, gpu_part in zip(slot.parts, gpu_parts):
            if not self._can_gather_kv_part(
                staging_part[:block_pairs],
                gpu_part,
                log_fallback=self._log_staging_pool_fallback_once,
            ):
                if not self.staging_pool_config.fallback_on_error:
                    raise RuntimeError("staging gather capability check failed")
                return False

        done_event = torch.npu.Event()
        gathered = 0
        for part_index, (staging_part, gpu_part) in enumerate(zip(slot.parts, gpu_parts)):
            staging_span = staging_part[:block_pairs]
            record_host_gather_event(
                "cpu_offload_connector_worker",
                "mapped_gather_call",
                mode="staging_pool",
                layer=int(layer),
                part_index=int(part_index),
                dtype=str(staging_span.dtype),
                src_shape=list(staging_span.shape),
                dst_shape=list(gpu_part.shape),
                staging_slot=int(slot.index),
                staging_pack_wall_ms=float(pack_wall_ms),
                staging_slot_wait_ms=float(slot_wait_ms),
                **mapping_profile,
            )
            start_event: torch.npu.Event | None = None
            end_event: torch.npu.Event | None = None
            if self.profile_timing:
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                start_event.record(self.load_stream)
            enqueue_start_s = time.perf_counter()
            self.host_gather_op(staging_src_ids, staging_span, dst_block_ids, gpu_part)
            enqueue_wall_ms = (time.perf_counter() - enqueue_start_s) * 1000.0
            gathered += 1
            if self.profile_timing and start_event is not None and end_event is not None:
                end_event.record(self.load_stream)
                block_bytes = int(staging_span.stride(0) * staging_span.element_size())
                self._pending_load_profiles.append(
                    {
                        "mode": "staging_pool",
                        "copy_reason": "",
                        "layer": int(layer),
                        "part_index": int(part_index),
                        "logical_bytes": int(block_pairs * block_bytes),
                        "source_span_bytes": int(block_pairs * block_bytes),
                        "block_bytes": block_bytes,
                        "enqueue_wall_ms": float(enqueue_wall_ms),
                        "staging_pack_wall_ms": float(pack_wall_ms),
                        "staging_slot_wait_ms": float(slot_wait_ms),
                        **mapping_profile,
                        "start_event": start_event,
                        "end_event": end_event,
                    }
                )

        done_event.record(self.load_stream)
        pool.mark_inflight(slot, done_event)
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "staging_pool_layer_enqueue_done",
            layer=int(layer),
            slot=int(slot.index),
            gathered_part_count=int(gathered),
            slot_wait_ms=float(slot_wait_ms),
            pack_wall_ms=float(pack_wall_ms),
            **mapping_profile,
        )
        return True

    def _load_kv_layer_host_gather(
        self,
        *,
        layer: int,
        cpu_parts: Sequence[torch.Tensor],
        gpu_parts: Sequence[torch.Tensor],
        mapping_profile: dict[str, Any],
    ) -> bool:
        if not self.use_host_gather or self.host_gather_op is None:
            return False
        if len(cpu_parts) != len(gpu_parts):
            self._log_host_gather_fallback_once("CPU/GPU KV part count mismatch")
            return False
        index_start_s = time.perf_counter()
        src_min, src_max, src_block_ids, dst_block_ids = self._make_host_gather_indices(gpu_parts[0].device)
        index_wall_ms = (time.perf_counter() - index_start_s) * 1000.0
        if self.profile_timing or self.real_path_trace:
            record_host_gather_event(
                "cpu_offload_connector_worker",
                "make_host_gather_indices_timing",
                layer=int(layer),
                wall_ms=float(index_wall_ms),
                **mapping_profile,
            )
        cpu_spans = [cpu_part[src_min : src_max + 1] for cpu_part in cpu_parts]
        for cpu_span, gpu_part in zip(cpu_spans, gpu_parts):
            if not self._can_gather_kv_part(
                cpu_span,
                gpu_part,
                log_fallback=self._log_host_gather_fallback_once,
            ):
                return False

        for part_index, (cpu_span, gpu_part) in enumerate(zip(cpu_spans, gpu_parts)):
            record_host_gather_event(
                "cpu_offload_connector_worker",
                "mapped_gather_call",
                mode="mapped",
                layer=int(layer),
                part_index=int(part_index),
                dtype=str(cpu_span.dtype),
                src_shape=list(cpu_span.shape),
                dst_shape=list(gpu_part.shape),
                **mapping_profile,
            )
            start_event: torch.npu.Event | None = None
            end_event: torch.npu.Event | None = None
            if self.profile_timing:
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                start_event.record(self.load_stream)
            enqueue_start_s = time.perf_counter()
            self.host_gather_op(src_block_ids, cpu_span, dst_block_ids, gpu_part)
            enqueue_wall_ms = (time.perf_counter() - enqueue_start_s) * 1000.0
            if self.profile_timing and start_event is not None and end_event is not None:
                end_event.record(self.load_stream)
                block_bytes = int(cpu_span.stride(0) * cpu_span.element_size())
                self._pending_load_profiles.append(
                    {
                        "mode": "mapped",
                        "copy_reason": "",
                        "layer": int(layer),
                        "part_index": int(part_index),
                        "logical_bytes": int(len(self.load_block_mapping) * block_bytes),
                        "source_span_bytes": int((src_max - src_min + 1) * block_bytes),
                        "block_bytes": block_bytes,
                        "enqueue_wall_ms": float(enqueue_wall_ms),
                        **mapping_profile,
                        "start_event": start_event,
                        "end_event": end_event,
                    }
                )
        return True

    def register_kv_caches(self, kv_caches: dict[str, Sequence[torch.Tensor]]):
        self.gpu_kv_caches = kv_caches
        model_config = self.vllm_config.model_config
        mla_config: MLAConfig | None = None
        if model_config.use_mla:
            mla_config = MLAConfig(
                model_config.hf_text_config.kv_lora_rank, model_config.hf_text_config.qk_rope_head_dim
            )
        self.cpu_kv_caches = list(
            self.zmq_rpc_client.call(
                "init_cpu_kv_caches",
                self.pp_rank,
                self.tp_rank,
                get_kv_cache_spec(self.vllm_config),
                mla_config,
            ).values()
        )

    def start_load_kv(self) -> None:
        self.current_layer = 0
        self.gpu_kv_caches_load_iter = iter(self.gpu_kv_caches.values())
        self._staging_src_ids_cpu = None
        self._staging_src_ids_npu = None
        self._staging_dst_ids_npu = None
        record_host_gather_event(
            "cpu_offload_connector_worker",
            "start_load_kv",
            load_mapping_count=int(len(self.load_block_mapping)),
            **self._load_mapping_profile(),
        )
        self.load_kv_layer(0)

    def wait_for_layer_load(self) -> None:
        if not self.load_block_mapping:
            self.current_layer += 1
            return
        # TODO: Replace with `torch.npu.current_stream().wait_stream(self.load_stream)` after fixing the bug.
        started = time.perf_counter()
        self.load_stream.synchronize()
        if _profile_enabled() and self.load_block_mapping:
            logger.info(
                "[cpu-offload-profile] worker_wait_layer_load layer=%d load_blocks=%d sync_ms=%.3f",
                self.current_layer,
                len(self.load_block_mapping),
                _elapsed_ms(started),
            )
        if self.profile_timing:
            self._flush_pending_load_profiles(layer=self.current_layer)
        self.current_layer += 1
        self.load_kv_layer(self.current_layer)

    def load_kv_layer(self, layer: int):
        if layer == len(self.gpu_kv_caches):
            return
        if not self.load_block_mapping:
            return
        started = time.perf_counter()
        gpu_kv_caches = next(self.gpu_kv_caches_load_iter)
        cpu_kv_caches = self.cpu_kv_caches[layer]
        layer_parts = list(zip(gpu_kv_caches, cpu_kv_caches))
        gpu_parts = tuple(gpu_layer_part for gpu_layer_part, _ in layer_parts)
        cpu_parts = tuple(cpu_layer_part for _, cpu_layer_part in layer_parts)
        mapping_profile = self._load_mapping_profile()
        block_spans = self.load_block_spans
        directional_block_spans = self.load_directional_block_spans
        original_copy_ops = len(self.load_block_mapping) * len(layer_parts)
        contiguous_blocks = 0
        reverse_blocks = 0
        fallback_blocks = 0
        copy_ops = 0
        copy_mode = "span"
        with torch.npu.stream(self.load_stream):
            if self.real_path_trace:
                record_host_gather_event(
                    "cpu_offload_connector_worker",
                    "load_layer_enqueue_start",
                    layer=int(layer),
                    mode=(
                        "staging_pool"
                        if self.use_staging_pool and self.host_gather_op is not None
                        else "mapped"
                        if self.use_host_gather and self.host_gather_op is not None
                        else "copy"
                    ),
                    **mapping_profile,
                )
            if self.use_staging_pool and self.host_gather_op is not None:
                staging_loaded = self._load_kv_layer_staging_pool(
                    layer=layer,
                    cpu_parts=cpu_parts,
                    gpu_parts=gpu_parts,
                    mapping_profile=mapping_profile,
                )
                if staging_loaded:
                    if self.real_path_trace:
                        record_host_gather_event(
                            "cpu_offload_connector_worker",
                            "load_layer_enqueue_end",
                            layer=int(layer),
                            mode="staging_pool",
                            **mapping_profile,
                        )
                    if _profile_enabled():
                        logger.info(
                            "[cpu-offload-profile] worker_schedule_layer_load layer=%d mode=staging_pool "
                            "load_blocks=%d spans=%d schedule_ms=%.3f",
                            layer,
                            len(self.load_block_mapping),
                            len(block_spans),
                            _elapsed_ms(started),
                        )
                    return
            if self.use_host_gather and self.host_gather_op is not None:
                mapped_loaded = self._load_kv_layer_host_gather(
                    layer=layer,
                    cpu_parts=cpu_parts,
                    gpu_parts=gpu_parts,
                    mapping_profile=mapping_profile,
                )
                if mapped_loaded:
                    if self.real_path_trace:
                        record_host_gather_event(
                            "cpu_offload_connector_worker",
                            "load_layer_enqueue_end",
                            layer=int(layer),
                            mode="mapped",
                            **mapping_profile,
                        )
                    if _profile_enabled():
                        logger.info(
                            "[cpu-offload-profile] worker_schedule_layer_load layer=%d mode=mapped "
                            "load_blocks=%d spans=%d schedule_ms=%.3f",
                            layer,
                            len(self.load_block_mapping),
                            len(block_spans),
                            _elapsed_ms(started),
                        )
                    return
            copy_start_event: torch.npu.Event | None = None
            copy_end_event: torch.npu.Event | None = None
            if self.profile_timing:
                copy_start_event = torch.npu.Event(enable_timing=True)
                copy_end_event = torch.npu.Event(enable_timing=True)
                copy_start_event.record(self.load_stream)
            copy_enqueue_start_s = time.perf_counter()
            if self.use_indexed_copy and len(block_spans) > 1:
                try:
                    cpu_indices = torch.tensor(
                        [cpu_block_id for cpu_block_id, _ in self.load_block_mapping],
                        dtype=torch.long,
                        device=cpu_kv_caches[0].device,
                    )
                    for gpu_layer_part, cpu_layer_part in layer_parts:
                        gpu_indices = torch.tensor(
                            [gpu_block_id for _, gpu_block_id in self.load_block_mapping],
                            dtype=torch.long,
                            device=gpu_layer_part.device,
                        )
                        packed = cpu_layer_part.index_select(0, cpu_indices).to(
                            gpu_layer_part.device,
                            non_blocking=True,
                        )
                        gpu_layer_part.index_copy_(0, gpu_indices, packed)
                        copy_ops += 1
                    copy_mode = "indexed"
                    fallback_blocks = len(self.load_block_mapping)
                except Exception:
                    logger.exception("indexed CPU offload load copy failed; falling back to span copy")
                    copy_ops = 0
                    fallback_blocks = 0
                    for cpu_start, gpu_start, span_len in block_spans:
                        if span_len == 1:
                            fallback_blocks += 1
                            for gpu_layer_part, cpu_layer_part in layer_parts:
                                gpu_layer_part[gpu_start].copy_(cpu_layer_part[cpu_start], non_blocking=True)
                                copy_ops += 1
                        else:
                            contiguous_blocks += span_len
                            cpu_end = cpu_start + span_len
                            gpu_end = gpu_start + span_len
                            for gpu_layer_part, cpu_layer_part in layer_parts:
                                gpu_layer_part[gpu_start:gpu_end].copy_(
                                    cpu_layer_part[cpu_start:cpu_end],
                                    non_blocking=True,
                                )
                            copy_ops += 1
            elif directional_block_spans is not None:
                copy_mode = "reverse_span"
                for cpu_start, gpu_start, span_len, direction in directional_block_spans:
                    if span_len == 1:
                        fallback_blocks += 1
                        for gpu_layer_part, cpu_layer_part in layer_parts:
                            gpu_layer_part[gpu_start].copy_(cpu_layer_part[cpu_start], non_blocking=True)
                            copy_ops += 1
                    elif direction == 1:
                        contiguous_blocks += span_len
                        cpu_end = cpu_start + span_len
                        gpu_end = gpu_start + span_len
                        for gpu_layer_part, cpu_layer_part in layer_parts:
                            gpu_layer_part[gpu_start:gpu_end].copy_(
                                cpu_layer_part[cpu_start:cpu_end],
                                non_blocking=True,
                            )
                            copy_ops += 1
                    else:
                        reverse_blocks += span_len
                        cpu_end = cpu_start + span_len
                        gpu_low = gpu_start - span_len + 1
                        gpu_high = gpu_start + 1
                        for gpu_layer_part, cpu_layer_part in layer_parts:
                            gpu_layer_part[gpu_low:gpu_high].copy_(
                                cpu_layer_part[cpu_start:cpu_end].flip(0),
                                non_blocking=True,
                            )
                            copy_ops += 1
            else:
                for cpu_start, gpu_start, span_len in block_spans:
                    if span_len == 1:
                        fallback_blocks += 1
                        for gpu_layer_part, cpu_layer_part in layer_parts:
                            gpu_layer_part[gpu_start].copy_(cpu_layer_part[cpu_start], non_blocking=True)
                            copy_ops += 1
                    else:
                        contiguous_blocks += span_len
                        cpu_end = cpu_start + span_len
                        gpu_end = gpu_start + span_len
                        for gpu_layer_part, cpu_layer_part in layer_parts:
                            gpu_layer_part[gpu_start:gpu_end].copy_(
                                cpu_layer_part[cpu_start:cpu_end],
                                non_blocking=True,
                            )
                            copy_ops += 1
            copy_enqueue_wall_ms = (time.perf_counter() - copy_enqueue_start_s) * 1000.0
            if self.profile_timing and copy_start_event is not None and copy_end_event is not None:
                copy_end_event.record(self.load_stream)
                block_bytes = [int(part.stride(0) * part.element_size()) for part in cpu_parts]
                self._pending_load_profiles.append(
                    {
                        "mode": copy_mode,
                        "copy_reason": "main_span_copy",
                        "layer": int(layer),
                        "part_index": None,
                        "logical_bytes": int(len(self.load_block_mapping) * sum(block_bytes)),
                        "source_span_bytes": int(sum(span_len for _, _, span_len in block_spans) * sum(block_bytes)),
                        "block_bytes": int(block_bytes[0]) if block_bytes else 0,
                        "enqueue_wall_ms": float(copy_enqueue_wall_ms),
                        **mapping_profile,
                        "start_event": copy_start_event,
                        "end_event": copy_end_event,
                    }
                )
            if self.real_path_trace:
                record_host_gather_event(
                    "cpu_offload_connector_worker",
                    "load_layer_enqueue_end",
                    layer=int(layer),
                    mode=copy_mode,
                    **mapping_profile,
                )
        if _profile_enabled() and self.load_block_mapping:
            logger.info(
                "[cpu-offload-profile] worker_schedule_layer_load layer=%d mode=%s load_blocks=%d spans=%d "
                "cpu_adjacent_pairs=%d gpu_adjacent_pairs=%d pair_sample=%s contiguous_blocks=%d "
                "reverse_blocks=%d fallback_blocks=%d copy_ops=%d original_copy_ops=%d copy_ops_saved=%d "
                "schedule_ms=%.3f",
                layer,
                copy_mode,
                len(self.load_block_mapping),
                len(block_spans),
                self.load_cpu_adjacent_pairs,
                self.load_gpu_adjacent_pairs,
                self.load_pair_sample,
                contiguous_blocks,
                reverse_blocks,
                fallback_blocks,
                copy_ops,
                original_copy_ops,
                original_copy_ops - copy_ops,
                _elapsed_ms(started),
            )

    def get_finished(self) -> set[str]:
        while True:
            try:
                req_id, req = self.save_input_queue.get_nowait()
            except queue.Empty:
                break
            self._save_req(req_id, req)

        done_sending: set[str] = set()
        while True:
            try:
                id = self.save_output_queue.get_nowait()
            except queue.Empty:
                break
            done_sending.add(id)
        for id in done_sending:
            del self.requests[id]
        if self.tp_world_size == 1:
            return done_sending
        if self.tp_rank == 0:
            for req_id in done_sending:
                self.done_sending_count[req_id] += 1
            other_ranks_finished_ids: list[str] = []
            for i in range(1, self.tp_world_size):
                other_ranks_finished_ids.extend(self.tp_group.recv_object(src=i))
            for req_id in other_ranks_finished_ids:
                self.done_sending_count[req_id] += 1
            all_done_sending: set[str] = set()
            for req_id in list(self.done_sending_count.keys()):
                if self.done_sending_count[req_id] == self.tp_world_size:
                    del self.done_sending_count[req_id]
                    all_done_sending.add(req_id)
            # release cpu_kv_cache after request sending finished
            # to avoid rpc blocking, use thread to call rpc asynchronously
            sending_finished_thread = threading.Thread(target=self._sending_finished, args=(all_done_sending,))
            sending_finished_thread.daemon = True
            sending_finished_thread.start()

            return all_done_sending
        else:
            self.tp_group.send_object(done_sending, dst=0)
            return done_sending

    def _sending_finished(self, all_done_sending):
        for req_id in all_done_sending:
            started = time.perf_counter()
            logger.debug("call cache_and_free_slots for req_id: %s", req_id)
            self.zmq_rpc_client.call("cache_and_free_slots", req_id)
            if _profile_enabled():
                logger.info(
                    "[cpu-offload-profile] worker_cache_and_free req=%s wall_ms=%.3f",
                    req_id,
                    _elapsed_ms(started),
                )

    def _save_req(self, req_id: str, req: ReqMeta):
        save_block_mapping = []
        started = time.perf_counter()
        save_start_block = req.num_cpu_computed_tokens // self.block_size
        save_end_block = min(
            (req.num_computed_tokens + req.num_scheduled_tokens) // self.block_size,
            len(req.cpu_block_ids),
            len(req.gpu_block_ids),
        )
        for i in range(
            save_start_block,
            save_end_block,
        ):
            save_block_mapping.append((req.gpu_block_ids[i], req.cpu_block_ids[i]))
        copy_ops = 0
        # MLA: kv_layer is tuple[tensor, tensor] means (rope, nope).
        # non-MLA: kv_layer is list[tensor], typically means [k, v].
        if self.use_mla:
            start, step = self.tp_rank, self.tp_world_size
        else:
            start, step = 0, 1
        rank_save_block_mapping = [
            (cpu_block_id, gpu_block_id)
            for gpu_block_id, cpu_block_id in save_block_mapping[start::step]
        ]
        cache_layer_parts = [
            (cpu_layer_part, gpu_layer_part)
            for cpu_kv_caches, gpu_kv_caches in zip(self.cpu_kv_caches, self.gpu_kv_caches.values())
            for cpu_layer_part, gpu_layer_part in zip(cpu_kv_caches, gpu_kv_caches)
        ]
        block_spans = _coalesce_block_copy_spans(rank_save_block_mapping)
        directional_block_spans = (
            _coalesce_directional_block_copy_spans(rank_save_block_mapping)
            if self.use_reverse_span_copy
            else None
        )
        cpu_adjacent, gpu_adjacent, pair_sample = _block_pair_locality(rank_save_block_mapping)
        original_copy_ops = len(rank_save_block_mapping) * len(cache_layer_parts)
        contiguous_blocks = 0
        reverse_blocks = 0
        fallback_blocks = 0
        if directional_block_spans is not None:
            for cpu_start, gpu_start, span_len, direction in directional_block_spans:
                if span_len == 1:
                    fallback_blocks += 1
                    for cpu_layer_part, gpu_layer_part in cache_layer_parts:
                        cpu_layer_part[cpu_start].copy_(gpu_layer_part[gpu_start], non_blocking=False)
                        copy_ops += 1
                elif direction == 1:
                    contiguous_blocks += span_len
                    cpu_end = cpu_start + span_len
                    gpu_end = gpu_start + span_len
                    for cpu_layer_part, gpu_layer_part in cache_layer_parts:
                        cpu_layer_part[cpu_start:cpu_end].copy_(
                            gpu_layer_part[gpu_start:gpu_end],
                            non_blocking=False,
                        )
                        copy_ops += 1
                else:
                    reverse_blocks += span_len
                    cpu_end = cpu_start + span_len
                    gpu_low = gpu_start - span_len + 1
                    gpu_high = gpu_start + 1
                    for cpu_layer_part, gpu_layer_part in cache_layer_parts:
                        cpu_layer_part[cpu_start:cpu_end].copy_(
                            gpu_layer_part[gpu_low:gpu_high].flip(0),
                            non_blocking=False,
                        )
                        copy_ops += 1
        else:
            for cpu_start, gpu_start, span_len in block_spans:
                if span_len == 1:
                    fallback_blocks += 1
                    for cpu_layer_part, gpu_layer_part in cache_layer_parts:
                        cpu_layer_part[cpu_start].copy_(gpu_layer_part[gpu_start], non_blocking=False)
                        copy_ops += 1
                else:
                    contiguous_blocks += span_len
                    cpu_end = cpu_start + span_len
                    gpu_end = gpu_start + span_len
                    for cpu_layer_part, gpu_layer_part in cache_layer_parts:
                        cpu_layer_part[cpu_start:cpu_end].copy_(
                            gpu_layer_part[gpu_start:gpu_end],
                            non_blocking=False,
                        )
                        copy_ops += 1
        if _profile_enabled():
            logger.info(
                "[cpu-offload-profile] worker_save req=%s mode=sync save_block_range=%d:%d save_blocks=%d "
                "rank_save_blocks=%d spans=%d cpu_adjacent_pairs=%d gpu_adjacent_pairs=%d pair_sample=%s "
                "contiguous_blocks=%d reverse_blocks=%d fallback_blocks=%d copy_ops=%d "
                "original_copy_ops=%d copy_ops_saved=%d cpu_tokens=%d computed_tokens=%d scheduled_tokens=%d "
                "gpu_blocks=%d cpu_blocks=%d wall_ms=%.3f",
                req_id,
                save_start_block,
                save_end_block,
                len(save_block_mapping),
                len(rank_save_block_mapping),
                len(block_spans),
                cpu_adjacent,
                gpu_adjacent,
                pair_sample,
                contiguous_blocks,
                reverse_blocks,
                fallback_blocks,
                copy_ops,
                original_copy_ops,
                original_copy_ops - copy_ops,
                req.num_cpu_computed_tokens,
                req.num_computed_tokens,
                req.num_scheduled_tokens,
                len(req.gpu_block_ids),
                len(req.cpu_block_ids),
                _elapsed_ms(started),
            )
            if (
                len(save_block_mapping) == 0
                and (req.num_computed_tokens + req.num_scheduled_tokens) // self.block_size > save_start_block
            ):
                logger.warning(
                    "[cpu-offload-profile] worker_save_empty req=%s start_block=%d full_blocks=%d "
                    "gpu_blocks=%d cpu_blocks=%d",
                    req_id,
                    save_start_block,
                    (req.num_computed_tokens + req.num_scheduled_tokens) // self.block_size,
                    len(req.gpu_block_ids),
                    len(req.cpu_block_ids),
                )
        self.save_output_queue.put(req_id)


# copied and modified from vllm_ascend/worker/model_runner_v1.py
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """
    Generates the KVCacheSpec by parsing the kv cache format from each
    Attention module in the static forward context.
    Returns:
        KVCacheSpec: A dictionary mapping layer names to their KV cache
        format. Layers that do not need KV cache are not included.
    """
    if has_ec_transfer() and get_ec_transfer().is_producer:
        return {}

    use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
    if vllm_config.cache_config.cache_dtype == "auto":
        kv_cache_dtype = vllm_config.model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[vllm_config.cache_config.cache_dtype]

    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
    # ordering expected by graph parameter update logic in attention backends.
    mamba_layers: dict[str, MambaBase] = {}
    for layer_name, attn_module in attn_layers.items():
        if isinstance(attn_module, Attention):
            if spec := attn_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec

        elif isinstance(attn_module, MLAAttention):
            if use_sparse:
                # TODO(cmq): This is a hack way to fix deepseek kvcache when
                # using DSA. Fix the spec in vLLM is the final way.
                block_size = vllm_config.cache_config.block_size
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size, num_kv_heads=1, head_size=attn_module.head_size, dtype=kv_cache_dtype
                )
            elif spec := attn_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec

        elif isinstance(attn_module, MambaBase):
            mamba_layers[layer_name] = attn_module

    if len(mamba_layers) > 0:
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("Prefix caching is not supported for Mamba yet.")
        for layer_name, mamba_module in mamba_layers.items():
            if spec := mamba_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec

    return kv_cache_spec
