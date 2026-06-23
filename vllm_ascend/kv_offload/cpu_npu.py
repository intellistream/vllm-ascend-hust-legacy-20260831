from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


@dataclass
class Transfer:
    job_id: int
    stream: Any
    start_event: Any
    end_event: Any
    num_bytes: int


class SingleDirectionNPUOffloadingHandler(OffloadingHandler):
    """Conservative CPU/NPU KV transfer handler for the current vLLM API."""

    def __init__(
        self,
        npu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        npu_to_cpu: bool,
    ) -> None:
        assert len(npu_tensors) == len(cpu_tensors)
        assert npu_tensors
        for npu_tensor, cpu_tensor in zip(npu_tensors, cpu_tensors):
            assert npu_tensor.dtype == torch.int8
            assert npu_tensor.ndim == 2
            assert npu_tensor.device.type == "npu"
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            assert cpu_tensor.shape[1] == npu_tensor.shape[1] * block_size_factor

        self.src_tensors = npu_tensors if npu_to_cpu else cpu_tensors
        self.dst_tensors = cpu_tensors if npu_to_cpu else npu_tensors
        self.npu_to_cpu = npu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self.src_block_size_factor = 1 if npu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if npu_to_cpu else 1
        self.transfer_type = ("GPU", "CPU") if npu_to_cpu else ("CPU", "GPU")

        self._transfer_events: dict[int, Any] = {}
        self._transfers: deque[Transfer] = deque()
        self._stream_pool: list[Any] = []
        self._event_pool: list[Any] = []

    def _get_stream(self) -> Any:
        if self._stream_pool:
            return self._stream_pool.pop()
        return torch.npu.Stream()

    def _get_event(self) -> Any:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.npu.Event(enable_timing=True)

    def _copy_group(
        self,
        group_size: int,
        block_idx: int,
        group_data_refs: list[CanonicalKVCacheRef],
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
    ) -> int:
        src_logical_blocks_to_skip = block_idx % self.src_block_size_factor
        dst_logical_blocks_to_skip = block_idx % self.dst_block_size_factor
        num_transfer_bytes = 0

        for data_ref in group_data_refs:
            src_tensor = self.src_tensors[data_ref.tensor_idx]
            dst_tensor = self.dst_tensors[data_ref.tensor_idx]
            src_page_bytes = src_tensor.shape[1] // self.src_block_size_factor
            dst_page_bytes = dst_tensor.shape[1] // self.dst_block_size_factor
            copy_bytes = data_ref.page_size_bytes
            assert copy_bytes <= src_page_bytes
            assert copy_bytes <= dst_page_bytes

            for logical_offset in range(group_size):
                src_pos = src_logical_blocks_to_skip + logical_offset
                dst_pos = dst_logical_blocks_to_skip + logical_offset
                src_block = int(src_blocks[src_pos // self.src_block_size_factor])
                dst_block = int(dst_blocks[dst_pos // self.dst_block_size_factor])
                src_sub_block = src_pos % self.src_block_size_factor
                dst_sub_block = dst_pos % self.dst_block_size_factor

                src_start = src_sub_block * src_page_bytes
                dst_start = dst_sub_block * dst_page_bytes
                src_view = src_tensor[src_block, src_start:src_start + copy_bytes]
                dst_view = dst_tensor[dst_block, dst_start:dst_start + copy_bytes]
                dst_view.copy_(src_view, non_blocking=True)
                num_transfer_bytes += copy_bytes

        return num_transfer_bytes

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        gpu_spec = src_spec if self.npu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        assert len(gpu_spec.group_sizes) == len(self.kv_cache_groups_data_refs)
        assert len(gpu_spec.block_indices) == len(self.kv_cache_groups_data_refs)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        src_offset = 0
        dst_offset = 0
        num_transfer_bytes = 0

        stream = self._get_stream()
        start_event = self._get_event()
        end_event = self._get_event()

        if self.npu_to_cpu:
            stream.wait_stream(torch.npu.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)

        with torch.npu.stream(stream):
            start_event.record(stream)
            for group_size, block_idx, group_data_refs in zip(
                gpu_spec.group_sizes,
                gpu_spec.block_indices,
                self.kv_cache_groups_data_refs,
            ):
                if group_size == 0:
                    continue

                src_count = cdiv(
                    group_size + block_idx % self.src_block_size_factor,
                    self.src_block_size_factor,
                )
                dst_count = cdiv(
                    group_size + block_idx % self.dst_block_size_factor,
                    self.dst_block_size_factor,
                )
                group_src = src_blocks[src_offset:src_offset + src_count]
                group_dst = dst_blocks[dst_offset:dst_offset + dst_count]
                num_transfer_bytes += self._copy_group(
                    group_size, block_idx, group_data_refs, group_src, group_dst
                )
                src_offset += src_count
                dst_offset += dst_count
            end_event.record(stream)

        assert src_offset == len(src_blocks)
        assert dst_offset == len(dst_blocks)
        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=transfer_time,
                    transfer_type=self.transfer_type,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            self._transfers.popleft().end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()


class CpuNpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ) -> None:
        logger.info("Allocating %d CPU tensors for NPU offload...", len(kv_caches.tensors))
        npu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, npu_page_size_bytes)
            )
            cpu_page_size_bytes = npu_page_size_bytes * block_size_factor
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, cpu_page_size_bytes),
                dtype=torch.int8,
                device="cpu",
            )
            npu_tensors.append(npu_tensor)
            cpu_tensors.append(cpu_tensor)

        self.npu_to_cpu_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=True,
        )
        self.cpu_to_npu_handler = SingleDirectionNPUOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=False,
        )
