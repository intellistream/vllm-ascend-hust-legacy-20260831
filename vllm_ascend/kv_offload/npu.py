from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandlers


class NPUOffloadingSpec(OffloadingSpec):
    """CPU KV offloading spec for Ascend NPU KV cache tensors."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config)

        if kv_cache_config.num_blocks > 0:
            total_npu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_npu_kv_bytes // kv_cache_config.num_blocks
            ) * vllm_config.parallel_config.world_size
        else:
            kv_bytes_per_block = 0

        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        if "num_cpu_blocks" in self.extra_config:
            self.num_blocks = int(self.extra_config["num_cpu_blocks"])
        else:
            cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
            if not cpu_bytes_to_use:
                raise Exception(
                    "Either cpu_bytes_to_use or num_cpu_blocks must be specified "
                    "in kv_connector_extra_config"
                )
            self.num_blocks = (
                int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
                if kv_bytes_per_offloaded_block > 0
                else 0
            )

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self._manager: OffloadingManager | None = None
        self._handlers: CpuNpuOffloadingHandlers | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
            )

            store_threshold = int(self.extra_config.get("store_threshold", 0))
            if store_threshold >= 2:
                max_tracker_size = int(
                    self.extra_config.get("max_tracker_size", 64_000)
                )
                self._manager = FilterReusedOffloadingManager(
                    backing=self._manager,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            self._handlers = CpuNpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
            )

        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.npu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_npu_handler
