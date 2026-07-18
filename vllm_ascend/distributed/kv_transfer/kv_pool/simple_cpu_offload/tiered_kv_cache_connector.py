"""Ascend adaptation of the Tiered KV cache connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.tiered_kv_cache_connector import (
    TieredKVCacheConnector,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)


class AscendTieredKVCacheConnector(
    TieredKVCacheConnector,
    AscendSimpleCPUOffloadConnector,
):
    """Tiered restore semantics with the native NPU offload worker.

    Cooperative ``super()`` follows this class's diamond MRO: Tiered setup,
    then the Ascend simple-offload adapter, then the upstream base. This keeps
    scheduler behavior unchanged while replacing the worker-side CUDA DMA,
    cache-layout, stream, and event implementation with the NPU versions.
    """

