# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_ascend.kv_offload import experimental_mapped
from vllm_ascend.kv_offload.npu import (
    CPUOffloadingSpec,
    NPUOffloadingSpec,
    _set_cpu_bytes_from_legacy_num_blocks,
)


def _canonical_caches(page_size_bytes: int = 32) -> CanonicalKVCaches:
    tensor = torch.empty((4, page_size_bytes), dtype=torch.int8)
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor, page_size_bytes)],
        group_data_refs=[[CanonicalKVCacheRef(0, page_size_bytes)]],
    )


def test_mapped_worker_rejects_unsupported_configuration_before_allocating():
    caches = _canonical_caches()

    with pytest.raises(ValueError, match="block_size_factor == 1"):
        experimental_mapped.MappedOffloadingHandler(caches, 2, 1)
    with pytest.raises(ValueError, match="greater than zero"):
        experimental_mapped.MappedOffloadingHandler(caches, 1, 0)


def test_mapped_worker_fails_when_runtime_lifecycle_ops_are_missing(monkeypatch):
    monkeypatch.setattr(
        experimental_mapped,
        "_load_mapped_gather_ops",
        lambda: (None, None, None),
    )

    with pytest.raises(RuntimeError, match="lifecycle ops are unavailable"):
        experimental_mapped.MappedOffloadingHandler(_canonical_caches(), 1, 1)


def test_mapped_layout_rejects_unaligned_pages():
    page_size_bytes = 31
    npu_tensors = [torch.empty((4, page_size_bytes), dtype=torch.int8)]
    cpu_tensors = [torch.empty((4, page_size_bytes), dtype=torch.int8)]

    with pytest.raises(ValueError, match="32-byte aligned"):
        experimental_mapped._validate_mapped_gather_layout(
            npu_tensors,
            cpu_tensors,
            [[CanonicalKVCacheRef(0, page_size_bytes)]],
        )


def test_mapped_spec_registers_one_handler_for_both_directions(monkeypatch):
    calls = []
    sentinel = object()

    def create_handler(kv_caches, block_size_factor, num_cpu_blocks):
        calls.append((kv_caches, block_size_factor, num_cpu_blocks))
        return sentinel

    monkeypatch.setattr(
        experimental_mapped,
        "MappedOffloadingHandler",
        create_handler,
    )
    spec = object.__new__(experimental_mapped.MappedOffloadingSpec)
    spec.block_size_factor = 1
    spec.num_blocks = 17
    caches = _canonical_caches()

    routes = list(spec.get_handlers(caches))

    assert routes == [
        (GPULoadStoreSpec, CPULoadStoreSpec, sentinel),
        (CPULoadStoreSpec, GPULoadStoreSpec, sentinel),
    ]
    assert list(spec.get_handlers(caches)) == routes
    assert calls == [(caches, 1, 17)]


def test_mapped_handler_dispatches_both_routes():
    handler = object.__new__(experimental_mapped.MappedOffloadingHandler)
    calls = []
    handler._submit_store = lambda job_id, src, dst: calls.append(("store", job_id, src, dst)) or True
    handler._submit_load = lambda job_id, src, dst: calls.append(("load", job_id, src, dst)) or True
    cpu_spec = CPULoadStoreSpec([1])
    gpu_spec = GPULoadStoreSpec([2], group_sizes=[1], block_indices=[0])

    assert handler.transfer_async(10, (gpu_spec, cpu_spec))
    assert handler.transfer_async(11, (cpu_spec, gpu_spec))
    assert [call[0] for call in calls] == ["store", "load"]


def test_store_descriptor_plan_covers_each_group_and_tensor(monkeypatch):
    monkeypatch.setattr(experimental_mapped, "is_pin_memory_available", lambda: False)
    handler = object.__new__(experimental_mapped.MappedOffloadingHandler)
    handler._store_descriptor_pool = []
    handler.npu_tensors = [
        torch.empty((4, 32), dtype=torch.int8),
        torch.empty((4, 64), dtype=torch.int8),
    ]
    handler.cpu_tensors = [
        torch.empty((4, 32), dtype=torch.int8),
        torch.empty((4, 64), dtype=torch.int8),
    ]
    handler.kv_cache_groups_data_refs = [
        [CanonicalKVCacheRef(0, 32)],
        [CanonicalKVCacheRef(1, 64)],
    ]

    _, _, _, copy_src, copy_dst, copy_sizes = handler._prepare_store_descriptors(
        np.array([1, 3], dtype=np.int64),
        np.array([2, 0], dtype=np.int64),
        [1, 1],
        2,
    )

    assert copy_src.tolist() == [
        handler.npu_tensors[0].data_ptr() + 32,
        handler.npu_tensors[1].data_ptr() + 3 * 64,
    ]
    assert copy_dst.tolist() == [
        handler.cpu_tensors[0].data_ptr() + 2 * 32,
        handler.cpu_tensors[1].data_ptr(),
    ]
    assert copy_sizes.tolist() == [32, 64]


def test_native_npu_spec_remains_the_default():
    assert CPUOffloadingSpec is NPUOffloadingSpec
    assert NPUOffloadingSpec.get_handlers is not experimental_mapped.MappedOffloadingSpec.get_handlers


def test_legacy_num_blocks_translation_preserves_explicit_bytes():
    extra_config = {"num_cpu_blocks": 10, "cpu_bytes_to_use": 1234}
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra_config,
        ),
    )

    _set_cpu_bytes_from_legacy_num_blocks(vllm_config, SimpleNamespace())

    assert extra_config["cpu_bytes_to_use"] == 1234
