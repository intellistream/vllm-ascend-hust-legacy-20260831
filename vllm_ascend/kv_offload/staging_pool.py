# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental worker-local staging pool for CPU-offload KV restore.

The staging pool is intentionally env-gated. It keeps fixed CPU staging slots
registered for mapped-host access, packs selected CPU KV blocks into those
slots, then lets the custom NPU gather op restore from dense staging blocks.
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from vllm.logger import logger

from vllm_ascend.kv_offload.host_gather_stats import record_host_gather_event


@dataclass(frozen=True)
class StagingPoolConfig:
    slots: int
    slab_blocks: int
    pack_backend: str
    pack_threads: int
    fused_kv: bool
    build_dir: str
    fallback_on_error: bool


@dataclass
class StagingSlot:
    index: int
    parts: list[torch.Tensor]
    event: torch.npu.Event | None = None
    uses: int = 0


class _CppStagingPacker:
    _build_lock = threading.Lock()

    def __init__(self, build_dir: str, threads: int, *, persistent: bool) -> None:
        self.threads = max(1, int(threads))
        self.persistent = persistent
        self.lib = ctypes.CDLL(str(self._ensure_library(Path(build_dir))))
        self._configure_symbols()
        self.handle: ctypes.c_void_p | None = None
        if self.persistent:
            self.handle = ctypes.c_void_p(
                self.lib.kv_staging_packer_create(ctypes.c_int32(self.threads))
            )
            if not self.handle.value:
                raise RuntimeError("kv_staging_packer_create failed")

    def close(self) -> None:
        if self.handle is not None and self.handle.value:
            self.lib.kv_staging_packer_destroy(self.handle)
        self.handle = None

    def pack(
        self,
        *,
        src_parts: Sequence[torch.Tensor],
        dst_parts: Sequence[torch.Tensor],
        src_ids: torch.Tensor,
        fused_kv: bool,
    ) -> None:
        if fused_kv and len(src_parts) == 2 and len(dst_parts) == 2:
            ret = self._pack_two(src_parts, dst_parts, src_ids)
            if ret != 0:
                raise RuntimeError(f"kv_staging_pack_blocks2 failed with code {ret}")
            return
        for src, dst in zip(src_parts, dst_parts):
            ret = self._pack_one(src, dst, src_ids)
            if ret != 0:
                raise RuntimeError(f"kv_staging_pack_blocks failed with code {ret}")

    def _pack_one(self, src: torch.Tensor, dst: torch.Tensor, src_ids: torch.Tensor) -> int:
        block_count = int(src_ids.numel())
        block_bytes = _block_bytes(src)
        if self.persistent:
            return int(
                self.lib.kv_staging_packer_pack_blocks(
                    self.handle,
                    ctypes.c_void_p(src.data_ptr()),
                    ctypes.c_void_p(dst.data_ptr()),
                    ctypes.c_void_p(src_ids.data_ptr()),
                    ctypes.c_int64(block_count),
                    ctypes.c_int64(block_bytes),
                )
            )
        return int(
            self.lib.kv_staging_pack_blocks(
                ctypes.c_void_p(src.data_ptr()),
                ctypes.c_void_p(dst.data_ptr()),
                ctypes.c_void_p(src_ids.data_ptr()),
                ctypes.c_int64(block_count),
                ctypes.c_int64(block_bytes),
                ctypes.c_int32(self.threads),
            )
        )

    def _pack_two(
        self,
        src_parts: Sequence[torch.Tensor],
        dst_parts: Sequence[torch.Tensor],
        src_ids: torch.Tensor,
    ) -> int:
        block_count = int(src_ids.numel())
        block_bytes = _block_bytes(src_parts[0])
        if self.persistent:
            return int(
                self.lib.kv_staging_packer_pack_blocks2(
                    self.handle,
                    ctypes.c_void_p(src_parts[0].data_ptr()),
                    ctypes.c_void_p(src_parts[1].data_ptr()),
                    ctypes.c_void_p(dst_parts[0].data_ptr()),
                    ctypes.c_void_p(dst_parts[1].data_ptr()),
                    ctypes.c_void_p(src_ids.data_ptr()),
                    ctypes.c_int64(block_count),
                    ctypes.c_int64(block_bytes),
                )
            )
        return int(
            self.lib.kv_staging_pack_blocks2(
                ctypes.c_void_p(src_parts[0].data_ptr()),
                ctypes.c_void_p(src_parts[1].data_ptr()),
                ctypes.c_void_p(dst_parts[0].data_ptr()),
                ctypes.c_void_p(dst_parts[1].data_ptr()),
                ctypes.c_void_p(src_ids.data_ptr()),
                ctypes.c_int64(block_count),
                ctypes.c_int64(block_bytes),
                ctypes.c_int32(self.threads),
            )
        )

    @classmethod
    def _ensure_library(cls, build_dir: Path) -> Path:
        source = Path(__file__).with_name("kv_staging_pack.cpp")
        lib_path = build_dir / "libkv_staging_pack.so"
        with cls._build_lock:
            build_dir.mkdir(parents=True, exist_ok=True)
            needs_build = (
                not lib_path.exists()
                or lib_path.stat().st_mtime < source.stat().st_mtime
            )
            if needs_build:
                subprocess.run(
                    [
                        "g++",
                        "-O3",
                        "-std=c++17",
                        "-fPIC",
                        "-shared",
                        "-pthread",
                        str(source),
                        "-o",
                        str(lib_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        return lib_path

    def _configure_symbols(self) -> None:
        self.lib.kv_staging_pack_blocks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
        ]
        self.lib.kv_staging_pack_blocks.restype = ctypes.c_int
        self.lib.kv_staging_pack_blocks2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
        ]
        self.lib.kv_staging_pack_blocks2.restype = ctypes.c_int
        self.lib.kv_staging_packer_create.argtypes = [ctypes.c_int32]
        self.lib.kv_staging_packer_create.restype = ctypes.c_void_p
        self.lib.kv_staging_packer_destroy.argtypes = [ctypes.c_void_p]
        self.lib.kv_staging_packer_destroy.restype = None
        self.lib.kv_staging_packer_pack_blocks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.kv_staging_packer_pack_blocks.restype = ctypes.c_int
        self.lib.kv_staging_packer_pack_blocks2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.kv_staging_packer_pack_blocks2.restype = ctypes.c_int


class _TorchStagingPacker:
    def close(self) -> None:
        return

    def pack(
        self,
        *,
        src_parts: Sequence[torch.Tensor],
        dst_parts: Sequence[torch.Tensor],
        src_ids: torch.Tensor,
        fused_kv: bool,
    ) -> None:
        del fused_kv
        for src, dst in zip(src_parts, dst_parts):
            torch.index_select(src, 0, src_ids, out=dst[: src_ids.numel()])


class WorkerLocalStagingPool:
    def __init__(
        self,
        *,
        config: StagingPoolConfig,
        part_shapes: Sequence[Sequence[int]],
        dtype: torch.dtype,
    ) -> None:
        if not part_shapes:
            raise ValueError("staging pool requires at least one KV part shape")
        self.config = config
        self.part_count = len(part_shapes)
        self.next_index = 0
        self.slots: list[StagingSlot] = []
        self._closed = False

        for slot_idx in range(max(1, int(config.slots))):
            parts = []
            for shape in part_shapes:
                if len(shape) < 1:
                    raise ValueError(f"invalid KV part shape for staging: {shape}")
                slab_blocks = max(1, int(config.slab_blocks))
                staging_shape = (slab_blocks, *tuple(int(dim) for dim in shape[1:]))
                parts.append(torch.empty(staging_shape, dtype=dtype, device="cpu"))
            self.slots.append(StagingSlot(index=slot_idx, parts=parts))

        self.packer = self._make_packer(config)
        self.register_stats = self._register_all()
        record_host_gather_event(
            "cpu_offload_staging_pool",
            "staging_pool_initialized",
            slots=len(self.slots),
            part_count=self.part_count,
            slab_blocks=int(config.slab_blocks),
            pack_backend=config.pack_backend,
            pack_threads=int(config.pack_threads),
            fused_kv=bool(config.fused_kv),
            **self.register_stats,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.synchronize_all()
        except Exception:
            pass
        try:
            self.packer.close()
        except Exception:
            logger.debug("failed to close staging packer", exc_info=True)

    def acquire(self) -> tuple[StagingSlot, float]:
        if not self.slots:
            raise RuntimeError("staging pool has no slots")
        slot = self.slots[self.next_index]
        self.next_index = (self.next_index + 1) % len(self.slots)
        wait_start_s = time.perf_counter()
        if slot.event is not None:
            slot.event.synchronize()
            slot.event = None
        wait_ms = (time.perf_counter() - wait_start_s) * 1000.0
        record_host_gather_event(
            "cpu_offload_staging_pool",
            "staging_slot_acquired",
            slot=int(slot.index),
            wait_wall_ms=float(wait_ms),
            previous_uses=int(slot.uses),
        )
        return slot, wait_ms

    def mark_inflight(self, slot: StagingSlot, event: torch.npu.Event) -> None:
        slot.uses += 1
        slot.event = event

    def synchronize_all(self) -> float:
        wait_start_s = time.perf_counter()
        for slot in self.slots:
            if slot.event is not None:
                slot.event.synchronize()
                slot.event = None
        return (time.perf_counter() - wait_start_s) * 1000.0

    def pack(
        self,
        *,
        layer: int,
        slot: StagingSlot,
        src_parts: Sequence[torch.Tensor],
        src_ids: torch.Tensor,
    ) -> float:
        if len(src_parts) != self.part_count:
            raise RuntimeError(
                "staging part count mismatch: "
                f"expected {self.part_count}, got {len(src_parts)}"
            )
        selected = int(src_ids.numel())
        if selected > int(slot.parts[0].shape[0]):
            raise RuntimeError(
                "staging slab too small: "
                f"selected={selected}, slab={int(slot.parts[0].shape[0])}"
            )
        pack_start_s = time.perf_counter()
        dst_parts = [part[:selected] for part in slot.parts]
        self.packer.pack(
            src_parts=src_parts,
            dst_parts=dst_parts,
            src_ids=src_ids,
            fused_kv=bool(self.config.fused_kv),
        )
        pack_ms = (time.perf_counter() - pack_start_s) * 1000.0
        record_host_gather_event(
            "cpu_offload_staging_pool",
            "staging_pack_timing",
            layer=int(layer),
            slot=int(slot.index),
            selected_blocks=selected,
            pack_wall_ms=float(pack_ms),
            pack_backend=self.config.pack_backend,
            pack_threads=int(self.config.pack_threads),
            fused_kv=bool(self.config.fused_kv),
        )
        return pack_ms

    def _register_all(self) -> dict[str, Any]:
        op = _register_mapping_op()
        start_s = time.perf_counter()
        register_bytes = 0
        requested_bytes = 0
        hits = 0
        misses = 0
        elapsed_us = 0
        for slot in self.slots:
            for part in slot.parts:
                part.zero_()
                stats = op(part)
                register_bytes += int(stats.get("register_bytes", 0))
                requested_bytes += int(stats.get("requested_bytes", 0))
                hit = int(stats.get("mapping_hit", 0))
                hits += hit
                misses += 0 if hit else 1
                elapsed_us += int(stats.get("elapsed_us", 0))
        return {
            "register_wall_ms": (time.perf_counter() - start_s) * 1000.0,
            "register_op_ms": float(elapsed_us) / 1000.0,
            "register_bytes": int(register_bytes),
            "requested_bytes": int(requested_bytes),
            "mapping_hits": int(hits),
            "mapping_misses": int(misses),
        }

    @staticmethod
    def _make_packer(config: StagingPoolConfig) -> Any:
        if config.pack_backend == "torch":
            return _TorchStagingPacker()
        if config.pack_backend == "cpp":
            return _CppStagingPacker(config.build_dir, config.pack_threads, persistent=False)
        if config.pack_backend == "cpp_pool":
            return _CppStagingPacker(config.build_dir, config.pack_threads, persistent=True)
        raise ValueError(f"unsupported staging pack backend: {config.pack_backend}")


def _register_mapping_op() -> Any:
    op = getattr(torch.ops._C_ascend, "register_host_mapping", None)
    if op is None:
        op = torch.ops._C_ascend.register_kv_cache_block_gather_host_mapping
    return op


def _block_bytes(tensor: torch.Tensor) -> int:
    if tensor.dim() < 1:
        raise ValueError("KV tensor must have at least one dimension")
    return int(tensor.stride(0) * tensor.element_size())
