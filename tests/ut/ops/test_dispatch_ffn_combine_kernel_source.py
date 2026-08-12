# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BF16_KERNEL = (
    REPO_ROOT / "csrc" / "mc2" / "dispatch_ffn_combine_bf16" / "op_kernel" / "dispatch_ffn_combine_bf16_kernel.hpp"
)
W8A8_KERNEL = REPO_ROOT / "csrc" / "mc2" / "dispatch_ffn_combine" / "op_kernel" / "dispatch_ffn_combine_kernel.hpp"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _chunk_elements(source: str) -> int:
    match = re.search(r"ACTIVE_MASK_COPY_CHUNK_ELEMENTS\s*=\s*(\d+)\s*\*\s*(\d+)", source)
    assert match is not None
    return int(match.group(1)) * int(match.group(2))


def _assert_chunked_mask_copy(source: str) -> None:
    chunk_elements = _chunk_elements(source)
    assert chunk_elements * 4 <= 65535
    assert "static_assert(ACTIVE_MASK_COPY_CHUNK_ELEMENTS * sizeof(int32_t) <= 65535" in source
    assert "chunkOffset += ACTIVE_MASK_COPY_CHUNK_ELEMENTS" in source
    assert "remaining = copySize - chunkOffset" in source
    assert "remaining > ACTIVE_MASK_COPY_CHUNK_ELEMENTS" in source
    assert "static_cast<int32_t>(ACTIVE_MASK_COPY_CHUNK_ELEMENTS)" in source
    assert "chunkBytes = static_cast<uint16_t>(chunkSize * sizeof(int32_t))" in source
    assert source.count("{1, chunkBytes, 0, 0}") == 1
    assert source.count("{1, chunkBytes, 0, 0, 0}") == 1
    assert "copySize * sizeof(int32_t)" not in source

    # Exercise the reported first-overflow boundary and multiple chunks.
    for copy_size in (1, 16383, 16384, 16385, 65537):
        chunks = [min(chunk_elements, copy_size - offset) for offset in range(0, copy_size, chunk_elements)]
        assert sum(chunks) == copy_size
        assert max(chunks) * 4 <= 65535


def test_bf16_active_mask_copy_is_chunked_below_datacopypad_limit() -> None:
    _assert_chunked_mask_copy(_read(BF16_KERNEL))


def test_w8a8_active_mask_copy_is_chunked_below_datacopypad_limit() -> None:
    _assert_chunked_mask_copy(_read(W8A8_KERNEL))
