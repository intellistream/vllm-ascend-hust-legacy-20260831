import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL = REPO_ROOT / ("csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp")
TILING = REPO_ROOT / ("csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp")


def test_active_mask_uses_bounded_chunks_and_private_scratch():
    source = KERNEL.read_text()
    match = re.search(
        r"ACTIVE_MASK_COPY_CHUNK_ELEMENTS\s*=\s*(\d+)\s*\*\s*(\d+)",
        source,
    )

    assert match is not None
    chunk_elements = int(match.group(1)) * int(match.group(2))
    assert chunk_elements * 4 <= 65535
    assert "chunkOffset += ACTIVE_MASK_COPY_CHUNK_ELEMENTS" in source
    assert "return workspaceInfo.ptrMaskedExpertIdx" in source
    assert "routingExpertIdx = ApplyXActiveMask(params)" in source
    assert "reinterpret_cast<GM_ADDR>(params.ptrA), routingExpertIdx" in source
    assert "DataCopyPad(expertIdxGm[startIdx]" not in source


def test_routing_workspace_covers_masked_routes_and_init_routing():
    kernel_source = KERNEL.read_text()
    tiling_source = TILING.read_text()

    assert "ptrInitRoutingWorkspace = ptrMaskedExpertIdx + maskedExpertIdxSize" in kernel_source
    assert "workspaceInfo.ptrInitRoutingWorkspace" in kernel_source
    assert "expandedRowIdxWorkspace + expertIdxScratchWorkspace + initRoutingWorkspace" in tiling_source
