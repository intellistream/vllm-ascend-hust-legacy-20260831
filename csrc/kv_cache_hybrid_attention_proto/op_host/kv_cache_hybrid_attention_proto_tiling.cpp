#include <algorithm>
#include <cstdint>
#include <cstring>

#include "kv_cache_hybrid_attention_proto_tiling_data.h"
#include "log/ops_log.h"
#include "register/op_impl_kernel_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int64_t DEFAULT_TILE_ELEMS = 1024;
}

struct KvCacheHybridAttentionProtoCompileInfo {};

static ge::graphStatus KvCacheHybridAttentionProtoTilingFunc(
    gert::TilingContext* context)
{
    auto kindsShape = context->GetInputShape(0);
    auto idsShape = context->GetInputShape(1);
    auto deviceShape = context->GetInputShape(3);
    auto queryShape = context->GetInputShape(4);
    if (kindsShape == nullptr || idsShape == nullptr ||
        deviceShape == nullptr || queryShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto kinds = kindsShape->GetStorageShape();
    const auto ids = idsShape->GetStorageShape();
    const auto device = deviceShape->GetStorageShape();
    const auto query = queryShape->GetStorageShape();
    if (kinds.GetDimNum() != 1 || ids.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t selectedBlocks = ids.GetDim(0);
    if (selectedBlocks <= 0 || kinds.GetDim(0) != selectedBlocks) {
        return ge::GRAPH_FAILED;
    }
    if (device.GetDimNum() < 2 || query.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    int64_t deviceElems = 1;
    for (size_t i = 0; i < device.GetDimNum(); ++i) {
        deviceElems *= device.GetDim(i);
    }
    const int64_t elemsPerBlock = deviceElems / device.GetDim(0);
    if (elemsPerBlock <= 0 || elemsPerBlock % 8 != 0 ||
        query.GetDim(0) != elemsPerBlock) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    auto tiling = context->GetTilingData<KvCacheHybridAttentionProtoTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->selectedBlocks = selectedBlocks;
    tiling->elemsPerBlock = elemsPerBlock;
    tiling->tileElems = std::min(DEFAULT_TILE_ELEMS, elemsPerBlock);
    context->SetBlockDim(static_cast<uint32_t>(
        std::min(coreNum, selectedBlocks)));
    // This prototype has one non-templated kernel entry, generated as key 0.
    context->SetTilingKey(0);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus KvCacheHybridAttentionProtoTilingParse(
    gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheHybridAttentionProto)
    .Tiling(KvCacheHybridAttentionProtoTilingFunc)
    .TilingParse<KvCacheHybridAttentionProtoCompileInfo>(
        KvCacheHybridAttentionProtoTilingParse);
} // namespace optiling
