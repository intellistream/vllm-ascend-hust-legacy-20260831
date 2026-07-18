#include "kv_cache_hybrid_attention_proto.h"

__global__ __aicore__ void kv_cache_hybrid_attention_proto(
    GM_ADDR sourceKinds, GM_ADDR sourceBlockIds, GM_ADDR hostPages,
    GM_ADDR devicePages, GM_ADDR query, GM_ADDR promoteFlag, GM_ADDR scores,
    GM_ADDR promotedPages, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(KvCacheHybridAttentionProtoTilingData);
    GET_TILING_DATA_WITH_STRUCT(KvCacheHybridAttentionProtoTilingData,
        tilingData, tiling);
    NsKvCacheHybridAttentionProto::KvCacheHybridAttentionProto op;
    op.Init(sourceKinds, sourceBlockIds, hostPages, devicePages, query,
        promoteFlag, scores, promotedPages, &tilingData);
    op.Process();
}
