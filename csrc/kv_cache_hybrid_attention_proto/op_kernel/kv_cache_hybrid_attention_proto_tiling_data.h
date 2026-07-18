#ifndef KV_CACHE_HYBRID_ATTENTION_PROTO_TILING_DATA_H
#define KV_CACHE_HYBRID_ATTENTION_PROTO_TILING_DATA_H

#include <cstdint>

struct KvCacheHybridAttentionProtoTilingData {
    int64_t selectedBlocks;
    int64_t elemsPerBlock;
    int64_t tileElems;
};

#endif
