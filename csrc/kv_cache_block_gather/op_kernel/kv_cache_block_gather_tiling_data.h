#ifndef KV_CACHE_BLOCK_GATHER_TILING_DATA_H
#define KV_CACHE_BLOCK_GATHER_TILING_DATA_H

#include <cstdint>

struct KvCacheBlockGatherTilingData {
    int64_t selectedBlocks;
    int64_t elemsPerBlock;
    int64_t tileElems;
};

#endif
