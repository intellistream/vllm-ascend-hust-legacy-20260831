#ifndef OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_
#define OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(KvCacheBlockGather)
    .INPUT(src_block_ids, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .INPUT(src_pages, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(dst_block_ids, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(KvCacheBlockGather)

} // namespace ge

#endif // OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_
