#ifndef OPS_OP_PROTO_INC_KV_CACHE_HYBRID_ATTENTION_PROTO_H_
#define OPS_OP_PROTO_INC_KV_CACHE_HYBRID_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
REG_OP(KvCacheHybridAttentionProto)
    .INPUT(source_kinds, TensorType({DT_INT32}))
    .INPUT(source_block_ids, TensorType({DT_INT32}))
    .INPUT(host_pages, TensorType({DT_FLOAT}))
    .INPUT(device_pages, TensorType({DT_FLOAT}))
    .INPUT(query, TensorType({DT_FLOAT}))
    .INPUT(promote_flag, TensorType({DT_INT32}))
    .OUTPUT(scores, TensorType({DT_FLOAT}))
    .OUTPUT(promoted_pages, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(KvCacheHybridAttentionProto)
} // namespace ge

#endif
