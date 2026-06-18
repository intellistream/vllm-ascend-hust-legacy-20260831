#include "kv_cache_block_gather_compat.h"
#include "log/ops_log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeKvCacheBlockGather(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeKvCacheBlockGather");

    const gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_LOGD(context->GetNodeName(), "End InferShapeKvCacheBlockGather");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheBlockGather).InferShape(InferShapeKvCacheBlockGather);
} // namespace ops
