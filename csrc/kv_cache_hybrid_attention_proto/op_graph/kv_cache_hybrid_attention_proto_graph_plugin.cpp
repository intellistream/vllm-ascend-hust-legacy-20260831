#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferDataTypeKvCacheHybridAttentionProto(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(KvCacheHybridAttentionProto)
    .InferDataType(InferDataTypeKvCacheHybridAttentionProto);
} // namespace ops
