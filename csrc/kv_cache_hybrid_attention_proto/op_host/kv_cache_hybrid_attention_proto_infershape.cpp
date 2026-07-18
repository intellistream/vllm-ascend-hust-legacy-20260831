#include "register/op_impl_registry.h"

namespace ge {
static graphStatus InferShapeKvCacheHybridAttentionProto(
    gert::InferShapeContext* context)
{
    const gert::Shape* sourceIds = context->GetInputShape(1);
    const gert::Shape* devicePages = context->GetInputShape(3);
    gert::Shape* scores = context->GetOutputShape(0);
    gert::Shape* promotedPages = context->GetOutputShape(1);
    if (sourceIds == nullptr || devicePages == nullptr || scores == nullptr ||
        promotedPages == nullptr) {
        return GRAPH_FAILED;
    }
    scores->SetDimNum(2);
    scores->SetDim(0, sourceIds->GetDim(0));
    scores->SetDim(1, 8);
    *promotedPages = *devicePages;
    promotedPages->SetDim(0, sourceIds->GetDim(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheHybridAttentionProto)
    .InferShape(InferShapeKvCacheHybridAttentionProto);
} // namespace ge
