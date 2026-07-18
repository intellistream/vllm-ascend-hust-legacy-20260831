#include "register/op_def_registry.h"

namespace ops {
class KvCacheHybridAttentionProto : public OpDef {
public:
    explicit KvCacheHybridAttentionProto(const char* name) : OpDef(name)
    {
        this->Input("source_kinds").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("source_block_ids").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("host_pages").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("device_pages").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("promote_flag").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("promoted_pages").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "kv_cache_hybrid_attention_proto");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend910_95", config);
    }
};
OP_ADD(KvCacheHybridAttentionProto);
} // namespace ops
