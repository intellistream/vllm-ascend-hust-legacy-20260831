#ifndef KV_CACHE_HYBRID_ATTENTION_PROTO_H
#define KV_CACHE_HYBRID_ATTENTION_PROTO_H

#include "kernel_operator.h"
#include "kv_cache_hybrid_attention_proto_tiling_data.h"

namespace NsKvCacheHybridAttentionProto {
using namespace AscendC;

// Research-only consumer for the central data path of decode attention.  Each
// logical block is read either from mapped host memory or device GM, dotted
// with a query, and optionally copied into a compact device promotion cache.
// It deliberately omits softmax/value aggregation: the experiment asks whether
// host reads can be hidden inside a compute consumer and amortized after one
// promote-on-first-use token, not whether this is a production attention op.
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t SCORE_ALIGN_ELEMS = 8;

class KvCacheHybridAttentionProto {
public:
    __aicore__ inline void Init(GM_ADDR sourceKinds, GM_ADDR sourceBlockIds,
        GM_ADDR hostPages, GM_ADDR devicePages, GM_ADDR query,
        GM_ADDR promoteFlag, GM_ADDR scores, GM_ADDR promotedPages,
        const KvCacheHybridAttentionProtoTilingData* tilingData)
    {
        selectedBlocks_ = tilingData->selectedBlocks;
        elemsPerBlock_ = tilingData->elemsPerBlock;
        tileElems_ = tilingData->tileElems;
        sourceKindsGM_.SetGlobalBuffer((__gm__ int32_t*)sourceKinds, selectedBlocks_);
        sourceBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)sourceBlockIds, selectedBlocks_);
        hostPagesGM_.SetGlobalBuffer((__gm__ float*)hostPages);
        devicePagesGM_.SetGlobalBuffer((__gm__ float*)devicePages);
        queryGM_.SetGlobalBuffer((__gm__ float*)query, elemsPerBlock_);
        promoteFlagGM_.SetGlobalBuffer((__gm__ int32_t*)promoteFlag, 1);
        scoresGM_.SetGlobalBuffer((__gm__ float*)scores,
            selectedBlocks_ * SCORE_ALIGN_ELEMS);
        promotedPagesGM_.SetGlobalBuffer((__gm__ float*)promotedPages,
            selectedBlocks_ * elemsPerBlock_);
        pipe_.InitBuffer(sourceQueue_, BUFFER_NUM, tileElems_ * sizeof(float));
        pipe_.InitBuffer(queryQueue_, BUFFER_NUM, tileElems_ * sizeof(float));
        pipe_.InitBuffer(promotionQueue_, BUFFER_NUM, tileElems_ * sizeof(float));
        pipe_.InitBuffer(scoreQueue_, BUFFER_NUM,
            SCORE_ALIGN_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const bool promote = promoteFlagGM_.GetValue(0) != 0;
        for (int64_t pair = static_cast<int64_t>(GetBlockIdx());
             pair < selectedBlocks_;
             pair += static_cast<int64_t>(GetBlockNum())) {
            ConsumeBlock(pair, promote);
        }
    }

private:
    __aicore__ inline void ConsumeBlock(int64_t pair, bool promote)
    {
        const int32_t sourceKind = sourceKindsGM_.GetValue(pair);
        const int32_t sourceBlock = sourceBlockIdsGM_.GetValue(pair);
        const int64_t sourceBase = static_cast<int64_t>(sourceBlock) * elemsPerBlock_;
        const int64_t promotionBase = pair * elemsPerBlock_;
        float score = 0.0F;

        for (int64_t copied = 0; copied < elemsPerBlock_; copied += tileElems_) {
            const int64_t remaining = elemsPerBlock_ - copied;
            const int32_t copyElems = static_cast<int32_t>(
                remaining < tileElems_ ? remaining : tileElems_);

            LocalTensor<float> sourceLocal = sourceQueue_.AllocTensor<float>();
            if (sourceKind == 0) {
                DataCopy(sourceLocal, devicePagesGM_[sourceBase + copied], copyElems);
            } else {
                DataCopy(sourceLocal, hostPagesGM_[sourceBase + copied], copyElems);
            }
            sourceQueue_.EnQue(sourceLocal);

            LocalTensor<float> queryLocal = queryQueue_.AllocTensor<float>();
            DataCopy(queryLocal, queryGM_[copied], copyElems);
            queryQueue_.EnQue(queryLocal);

            sourceLocal = sourceQueue_.DeQue<float>();
            queryLocal = queryQueue_.DeQue<float>();

            // Preserve the unmodified source in a second UB queue before Mul.
            // This is correctness-first prototype plumbing; a production fused
            // kernel would pipeline tiles and write the already resident tile.
            if (promote) {
                LocalTensor<float> promotionLocal =
                    promotionQueue_.AllocTensor<float>();
                DataCopy(promotionLocal, sourceLocal, copyElems);
                promotionQueue_.EnQue(promotionLocal);
            }

            Mul(sourceLocal, sourceLocal, queryLocal, copyElems);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(sourceLocal, sourceLocal, sourceLocal, copyElems);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            score += sourceLocal.GetValue(0);

            if (promote) {
                LocalTensor<float> promotionLocal =
                    promotionQueue_.DeQue<float>();
                DataCopy(promotedPagesGM_[promotionBase + copied],
                    promotionLocal, copyElems);
                promotionQueue_.FreeTensor(promotionLocal);
            }
            queryQueue_.FreeTensor(queryLocal);
            sourceQueue_.FreeTensor(sourceLocal);
        }
        // GM DataCopy is 32-byte aligned; publish each scalar in an 8-float
        // lane rather than relying on scalar-pipeline GM SetValue stores.
        LocalTensor<float> scoreLocal = scoreQueue_.AllocTensor<float>();
        Duplicate(scoreLocal, score, SCORE_ALIGN_ELEMS);
        scoreQueue_.EnQue(scoreLocal);
        scoreLocal = scoreQueue_.DeQue<float>();
        DataCopy(scoresGM_[pair * SCORE_ALIGN_ELEMS], scoreLocal,
            SCORE_ALIGN_ELEMS);
        scoreQueue_.FreeTensor(scoreLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sourceQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> queryQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> promotionQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scoreQueue_;
    GlobalTensor<int32_t> sourceKindsGM_;
    GlobalTensor<int32_t> sourceBlockIdsGM_;
    GlobalTensor<int32_t> promoteFlagGM_;
    GlobalTensor<float> hostPagesGM_;
    GlobalTensor<float> devicePagesGM_;
    GlobalTensor<float> queryGM_;
    GlobalTensor<float> scoresGM_;
    GlobalTensor<float> promotedPagesGM_;
    int64_t selectedBlocks_ = 0;
    int64_t elemsPerBlock_ = 0;
    int64_t tileElems_ = 0;
};

} // namespace NsKvCacheHybridAttentionProto

#endif
