#ifndef KV_CACHE_BLOCK_GATHER_H
#define KV_CACHE_BLOCK_GATHER_H

#include "kernel_operator.h"
#include "kv_cache_block_gather_tiling_data.h"
#include "kv_cache_block_gather_tiling_key.h"

namespace NsKvCacheBlockGather {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KvCacheBlockGather {
public:
    __aicore__ inline KvCacheBlockGather() {}
    __aicore__ inline void Init(GM_ADDR srcBlockIds, GM_ADDR srcPages,
        GM_ADDR dstBlockIds, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling,
        const KvCacheBlockGatherTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyBlock(int64_t pairIdx);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<int32_t> srcBlockIdsGM_;
    GlobalTensor<int32_t> dstBlockIdsGM_;
    GlobalTensor<T> srcPagesGM_;
    GlobalTensor<T> outGM_;

    int64_t selectedBlocks_ = 0;
    int64_t elemsPerBlock_ = 0;
    int64_t tileElems_ = 0;
};

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Init(
    GM_ADDR srcBlockIds, GM_ADDR srcPages, GM_ADDR dstBlockIds, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling,
    const KvCacheBlockGatherTilingData* tilingData)
{
    selectedBlocks_ = tilingData->selectedBlocks;
    elemsPerBlock_ = tilingData->elemsPerBlock;
    tileElems_ = tilingData->tileElems;
    (void)workspace;
    (void)tiling;

    srcBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds, selectedBlocks_);
    dstBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds, selectedBlocks_);
    srcPagesGM_.SetGlobalBuffer((__gm__ T*)srcPages);
    outGM_.SetGlobalBuffer((__gm__ T*)out);

    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::CopyBlock(int64_t pairIdx)
{
    int32_t srcPageIdx = srcBlockIdsGM_.GetValue(pairIdx);
    int32_t dstPageIdx = dstBlockIdsGM_.GetValue(pairIdx);
    int64_t srcBase = static_cast<int64_t>(srcPageIdx) * elemsPerBlock_;
    int64_t dstBase = static_cast<int64_t>(dstPageIdx) * elemsPerBlock_;

    int64_t copied = 0;
    while (copied < elemsPerBlock_) {
        int64_t remain = elemsPerBlock_ - copied;
        int64_t copyElems = remain < tileElems_ ? remain : tileElems_;

        LocalTensor<T> inLocal = inputQueue_.AllocTensor<T>();
        DataCopy(inLocal, srcPagesGM_[srcBase + copied], copyElems);
        inputQueue_.EnQue(inLocal);

        inLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outputQueue_.AllocTensor<T>();
        DataCopy(outLocal, inLocal, copyElems);
        outputQueue_.EnQue(outLocal);
        inputQueue_.FreeTensor(inLocal);

        outLocal = outputQueue_.DeQue<T>();
        DataCopy(outGM_[dstBase + copied], outLocal, copyElems);
        outputQueue_.FreeTensor(outLocal);

        copied += copyElems;
    }
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Process()
{
    for (int64_t pair = static_cast<int64_t>(AscendC::GetBlockIdx());
         pair < selectedBlocks_;
         pair += static_cast<int64_t>(AscendC::GetBlockNum())) {
        CopyBlock(pair);
    }
}

} // namespace NsKvCacheBlockGather

#endif
