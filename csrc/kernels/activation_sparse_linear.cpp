/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "types.h"

namespace {

__aicore__ inline float AbsFloat(float value)
{
    return value < 0.0F ? -value : value;
}

template <typename scalar_t>
class ActivationSparsePack {
public:
    using X_T = scalar_t;

    __aicore__ inline ActivationSparsePack() {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* threshold,
                                __gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, uint32_t batch_size,
                                uint32_t input_dim, uint32_t block_dim,
                                bool threshold_per_row, bool inclusive)
    {
        batchSize_ = batch_size;
        inputDim_ = input_dim;
        blockDim_ = block_dim;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t row = block_idx; row < batchSize_; row += blockDim_) {
            float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                               : thresholdGm_.GetValue(0);
            uint64_t row_offset = static_cast<uint64_t>(row) * inputDim_;
            int32_t count = 0;
            for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
                X_T raw_value = xGm_.GetValue(row_offset + in_col);
                float x_value = static_cast<float>(raw_value);
                float magnitude = AbsFloat(x_value);
                bool active = inclusive_ ? (magnitude >= threshold)
                                         : (magnitude > threshold);
                if (active) {
                    uint64_t out_offset =
                        row_offset + static_cast<uint32_t>(count);
                    valuesGm_.SetValue(out_offset, raw_value);
                    indicesGm_.SetValue(out_offset, static_cast<int32_t>(in_col));
                    ++count;
                }
            }
            countsGm_.SetValue(row, count);
        }
    }

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    uint32_t batchSize_;
    uint32_t inputDim_;
    uint32_t blockDim_;
    bool thresholdPerRow_;
    bool inclusive_;
};

template <typename scalar_t>
class ActivationSparseLinearPacked {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    __aicore__ inline ActivationSparseLinearPacked() {}

    __aicore__ inline void Init(__gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, __gm__ void* weight,
                                __gm__ void* y, uint32_t input_dim,
                                uint32_t output_dim, uint32_t work_items,
                                uint32_t block_dim)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        workItems_ = work_items;
        blockDim_ = block_dim;

        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
        weightGm_.SetGlobalBuffer((__gm__ W_T*)weight);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t linear_idx = block_idx; linear_idx < workItems_;
             linear_idx += blockDim_) {
            uint32_t row = linear_idx / outputDim_;
            uint32_t out_col = linear_idx - row * outputDim_;
            int32_t nnz = countsGm_.GetValue(row);
            float acc = 0.0F;

            uint64_t packed_offset = static_cast<uint64_t>(row) * inputDim_;
            uint64_t w_offset = static_cast<uint64_t>(out_col) * inputDim_;
            for (int32_t nz_pos = 0; nz_pos < nnz; ++nz_pos) {
                uint64_t value_offset = packed_offset + static_cast<uint32_t>(nz_pos);
                int32_t in_col = indicesGm_.GetValue(value_offset);
                float x_value = static_cast<float>(valuesGm_.GetValue(value_offset));
                float w_value = static_cast<float>(
                    weightGm_.GetValue(w_offset + static_cast<uint32_t>(in_col)));
                acc += x_value * w_value;
            }

            yGm_.SetValue(linear_idx, static_cast<Y_T>(acc));
        }
    }

private:
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    AscendC::GlobalTensor<W_T> weightGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    uint32_t workItems_;
    uint32_t blockDim_;
};

template <typename scalar_t>
class ActivationSparseLinearPackedTransposed {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr uint32_t OUTPUT_TILE = 1024;

    __aicore__ inline ActivationSparseLinearPackedTransposed(AscendC::TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(__gm__ void* values, __gm__ void* indices,
                                __gm__ void* counts, __gm__ void* weight_t,
                                __gm__ void* y, uint32_t input_dim,
                                uint32_t output_dim, uint32_t tile_count,
                                uint32_t work_items, uint32_t block_dim)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        tileCount_ = tile_count;
        workItems_ = work_items;
        blockDim_ = block_dim;

        valuesGm_.SetGlobalBuffer((__gm__ X_T*)values);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        countsGm_.SetGlobalBuffer((__gm__ int32_t*)counts);
        weightTGm_.SetGlobalBuffer((__gm__ W_T*)weight_t);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);

        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, OUTPUT_TILE * sizeof(W_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, OUTPUT_TILE * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferW_, OUTPUT_TILE * sizeof(float));
        pipe_->InitBuffer(accBufferY_, OUTPUT_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t work_idx = block_idx; work_idx < workItems_;
             work_idx += blockDim_) {
            uint32_t row = work_idx / tileCount_;
            uint32_t tile_idx = work_idx - row * tileCount_;
            uint32_t out_start = tile_idx * OUTPUT_TILE;
            uint32_t tile_len = outputDim_ - out_start;
            if (tile_len > OUTPUT_TILE) {
                tile_len = OUTPUT_TILE;
            }
            ComputeTile(row, out_start, tile_len);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t row, uint32_t out_start,
                                       uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        Duplicate(acc, 0.0F, tile_len);
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t packed_offset = static_cast<uint64_t>(row) * inputDim_;
        int32_t nnz = countsGm_.GetValue(row);
        for (int32_t nz_pos = 0; nz_pos < nnz; ++nz_pos) {
            uint64_t value_offset = packed_offset + static_cast<uint32_t>(nz_pos);
            int32_t in_col = indicesGm_.GetValue(value_offset);
            float x_value = static_cast<float>(valuesGm_.GetValue(value_offset));
            CopyInW(static_cast<uint32_t>(in_col), out_start, tile_len);
            Compute(x_value, tile_len);
        }
        CopyOut(row, out_start, tile_len);
    }

    __aicore__ inline void CopyInW(uint32_t in_col, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        uint64_t weight_offset =
            static_cast<uint64_t>(in_col) * outputDim_ + out_start;
        AscendC::DataCopyPadExtParams<W_T> pad_params{false, 0, 0, 0};
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(W_T)),
            0,
            0,
            0,
        };
        DataCopyPad(wLocal, weightTGm_[weight_offset], copy_params, pad_params);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void Compute(float x_value, uint32_t tile_len)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, tile_len);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueW_.FreeTensor(wLocal);

        Muls(wTmp, wTmp, x_value, tile_len);
        AscendC::PipeBarrier<PIPE_V>();
        Add(acc, acc, wTmp, tile_len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t out_start,
                                   uint32_t tile_len)
    {
        AscendC::LocalTensor<float> acc = accBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yLocal, acc, AscendC::RoundMode::CAST_RINT, tile_len);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue<Y_T>(yLocal);

        yLocal = outQueueY_.DeQue<Y_T>();
        uint64_t y_offset = static_cast<uint64_t>(row) * outputDim_ + out_start;
        AscendC::DataCopyExtParams copy_params{
            1,
            static_cast<uint32_t>(tile_len * sizeof(Y_T)),
            0,
            0,
            0,
        };
        DataCopyPad(yGm_[y_offset], yLocal, copy_params);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_, accBufferY_;
    AscendC::GlobalTensor<X_T> valuesGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;
    AscendC::GlobalTensor<int32_t> countsGm_;
    AscendC::GlobalTensor<W_T> weightTGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    uint32_t tileCount_;
    uint32_t workItems_;
    uint32_t blockDim_;
};

template <typename scalar_t>
class ActivationSparseLinear {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    __aicore__ inline ActivationSparseLinear() {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight,
                                __gm__ void* threshold, __gm__ void* y,
                                uint32_t input_dim, uint32_t output_dim,
                                bool threshold_per_row, bool inclusive,
                                uint32_t work_items, uint32_t block_dim)
    {
        inputDim_ = input_dim;
        outputDim_ = output_dim;
        thresholdPerRow_ = threshold_per_row;
        inclusive_ = inclusive;
        workItems_ = work_items;
        blockDim_ = block_dim;

        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        weightGm_.SetGlobalBuffer((__gm__ W_T*)weight);
        thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold);
        yGm_.SetGlobalBuffer((__gm__ Y_T*)y);
    }

    __aicore__ inline void Process()
    {
        uint32_t block_idx = AscendC::GetBlockIdx();
        for (uint32_t linear_idx = block_idx; linear_idx < workItems_;
             linear_idx += blockDim_) {
            uint32_t row = linear_idx / outputDim_;
            uint32_t out_col = linear_idx - row * outputDim_;
            float threshold = thresholdPerRow_ ? thresholdGm_.GetValue(row)
                                               : thresholdGm_.GetValue(0);
            float acc = 0.0F;

            uint64_t x_offset = static_cast<uint64_t>(row) * inputDim_;
            uint64_t w_offset = static_cast<uint64_t>(out_col) * inputDim_;
            for (uint32_t in_col = 0; in_col < inputDim_; ++in_col) {
                float x_value =
                    static_cast<float>(xGm_.GetValue(x_offset + in_col));
                float magnitude = AbsFloat(x_value);
                bool active = inclusive_ ? (magnitude >= threshold)
                                         : (magnitude > threshold);
                if (active) {
                    float w_value = static_cast<float>(
                        weightGm_.GetValue(w_offset + in_col));
                    acc += x_value * w_value;
                }
            }

            yGm_.SetValue(linear_idx, static_cast<Y_T>(acc));
        }
    }

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> weightGm_;
    AscendC::GlobalTensor<float> thresholdGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    uint32_t inputDim_;
    uint32_t outputDim_;
    bool thresholdPerRow_;
    bool inclusive_;
    uint32_t workItems_;
    uint32_t blockDim_;
};

#define ACTIVATION_SPARSE_PACK_TYPE_DECLARE(TYPE)                                  \
    extern "C" __global__ __aicore__ void activation_sparse_pack_##TYPE(           \
        __gm__ void* x, __gm__ void* threshold, __gm__ void* values,               \
        __gm__ void* indices, __gm__ void* counts, uint32_t batch_size,            \
        uint32_t input_dim, uint32_t block_dim, bool threshold_per_row,            \
        bool inclusive)                                                           \
    {                                                                              \
        ActivationSparsePack<TYPE> op;                                             \
        op.Init(x, threshold, values, indices, counts, batch_size, input_dim,       \
                block_dim, threshold_per_row, inclusive);                          \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(TYPE)                         \
    extern "C" __global__ __aicore__ void activation_sparse_linear_packed_##TYPE(  \
        __gm__ void* values, __gm__ void* indices, __gm__ void* counts,            \
        __gm__ void* weight, __gm__ void* y, uint32_t input_dim,                   \
        uint32_t output_dim, uint32_t work_items, uint32_t block_dim)              \
    {                                                                              \
        ActivationSparseLinearPacked<TYPE> op;                                     \
        op.Init(values, indices, counts, weight, y, input_dim, output_dim,         \
                work_items, block_dim);                                            \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(TYPE)                       \
    extern "C" __global__ __aicore__ void                                          \
    activation_sparse_linear_packed_t_##TYPE(                                      \
        __gm__ void* values, __gm__ void* indices, __gm__ void* counts,            \
        __gm__ void* weight_t, __gm__ void* y, uint32_t input_dim,                 \
        uint32_t output_dim, uint32_t tile_count, uint32_t work_items,             \
        uint32_t block_dim)                                                        \
    {                                                                              \
        AscendC::TPipe pipe;                                                       \
        ActivationSparseLinearPackedTransposed<TYPE> op(&pipe);                    \
        op.Init(values, indices, counts, weight_t, y, input_dim, output_dim,       \
                tile_count, work_items, block_dim);                                \
        op.Process();                                                              \
    }

#define ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(TYPE)                                \
    extern "C" __global__ __aicore__ void activation_sparse_linear_##TYPE(         \
        __gm__ void* x, __gm__ void* weight, __gm__ void* threshold,               \
        __gm__ void* y, uint32_t input_dim, uint32_t output_dim,                   \
        bool threshold_per_row, bool inclusive, uint32_t work_items,               \
        uint32_t block_dim)                                                        \
    {                                                                              \
        ActivationSparseLinear<TYPE> op;                                           \
        op.Init(x, weight, threshold, y, input_dim, output_dim,                    \
                threshold_per_row, inclusive, work_items, block_dim);              \
        op.Process();                                                              \
    }

ACTIVATION_SPARSE_PACK_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(half)
ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ACTIVATION_SPARSE_PACK_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_PACKED_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_PACKED_T_TYPE_DECLARE(bfloat16_t)
ACTIVATION_SPARSE_LINEAR_TYPE_DECLARE(bfloat16_t)
#endif

} // namespace

namespace vllm_ascend {

extern void activation_sparse_pack_impl(AscendType type, void* stream, void* x,
                                        void* threshold, void* values,
                                        void* indices, void* counts,
                                        uint32_t batch_size,
                                        uint32_t input_dim,
                                        uint32_t block_dim,
                                        bool threshold_per_row,
                                        bool inclusive)
{
    if (batch_size == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_pack_half<<<block_dim, nullptr, stream>>>(
            x, threshold, values, indices, counts, batch_size, input_dim,
            block_dim, threshold_per_row, inclusive);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_pack_bfloat16_t<<<block_dim, nullptr, stream>>>(
            x, threshold, values, indices, counts, batch_size, input_dim,
            block_dim, threshold_per_row, inclusive);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_packed_impl(
    AscendType type, void* stream, void* values, void* indices, void* counts,
    void* weight, void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t output_dim, uint32_t block_dim)
{
    uint32_t work_items = batch_size * output_dim;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_packed_half<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight, y, input_dim, output_dim,
            work_items, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_packed_bfloat16_t<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight, y, input_dim, output_dim,
            work_items, block_dim);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_packed_t_impl(
    AscendType type, void* stream, void* values, void* indices, void* counts,
    void* weight_t, void* y, uint32_t batch_size, uint32_t input_dim,
    uint32_t output_dim, uint32_t block_dim)
{
    constexpr uint32_t output_tile = 1024;
    uint32_t tile_count = (output_dim + output_tile - 1) / output_tile;
    uint32_t work_items = batch_size * tile_count;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_packed_t_half<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight_t, y, input_dim, output_dim,
            tile_count, work_items, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_packed_t_bfloat16_t<<<block_dim, nullptr, stream>>>(
            values, indices, counts, weight_t, y, input_dim, output_dim,
            tile_count, work_items, block_dim);
#endif
    } else {
        return;
    }
}

extern void activation_sparse_linear_impl(AscendType type, void* stream, void* x,
                                          void* weight, void* threshold,
                                          void* y, uint32_t batch_size,
                                          uint32_t input_dim,
                                          uint32_t output_dim,
                                          bool threshold_per_row,
                                          bool inclusive,
                                          uint32_t block_dim)
{
    uint32_t work_items = batch_size * output_dim;
    if (work_items == 0 || block_dim == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        activation_sparse_linear_half<<<block_dim, nullptr, stream>>>(
            x, weight, threshold, y, input_dim, output_dim, threshold_per_row,
            inclusive, work_items, block_dim);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        activation_sparse_linear_bfloat16_t<<<block_dim, nullptr, stream>>>(
            x, weight, threshold, y, input_dim, output_dim, threshold_per_row,
            inclusive, work_items, block_dim);
#endif
    } else {
        return;
    }
}

} // namespace vllm_ascend
