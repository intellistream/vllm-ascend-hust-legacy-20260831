/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ops.h"

#include <stdexcept>
#include <string>

namespace vllm_ascend {
namespace {

[[noreturn]] void raise_unavailable_kernel(const char* name)
{
    throw std::runtime_error(std::string(name) +
                             " is unavailable because "
                             "VLLM_ASCEND_BUILD_ASCENDC_KERNELS=OFF");
}

}  // namespace

void get_masked_input_and_mask_impl(void* stream, void* input,
                                    void* masked_input, void* mask_out,
                                    const int64_t org_vocab_start_index,
                                    const int64_t org_vocab_end_index,
                                    const int64_t num_org_vocab_padding,
                                    const int64_t added_vocab_start_index,
                                    const int64_t added_vocab_end_index,
                                    const int64_t size,
                                    const uint32_t loop_cnt,
                                    const uint32_t aiv_num)
{
    raise_unavailable_kernel("get_masked_input_and_mask_impl");
}

void bgmv_shrink_impl(AscendType type, void* stream, void* x, void* weight,
                      void* indices, uint32_t indicesSize, void* y,
                      uint32_t batch_size, uint32_t num_tokens_per_core,
                      uint32_t input_hidden_dim, uint32_t lora_rank,
                      float scale)
{
    raise_unavailable_kernel("bgmv_shrink_impl");
}

void bgmv_expand_impl(AscendType type, void* stream, void* x, void* weight,
                      void* indices, uint32_t indicesSize, void* y,
                      void* y_out, uint32_t batch_size,
                      uint32_t num_tokens_per_core, uint32_t lora_rank,
                      uint32_t output_hidden_dim, uint32_t slice_offset,
                      uint32_t output_full_dim)
{
    raise_unavailable_kernel("bgmv_expand_impl");
}

void sgmv_shrink_impl(AscendType type, void* stream, void* x, void* weight,
                      void* loraIndices, uint32_t loraIndicesSize,
                      void* seqLen, uint32_t seqLenSize, void* y,
                      uint32_t batch_size, uint32_t num_tokens_per_core,
                      uint32_t input_hidden_dim, uint32_t lora_rank,
                      float scale)
{
    raise_unavailable_kernel("sgmv_shrink_impl");
}

void sgmv_expand_impl(AscendType type, void* stream, void* x, void* weight,
                      void* loraIndices, uint32_t loraIndicesSize,
                      void* seqLen, uint32_t seqLenSize, void* y,
                      void* y_out, uint32_t batch_size,
                      uint32_t num_tokens_per_core, uint32_t lora_rank,
                      uint32_t output_hidden_dim, uint32_t slice_offset,
                      uint32_t output_full_dim)
{
    raise_unavailable_kernel("sgmv_expand_impl");
}

void mla_preprocess_impl(void* stream, void* hidden_state, void* quant_scale1,
                         void* quant_offset1, void* wdqkv, void* bias1,
                         void* gamma2, void* beta2, void* quant_scale2,
                         void* quant_offset2, void* gamma3, void* sin1,
                         void* cos1, void* sin2, void* cos2, void* keycache,
                         void* slot_mapping, void* wuq, void* bias2,
                         void* wuk, void* descale1, void* descale2,
                         void* ctkv_scale, void* qnope_scale, void* q,
                         void* keycache_out, void* q2, void* keycache_out2,
                         void* inner_out, void* workspace, void* tiling,
                         const uint32_t block_dim)
{
    raise_unavailable_kernel("mla_preprocess_impl");
}

void batch_matmul_transpose_impl(void* stream, void* gm_a, void* gm_b,
                                 void* gm_c, void* gm_tiling_data,
                                 const uint32_t block_dim)
{
    raise_unavailable_kernel("batch_matmul_transpose_impl");
}

}  // namespace vllm_ascend
