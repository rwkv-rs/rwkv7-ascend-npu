#include "kernel_operator.h"

using namespace AscendC;

// Apply the four DPLR causal masks in one launch.  One vector core owns one
// group matrix and reuses a single full-matrix UB tile for all four inputs.
class RwkvChunkMaskDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR a_qk, GM_ADDR a_qb, GM_ADDR a_ak, GM_ADDR a_ab,
        uint32_t matrix_size, uint32_t input_bf16) {
        size_ = matrix_size;
        elements_ = size_ * size_;
        input_bf16_ = input_bf16 != 0;
        const uint32_t block = GetBlockIdx();
        a_qk_gm_.SetGlobalBuffer(
            (__gm__ float*)a_qk + block * elements_, elements_);
        a_qb_gm_.SetGlobalBuffer(
            (__gm__ float*)a_qb + block * elements_, elements_);
        a_ak_gm_.SetGlobalBuffer(
            (__gm__ float*)a_ak + block * elements_, elements_);
        a_ab_gm_.SetGlobalBuffer(
            (__gm__ float*)a_ab + block * elements_, elements_);
        a_qk_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)a_qk + block * elements_, elements_);
        a_qb_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)a_qb + block * elements_, elements_);
        a_ak_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)a_ak + block * elements_, elements_);
        pipe_.InitBuffer(matrix_buffer_, 2 * elements_ * sizeof(float));
        pipe_.InitBuffer(
            matrix_bf16_buffer_, elements_ * sizeof(bfloat16_t));
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
        output_done_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE3_V>());
        input_reusable_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE2>());
    }

    __aicore__ inline void BuildMask(bool strict_lower) {
        auto storage = matrix_buffer_.Get<float>();
        auto mask = storage[elements_];
        Duplicate(mask, 0.0f, elements_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < size_; ++row) {
            const uint32_t keep = row + (strict_lower ? 0 : 1);
            if (keep > 0) {
                // Every row starts on a 32-byte boundary for supported C.
                Duplicate(mask[row * size_], 1.0f, keep);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOne(GlobalTensor<float>& matrix_gm) {
        auto matrix = matrix_buffer_.Get<float>();
        auto mask = matrix[elements_];
        DataCopy(matrix, matrix_gm, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Mul(matrix, matrix, mask, elements_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(matrix_gm, matrix, elements_);
        SetFlag<HardEvent::MTE3_V>(output_done_);
        WaitFlag<HardEvent::MTE3_V>(output_done_);
        SetFlag<HardEvent::V_MTE2>(input_reusable_);
        WaitFlag<HardEvent::V_MTE2>(input_reusable_);
    }

    __aicore__ inline void ProcessOneBf16(
        GlobalTensor<bfloat16_t>& matrix_gm) {
        auto matrix_bf16 = matrix_bf16_buffer_.Get<bfloat16_t>();
        auto matrix = matrix_buffer_.Get<float>();
        auto mask = matrix[elements_];
        DataCopy(matrix_bf16, matrix_gm, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Cast(matrix, matrix_bf16, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();
        Mul(matrix, matrix, mask, elements_);
        PipeBarrier<PIPE_V>();
        Cast(matrix_bf16, matrix, RoundMode::CAST_RINT, elements_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(matrix_gm, matrix_bf16, elements_);
        SetFlag<HardEvent::MTE3_V>(output_done_);
        WaitFlag<HardEvent::MTE3_V>(output_done_);
        SetFlag<HardEvent::V_MTE2>(input_reusable_);
        WaitFlag<HardEvent::V_MTE2>(input_reusable_);
    }

    __aicore__ inline void Process() {
        if (input_bf16_) {
            BuildMask(false);
            ProcessOneBf16(a_qk_bf16_gm_);
            ProcessOneBf16(a_qb_bf16_gm_);
            BuildMask(true);
            ProcessOneBf16(a_ak_bf16_gm_);
        } else {
            BuildMask(false);
            ProcessOne(a_qk_gm_);
            ProcessOne(a_qb_gm_);
            BuildMask(true);
            ProcessOne(a_ak_gm_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(output_done_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(input_reusable_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> matrix_buffer_, matrix_bf16_buffer_;
    GlobalTensor<float> a_qk_gm_, a_qb_gm_, a_ak_gm_, a_ab_gm_;
    GlobalTensor<bfloat16_t> a_qk_bf16_gm_, a_qb_bf16_gm_, a_ak_bf16_gm_;
    uint32_t size_, elements_;
    bool input_bf16_;
    event_t input_ready_, output_ready_, output_done_, input_reusable_;
};

extern "C" __global__ __aicore__ void rwkv_chunk_mask_direct(
    GM_ADDR a_qk, GM_ADDR a_qb, GM_ADDR a_ak, GM_ADDR a_ab,
    uint32_t matrix_size, uint32_t input_bf16) {
    RwkvChunkMaskDirectKernel kernel;
    kernel.Init(a_qk, a_qb, a_ak, a_ab, matrix_size, input_bf16);
    kernel.Process();
}
