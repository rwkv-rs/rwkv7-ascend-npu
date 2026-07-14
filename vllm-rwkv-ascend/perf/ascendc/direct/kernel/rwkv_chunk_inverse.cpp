#include "kernel_operator.h"

using namespace AscendC;

// Invert batches of (I - L), where L is a strict-lower fp32 matrix.  One
// vector core owns one matrix and keeps both L and its inverse in UB.
class RwkvChunkInverseDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR lower,
        GM_ADDR output,
        uint32_t matrix_size,
        uint32_t input_bf16,
        uint32_t output_bf16) {
        size_ = matrix_size;
        elements_ = size_ * size_;
        input_bf16_ = input_bf16 != 0;
        output_bf16_ = output_bf16 != 0;
        const uint32_t block = GetBlockIdx();
        lower_gm_.SetGlobalBuffer((__gm__ float*)lower + block * elements_);
        lower_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)lower + block * elements_);
        output_gm_.SetGlobalBuffer((__gm__ float*)output + block * elements_);
        output_bf16_gm_.SetGlobalBuffer(
            (__gm__ bfloat16_t*)output + block * elements_);
        pipe_.InitBuffer(lower_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(lower_bf16_buffer_, elements_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(inverse_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(output_bf16_buffer_, elements_ * sizeof(bfloat16_t));
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto lower = lower_buffer_.Get<float>();
        auto lower_bf16 = lower_bf16_buffer_.Get<bfloat16_t>();
        auto inverse = inverse_buffer_.Get<float>();
        if (input_bf16_) {
            DataCopy(lower_bf16, lower_bf16_gm_, elements_);
        } else {
            DataCopy(lower, lower_gm_, elements_);
        }
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        if (input_bf16_) {
            Cast(lower, lower_bf16, RoundMode::CAST_NONE, elements_);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(inverse, 0.0f, elements_);
        PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < size_; ++row) {
            inverse.SetValue(row * size_ + row, 1.0f);
            for (uint32_t source_row = 0; source_row < row; ++source_row) {
                Axpy(
                    inverse[row * size_],
                    inverse[source_row * size_],
                    lower.GetValue(row * size_ + source_row),
                    size_);
                PipeBarrier<PIPE_V>();
            }
        }

        if (output_bf16_) {
            auto output_bf16 = output_bf16_buffer_.Get<bfloat16_t>();
            Cast(output_bf16, inverse, RoundMode::CAST_RINT, elements_);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(output_ready_);
            WaitFlag<HardEvent::V_MTE3>(output_ready_);
            DataCopy(output_bf16_gm_, output_bf16, elements_);
        } else {
            SetFlag<HardEvent::V_MTE3>(output_ready_);
            WaitFlag<HardEvent::V_MTE3>(output_ready_);
            DataCopy(output_gm_, inverse, elements_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lower_buffer_, lower_bf16_buffer_;
    TBuf<TPosition::VECCALC> inverse_buffer_;
    TBuf<TPosition::VECCALC> output_bf16_buffer_;
    GlobalTensor<float> lower_gm_, output_gm_;
    GlobalTensor<bfloat16_t> lower_bf16_gm_;
    GlobalTensor<bfloat16_t> output_bf16_gm_;
    uint32_t size_, elements_;
    bool input_bf16_;
    bool output_bf16_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_chunk_inverse_direct(
    GM_ADDR lower,
    GM_ADDR output,
    uint32_t matrix_size,
    uint32_t input_bf16,
    uint32_t output_bf16) {
    RwkvChunkInverseDirectKernel kernel;
    kernel.Init(lower, output, matrix_size, input_bf16, output_bf16);
    kernel.Process();
}
