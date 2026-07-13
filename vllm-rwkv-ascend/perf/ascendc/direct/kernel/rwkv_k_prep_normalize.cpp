#include "kernel_operator.h"

using namespace AscendC;

class RwkvKPrepNormalizeDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR k,
        GM_ADDR a,
        GM_ADDR k_k,
        GM_ADDR k_a,
        GM_ADDR kk,
        GM_ADDR k_out,
        GM_ADDR a_out,
        uint32_t heads,
        uint32_t head_size) {
        heads_ = heads;
        head_size_ = head_size;
        hidden_ = heads * head_size;
        output_index_ = GetBlockIdx();
        const uint32_t half_bytes =
            ((hidden_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes =
            ((hidden_ + 7) / 8 * 8) * sizeof(float);
        if (output_index_ < heads_) {
            const uint32_t offset = output_index_ * head_size_;
            k_gm_.SetGlobalBuffer((__gm__ half*)k + offset);
            scale_gm_.SetGlobalBuffer((__gm__ half*)k_k + offset);
            out_gm_.SetGlobalBuffer((__gm__ half*)kk + offset);
            elements_ = head_size_;
        } else {
            k_gm_.SetGlobalBuffer((__gm__ half*)k);
            a_gm_.SetGlobalBuffer((__gm__ half*)a);
            scale_gm_.SetGlobalBuffer((__gm__ half*)k_a);
            out_gm_.SetGlobalBuffer((__gm__ half*)k_out);
            a_out_gm_.SetGlobalBuffer((__gm__ half*)a_out);
            elements_ = hidden_;
        }
        pipe_.InitBuffer(k_buffer_, half_bytes);
        pipe_.InitBuffer(a_buffer_, half_bytes);
        pipe_.InitBuffer(scale_buffer_, half_bytes);
        pipe_.InitBuffer(mid1_buffer_, half_bytes);
        pipe_.InitBuffer(mid2_buffer_, half_bytes);
        pipe_.InitBuffer(out_buffer_, half_bytes);
        pipe_.InitBuffer(float1_buffer_, float_bytes);
        pipe_.InitBuffer(float2_buffer_, float_bytes);
        pipe_.InitBuffer(one_buffer_, float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum_half_buffer_, 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto k = k_buffer_.Get<half>();
        auto a = a_buffer_.Get<half>();
        auto scale = scale_buffer_.Get<half>();
        auto mid1 = mid1_buffer_.Get<half>();
        auto mid2 = mid2_buffer_.Get<half>();
        auto out = out_buffer_.Get<half>();
        auto float1 = float1_buffer_.Get<float>();
        auto float2 = float2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum_half = sum_half_buffer_.Get<half>();
        DataCopy(k, k_gm_, elements_);
        DataCopy(scale, scale_gm_, elements_);
        if (output_index_ == heads_) DataCopy(a, a_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        if (output_index_ < heads_) {
            Mul(mid1, k, scale, elements_);
            PipeBarrier<PIPE_V>();
            Cast(float1, mid1, RoundMode::CAST_NONE, elements_);
            PipeBarrier<PIPE_V>();
            Mul(float2, float1, float1, elements_);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(sum, float2, 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Sqrt(sum, sum, 1);
            PipeBarrier<PIPE_V>();
            Cast(sum_half, sum, RoundMode::CAST_NONE, 1);
            PipeBarrier<PIPE_V>();
            Duplicate(mid2, sum_half.GetValue(0), elements_);
            PipeBarrier<PIPE_V>();
            Div(out, mid1, mid2, elements_);
        } else {
            Cast(float1, a, RoundMode::CAST_NONE, elements_);
            PipeBarrier<PIPE_V>();
            Muls(float2, float1, -1.0f, elements_);
            PipeBarrier<PIPE_V>();
            Exp(float1, float2, elements_);
            PipeBarrier<PIPE_V>();
            Adds(float2, float1, 1.0f, elements_);
            PipeBarrier<PIPE_V>();
            Duplicate(one, 1.0f, elements_);
            PipeBarrier<PIPE_V>();
            Div(float1, one, float2, elements_);
            PipeBarrier<PIPE_V>();
            Cast(a, float1, RoundMode::CAST_RINT, elements_);
            PipeBarrier<PIPE_V>();
            Adds(mid1, a, static_cast<half>(-1.0f), elements_);
            PipeBarrier<PIPE_V>();
            Mul(mid2, mid1, scale, elements_);
            PipeBarrier<PIPE_V>();
            Adds(mid1, mid2, static_cast<half>(1.0f), elements_);
            PipeBarrier<PIPE_V>();
            Mul(out, k, mid1, elements_);
        }
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        if (output_index_ == heads_) {
            DataCopy(a_out_gm_, a, elements_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> k_buffer_, a_buffer_, scale_buffer_;
    TBuf<TPosition::VECCALC> mid1_buffer_, mid2_buffer_, out_buffer_;
    TBuf<TPosition::VECCALC> float1_buffer_, float2_buffer_, one_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, sum_half_buffer_;
    GlobalTensor<half> k_gm_, a_gm_, scale_gm_, out_gm_, a_out_gm_;
    uint32_t heads_, head_size_, hidden_, output_index_, elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_k_prep_normalize_direct(
    GM_ADDR k,
    GM_ADDR a,
    GM_ADDR k_k,
    GM_ADDR k_a,
    GM_ADDR kk,
    GM_ADDR k_out,
    GM_ADDR a_out,
    uint32_t heads,
    uint32_t head_size) {
    RwkvKPrepNormalizeDirectKernel kernel;
    kernel.Init(k, a, k_k, k_a, kk, k_out, a_out, heads, head_size);
    kernel.Process();
}
