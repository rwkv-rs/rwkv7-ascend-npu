#include "kernel_operator.h"

using namespace AscendC;

// Post-process packed second-stage [w, a, g, v] projections:
// exp(-0.606531 * sigmoid(w+b)), a+b, g+b, sigmoid(v+b).
class RwkvLowRankPostDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR bias, GM_ADDR out, uint32_t elements_per_row) {
        n_ = elements_per_row;
        row_ = GetBlockIdx();
        const uint32_t offset = row_ * n_;
        x_gm_.SetGlobalBuffer((__gm__ half*)x + offset);
        bias_gm_.SetGlobalBuffer((__gm__ half*)bias + offset);
        out_gm_.SetGlobalBuffer((__gm__ half*)out + offset);
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_half_buffer_, half_bytes);
        pipe_.InitBuffer(bias_half_buffer_, half_bytes);
        pipe_.InitBuffer(mid_half_buffer_, half_bytes);
        pipe_.InitBuffer(out_half_buffer_, half_bytes);
        pipe_.InitBuffer(float1_buffer_, float_bytes);
        pipe_.InitBuffer(float2_buffer_, float_bytes);
        pipe_.InitBuffer(one_buffer_, float_bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Sigmoid(
        LocalTensor<half>& input,
        LocalTensor<half>& output,
        LocalTensor<float>& float1,
        LocalTensor<float>& float2,
        LocalTensor<float>& one) {
        Cast(float1, input, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        Muls(float2, float1, -1.0f, n_);
        PipeBarrier<PIPE_V>();
        Exp(float1, float2, n_);
        PipeBarrier<PIPE_V>();
        Adds(float2, float1, 1.0f, n_);
        PipeBarrier<PIPE_V>();
        Duplicate(one, 1.0f, n_);
        PipeBarrier<PIPE_V>();
        Div(float1, one, float2, n_);
        PipeBarrier<PIPE_V>();
        Cast(output, float1, RoundMode::CAST_RINT, n_);
    }

    __aicore__ inline void Process() {
        auto x_half = x_half_buffer_.Get<half>();
        auto bias_half = bias_half_buffer_.Get<half>();
        auto mid_half = mid_half_buffer_.Get<half>();
        auto out_half = out_half_buffer_.Get<half>();
        auto float1 = float1_buffer_.Get<float>();
        auto float2 = float2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        DataCopy(x_half, x_gm_, n_);
        DataCopy(bias_half, bias_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Add(mid_half, x_half, bias_half, n_);
        PipeBarrier<PIPE_V>();

        if (row_ == 0) {
            Sigmoid(mid_half, out_half, float1, float2, one);
            PipeBarrier<PIPE_V>();
            Muls(out_half, out_half, static_cast<half>(-0.606531f), n_);
            PipeBarrier<PIPE_V>();
            Cast(float1, out_half, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
            Exp(float2, float1, n_);
            PipeBarrier<PIPE_V>();
            Cast(out_half, float2, RoundMode::CAST_RINT, n_);
        } else if (row_ == 3) {
            Sigmoid(mid_half, out_half, float1, float2, one);
        } else {
            Adds(out_half, mid_half, static_cast<half>(0.0f), n_);
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out_half, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_half_buffer_, bias_half_buffer_;
    TBuf<TPosition::VECCALC> mid_half_buffer_, out_half_buffer_;
    TBuf<TPosition::VECCALC> float1_buffer_, float2_buffer_, one_buffer_;
    GlobalTensor<half> x_gm_, bias_gm_, out_gm_;
    uint32_t n_, row_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_lowrank_post_direct(
    GM_ADDR x, GM_ADDR bias, GM_ADDR out, uint32_t elements_per_row) {
    RwkvLowRankPostDirectKernel kernel;
    kernel.Init(x, bias, out, elements_per_row);
    kernel.Process();
}
