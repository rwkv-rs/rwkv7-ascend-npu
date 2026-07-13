#include "kernel_operator.h"

using namespace AscendC;

// Activate the packed [w, a, g, v] rank-64 projections in one launch:
// tanh(w), identity(a), sigmoid(g), identity(v).
class RwkvLowRankActivateDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR out, uint32_t elements_per_row) {
        n_ = elements_per_row;
        row_ = GetBlockIdx() * 2;
        const uint32_t offset = row_ * n_;
        x_gm_.SetGlobalBuffer((__gm__ half*)x + offset);
        out_gm_.SetGlobalBuffer((__gm__ half*)out + offset);
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_half_buffer_, half_bytes);
        pipe_.InitBuffer(out_half_buffer_, half_bytes);
        pipe_.InitBuffer(float1_buffer_, float_bytes);
        pipe_.InitBuffer(float2_buffer_, float_bytes);
        pipe_.InitBuffer(one_buffer_, float_bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x_half = x_half_buffer_.Get<half>();
        auto out_half = out_half_buffer_.Get<half>();
        auto float1 = float1_buffer_.Get<float>();
        auto float2 = float2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        DataCopy(x_half, x_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        if (row_ == 0) {
            Cast(float1, x_half, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
            Tanh(float2, float1, n_);
            PipeBarrier<PIPE_V>();
            Cast(out_half, float2, RoundMode::CAST_RINT, n_);
        } else {
            Cast(float1, x_half, RoundMode::CAST_NONE, n_);
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
            Cast(out_half, float1, RoundMode::CAST_RINT, n_);
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out_half, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_half_buffer_, out_half_buffer_;
    TBuf<TPosition::VECCALC> float1_buffer_, float2_buffer_, one_buffer_;
    GlobalTensor<half> x_gm_, out_gm_;
    uint32_t n_, row_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_lowrank_activate_direct(
    GM_ADDR x, GM_ADDR out, uint32_t elements_per_row) {
    RwkvLowRankActivateDirectKernel kernel;
    kernel.Init(x, out, elements_per_row);
    kernel.Process();
}
