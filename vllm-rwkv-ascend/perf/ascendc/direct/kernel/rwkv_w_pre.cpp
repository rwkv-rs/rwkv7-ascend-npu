#include "kernel_operator.h"

using namespace AscendC;

class RwkvWPreDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR out, uint32_t elements) {
        elements_ = elements;
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        out_gm_.SetGlobalBuffer((__gm__ half*)out);
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes =
            ((elements + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(mid1_buffer_, float_bytes);
        pipe_.InitBuffer(mid2_buffer_, float_bytes);
        pipe_.InitBuffer(one_buffer_, float_bytes);
        pipe_.InitBuffer(out_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto mid1 = mid1_buffer_.Get<float>();
        auto mid2 = mid2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        auto out = out_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Cast(mid1, x, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();
        Muls(mid2, mid1, -1.0f, elements_);
        PipeBarrier<PIPE_V>();
        Exp(mid1, mid2, elements_);
        PipeBarrier<PIPE_V>();
        Adds(mid2, mid1, 1.0f, elements_);
        PipeBarrier<PIPE_V>();
        Duplicate(one, 1.0f, elements_);
        PipeBarrier<PIPE_V>();
        Div(mid1, one, mid2, elements_);
        PipeBarrier<PIPE_V>();
        Cast(out, mid1, RoundMode::CAST_RINT, elements_);
        PipeBarrier<PIPE_V>();
        Muls(out, out, static_cast<half>(-0.606531f), elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, mid1_buffer_, mid2_buffer_;
    TBuf<TPosition::VECCALC> one_buffer_, out_buffer_;
    GlobalTensor<half> x_gm_, out_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_w_pre_direct(
    GM_ADDR x, GM_ADDR out, uint32_t elements) {
    RwkvWPreDirectKernel kernel;
    kernel.Init(x, out, elements);
    kernel.Process();
}
