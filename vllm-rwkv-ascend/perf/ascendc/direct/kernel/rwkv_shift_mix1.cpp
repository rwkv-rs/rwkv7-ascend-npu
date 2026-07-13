#include "kernel_operator.h"

using namespace AscendC;

class RwkvShiftMix1DirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR xx, GM_ADDR mix, GM_ADDR y, uint32_t elements) {
        elements_ = elements;
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        xx_gm_.SetGlobalBuffer((__gm__ half*)xx);
        mix_gm_.SetGlobalBuffer((__gm__ half*)mix);
        y_gm_.SetGlobalBuffer((__gm__ half*)y);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(xx_buffer_, bytes);
        pipe_.InitBuffer(mix_buffer_, bytes);
        pipe_.InitBuffer(delta_buffer_, bytes);
        pipe_.InitBuffer(mid_buffer_, bytes);
        pipe_.InitBuffer(y_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto xx = xx_buffer_.Get<half>();
        auto mix = mix_buffer_.Get<half>();
        auto delta = delta_buffer_.Get<half>();
        auto mid = mid_buffer_.Get<half>();
        auto y = y_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        DataCopy(xx, xx_gm_, elements_);
        DataCopy(mix, mix_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Sub(delta, xx, x, elements_);
        PipeBarrier<PIPE_V>();
        Mul(mid, delta, mix, elements_);
        PipeBarrier<PIPE_V>();
        Add(y, x, mid, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(y_gm_, y, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, xx_buffer_, mix_buffer_;
    TBuf<TPosition::VECCALC> delta_buffer_, mid_buffer_, y_buffer_;
    GlobalTensor<half> x_gm_, xx_gm_, mix_gm_, y_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_shift_mix1_direct(
    GM_ADDR x, GM_ADDR xx, GM_ADDR mix, GM_ADDR y, uint32_t elements) {
    RwkvShiftMix1DirectKernel kernel;
    kernel.Init(x, xx, mix, y, elements);
    kernel.Process();
}
