#include "kernel_operator.h"

using namespace AscendC;

class RwkvShiftMix6DirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR xx,
        GM_ADDR mix1,
        GM_ADDR mix2,
        GM_ADDR mix3,
        GM_ADDR mix4,
        GM_ADDR mix5,
        GM_ADDR mix6,
        GM_ADDR y1,
        GM_ADDR y2,
        GM_ADDR y3,
        GM_ADDR y4,
        GM_ADDR y5,
        GM_ADDR y6,
        GM_ADDR y7,
        uint32_t elements) {
        elements_ = elements;
        const uint32_t output_index = GetBlockIdx();
        GM_ADDR mixes[7] = {mix1, mix2, mix3, mix4, mix5, mix6, mix3};
        GM_ADDR outputs[7] = {y1, y2, y3, y4, y5, y6, y7};
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        xx_gm_.SetGlobalBuffer((__gm__ half*)xx);
        mix_gm_.SetGlobalBuffer((__gm__ half*)mixes[output_index]);
        y_gm_.SetGlobalBuffer((__gm__ half*)outputs[output_index]);
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
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

extern "C" __global__ __aicore__ void rwkv_shift_mix6_direct(
    GM_ADDR x,
    GM_ADDR xx,
    GM_ADDR mix1,
    GM_ADDR mix2,
    GM_ADDR mix3,
    GM_ADDR mix4,
    GM_ADDR mix5,
    GM_ADDR mix6,
    GM_ADDR y1,
    GM_ADDR y2,
    GM_ADDR y3,
    GM_ADDR y4,
    GM_ADDR y5,
    GM_ADDR y6,
    GM_ADDR y7,
    uint32_t elements) {
    RwkvShiftMix6DirectKernel kernel;
    kernel.Init(
        x, xx, mix1, mix2, mix3, mix4, mix5, mix6,
        y1, y2, y3, y4, y5, y6, y7, elements);
    kernel.Process();
}
