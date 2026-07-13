#include "kernel_operator.h"

using namespace AscendC;

class RwkvReluSquareDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t elements) {
        elements_ = elements;
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        y_gm_.SetGlobalBuffer((__gm__ half*)y);
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(mid_buffer_, bytes);
        pipe_.InitBuffer(y_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto mid = mid_buffer_.Get<half>();
        auto y = y_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Maxs(mid, x, static_cast<half>(0.0f), elements_);
        PipeBarrier<PIPE_V>();
        Mul(y, mid, mid, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(y_gm_, y, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, mid_buffer_, y_buffer_;
    GlobalTensor<half> x_gm_, y_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_relu_square_direct(
    GM_ADDR x, GM_ADDR y, uint32_t elements) {
    RwkvReluSquareDirectKernel kernel;
    kernel.Init(x, y, elements);
    kernel.Process();
}
