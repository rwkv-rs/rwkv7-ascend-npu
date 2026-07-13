#include "kernel_operator.h"

using namespace AscendC;

class RwkvValueMixDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR v,
        GM_ADDR v_first,
        GM_ADDR mix,
        GM_ADDR out,
        uint32_t elements) {
        elements_ = elements;
        v_gm_.SetGlobalBuffer((__gm__ half*)v);
        first_gm_.SetGlobalBuffer((__gm__ half*)v_first);
        mix_gm_.SetGlobalBuffer((__gm__ half*)mix);
        out_gm_.SetGlobalBuffer((__gm__ half*)out);
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(v_buffer_, bytes);
        pipe_.InitBuffer(first_buffer_, bytes);
        pipe_.InitBuffer(mix_buffer_, bytes);
        pipe_.InitBuffer(mid_buffer_, bytes);
        pipe_.InitBuffer(out_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto v = v_buffer_.Get<half>();
        auto first = first_buffer_.Get<half>();
        auto mix = mix_buffer_.Get<half>();
        auto mid = mid_buffer_.Get<half>();
        auto out = out_buffer_.Get<half>();
        DataCopy(v, v_gm_, elements_);
        DataCopy(first, first_gm_, elements_);
        DataCopy(mix, mix_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Sub(mid, first, v, elements_);
        PipeBarrier<PIPE_V>();
        Mul(mid, mid, mix, elements_);
        PipeBarrier<PIPE_V>();
        Add(out, v, mid, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> v_buffer_, first_buffer_, mix_buffer_;
    TBuf<TPosition::VECCALC> mid_buffer_, out_buffer_;
    GlobalTensor<half> v_gm_, first_gm_, mix_gm_, out_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_value_mix_direct(
    GM_ADDR v,
    GM_ADDR v_first,
    GM_ADDR mix,
    GM_ADDR out,
    uint32_t elements) {
    RwkvValueMixDirectKernel kernel;
    kernel.Init(v, v_first, mix, out, elements);
    kernel.Process();
}
