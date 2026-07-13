#include "kernel_operator.h"

using namespace AscendC;

class RwkvHeadScaledAddDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR scale,
        GM_ADDR v,
        GM_ADDR out,
        uint32_t heads,
        uint32_t head_size) {
        heads_ = heads;
        head_size_ = head_size;
        elements_ = heads * head_size;
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        scale_gm_.SetGlobalBuffer((__gm__ half*)scale);
        v_gm_.SetGlobalBuffer((__gm__ half*)v);
        out_gm_.SetGlobalBuffer((__gm__ half*)out);
        const uint32_t bytes = ((elements_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(v_buffer_, bytes);
        pipe_.InitBuffer(mid_buffer_, bytes);
        pipe_.InitBuffer(out_buffer_, bytes);
        pipe_.InitBuffer(scale_buffer_, 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto v = v_buffer_.Get<half>();
        auto mid = mid_buffer_.Get<half>();
        auto out = out_buffer_.Get<half>();
        auto scale = scale_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        DataCopy(v, v_gm_, elements_);
        DataCopy(scale, scale_gm_, 16);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        for (uint32_t head = 0; head < heads_; ++head) {
            const uint32_t offset = head * head_size_;
            Muls(mid[offset], v[offset], scale.GetValue(head), head_size_);
        }
        PipeBarrier<PIPE_V>();
        Add(out, x, mid, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, v_buffer_, mid_buffer_, out_buffer_;
    TBuf<TPosition::VECCALC> scale_buffer_;
    GlobalTensor<half> x_gm_, scale_gm_, v_gm_, out_gm_;
    uint32_t heads_, head_size_, elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_head_scaled_add_direct(
    GM_ADDR x,
    GM_ADDR scale,
    GM_ADDR v,
    GM_ADDR out,
    uint32_t heads,
    uint32_t head_size) {
    RwkvHeadScaledAddDirectKernel kernel;
    kernel.Init(x, scale, v, out, heads, head_size);
    kernel.Process();
}
