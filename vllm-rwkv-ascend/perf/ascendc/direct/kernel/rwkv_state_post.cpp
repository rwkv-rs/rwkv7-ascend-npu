#include "kernel_operator.h"

using namespace AscendC;

// Finish the RWKV state recurrence after the two matrix multiplications:
//   out = state * w + term2 + vk
// One vector core owns one head.  The target decode shape is N=64, so the
// complete head fits in UB and w can be reused for every state row.
class RwkvStatePostDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR state,
        GM_ADDR w,
        GM_ADDR term2,
        GM_ADDR vk,
        GM_ADDR out,
        GM_ADDR out_half,
        uint32_t head_size) {
        head_size_ = head_size;
        head_elements_ = head_size * head_size;
        const uint32_t head = GetBlockIdx();
        state_gm_.SetGlobalBuffer(
            (__gm__ float*)state + head * head_elements_);
        w_gm_.SetGlobalBuffer((__gm__ half*)w + head * head_size_);
        term2_gm_.SetGlobalBuffer(
            (__gm__ float*)term2 + head * head_elements_);
        vk_gm_.SetGlobalBuffer((__gm__ half*)vk + head * head_elements_);
        out_gm_.SetGlobalBuffer((__gm__ float*)out + head * head_elements_);
        out_half_gm_.SetGlobalBuffer(
            (__gm__ half*)out_half + head * head_elements_);

        const uint32_t float_bytes = head_elements_ * sizeof(float);
        const uint32_t half_bytes = head_elements_ * sizeof(half);
        const uint32_t w_half_bytes = ((head_size_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t w_float_bytes = ((head_size_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(state_buffer_, float_bytes);
        pipe_.InitBuffer(term2_buffer_, float_bytes);
        pipe_.InitBuffer(vk_half_buffer_, half_bytes);
        pipe_.InitBuffer(vk_float_buffer_, float_bytes);
        pipe_.InitBuffer(w_half_buffer_, w_half_bytes);
        pipe_.InitBuffer(w_float_buffer_, w_float_bytes);
        pipe_.InitBuffer(mid_buffer_, float_bytes);
        pipe_.InitBuffer(out_buffer_, float_bytes);
        pipe_.InitBuffer(out_half_buffer_, half_bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto state = state_buffer_.Get<float>();
        auto term2 = term2_buffer_.Get<float>();
        auto vk_half = vk_half_buffer_.Get<half>();
        auto vk_float = vk_float_buffer_.Get<float>();
        auto w_half = w_half_buffer_.Get<half>();
        auto w_float = w_float_buffer_.Get<float>();
        auto mid = mid_buffer_.Get<float>();
        auto out = out_buffer_.Get<float>();
        auto out_half = out_half_buffer_.Get<half>();

        DataCopy(state, state_gm_, head_elements_);
        DataCopy(term2, term2_gm_, head_elements_);
        DataCopy(vk_half, vk_gm_, head_elements_);
        DataCopy(w_half, w_gm_, head_size_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Cast(vk_float, vk_half, RoundMode::CAST_NONE, head_elements_);
        Cast(w_float, w_half, RoundMode::CAST_NONE, head_size_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < head_size_; ++row) {
            const uint32_t offset = row * head_size_;
            Mul(mid[offset], state[offset], w_float, head_size_);
        }
        PipeBarrier<PIPE_V>();
        Add(out, mid, term2, head_elements_);
        PipeBarrier<PIPE_V>();
        Add(out, out, vk_float, head_elements_);
        PipeBarrier<PIPE_V>();
        Cast(out_half, out, RoundMode::CAST_NONE, head_elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, head_elements_);
        DataCopy(out_half_gm_, out_half, head_elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> state_buffer_, term2_buffer_;
    TBuf<TPosition::VECCALC> vk_half_buffer_, vk_float_buffer_;
    TBuf<TPosition::VECCALC> w_half_buffer_, w_float_buffer_;
    TBuf<TPosition::VECCALC> mid_buffer_, out_buffer_, out_half_buffer_;
    GlobalTensor<float> state_gm_, term2_gm_, out_gm_;
    GlobalTensor<half> w_gm_, vk_gm_, out_half_gm_;
    uint32_t head_size_, head_elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_state_post_direct(
    GM_ADDR state,
    GM_ADDR w,
    GM_ADDR term2,
    GM_ADDR vk,
    GM_ADDR out,
    GM_ADDR out_half,
    uint32_t head_size) {
    RwkvStatePostDirectKernel kernel;
    kernel.Init(state, w, term2, vk, out, out_half, head_size);
    kernel.Process();
}
