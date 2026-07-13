#include "kernel_operator.h"

using namespace AscendC;

// Produce the normalized-key input and adapted key together:
//   kk_raw = k * k_k
//   k_out  = k * (1 + (a - 1) * k_a)
// The two independent outputs use separate vector cores.
class RwkvKPrepDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR k,
        GM_ADDR a,
        GM_ADDR k_k,
        GM_ADDR k_a,
        GM_ADDR kk_raw,
        GM_ADDR k_out,
        uint32_t elements) {
        elements_ = elements;
        output_index_ = GetBlockIdx();
        k_gm_.SetGlobalBuffer((__gm__ half*)k);
        if (output_index_ == 0) {
            scale_gm_.SetGlobalBuffer((__gm__ half*)k_k);
            out_gm_.SetGlobalBuffer((__gm__ half*)kk_raw);
        } else {
            a_gm_.SetGlobalBuffer((__gm__ half*)a);
            scale_gm_.SetGlobalBuffer((__gm__ half*)k_a);
            out_gm_.SetGlobalBuffer((__gm__ half*)k_out);
        }
        const uint32_t bytes = ((elements + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(k_buffer_, bytes);
        pipe_.InitBuffer(a_buffer_, bytes);
        pipe_.InitBuffer(scale_buffer_, bytes);
        pipe_.InitBuffer(mid1_buffer_, bytes);
        pipe_.InitBuffer(mid2_buffer_, bytes);
        pipe_.InitBuffer(out_buffer_, bytes);
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
        DataCopy(k, k_gm_, elements_);
        DataCopy(scale, scale_gm_, elements_);
        if (output_index_ != 0) DataCopy(a, a_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        if (output_index_ == 0) {
            Mul(out, k, scale, elements_);
        } else {
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
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> k_buffer_, a_buffer_, scale_buffer_;
    TBuf<TPosition::VECCALC> mid1_buffer_, mid2_buffer_, out_buffer_;
    GlobalTensor<half> k_gm_, a_gm_, scale_gm_, out_gm_;
    uint32_t elements_, output_index_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_k_prep_direct(
    GM_ADDR k,
    GM_ADDR a,
    GM_ADDR k_k,
    GM_ADDR k_a,
    GM_ADDR kk_raw,
    GM_ADDR k_out,
    uint32_t elements) {
    RwkvKPrepDirectKernel kernel;
    kernel.Init(k, a, k_k, k_a, kk_raw, k_out, elements);
    kernel.Process();
}
