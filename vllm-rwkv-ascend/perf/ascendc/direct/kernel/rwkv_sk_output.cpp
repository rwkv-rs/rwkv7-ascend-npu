#include "kernel_operator.h"

using namespace AscendC;

class RwkvSkOutputDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR r, GM_ADDR k, GM_ADDR r_k, GM_ADDR v,
        GM_ADDR out, uint32_t head_size) {
        elements_ = head_size;
        const uint32_t offset = GetBlockIdx() * head_size;
        x_gm_.SetGlobalBuffer((__gm__ half*)x + offset);
        r_gm_.SetGlobalBuffer((__gm__ half*)r + offset);
        k_gm_.SetGlobalBuffer((__gm__ half*)k + offset);
        rk_gm_.SetGlobalBuffer((__gm__ half*)r_k + offset);
        v_gm_.SetGlobalBuffer((__gm__ half*)v + offset);
        out_gm_.SetGlobalBuffer((__gm__ half*)out + offset);
        const uint32_t bytes = ((head_size + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(r_buffer_, bytes);
        pipe_.InitBuffer(k_buffer_, bytes);
        pipe_.InitBuffer(rk_buffer_, bytes);
        pipe_.InitBuffer(v_buffer_, bytes);
        pipe_.InitBuffer(mid1_buffer_, bytes);
        pipe_.InitBuffer(mid2_buffer_, bytes);
        pipe_.InitBuffer(mid_float_buffer_, head_size * sizeof(float));
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum_half_buffer_, 32);
        pipe_.InitBuffer(tmp_buffer_, 4096);
        pipe_.InitBuffer(out_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto r = r_buffer_.Get<half>();
        auto k = k_buffer_.Get<half>();
        auto rk = rk_buffer_.Get<half>();
        auto v = v_buffer_.Get<half>();
        auto mid1 = mid1_buffer_.Get<half>();
        auto mid2 = mid2_buffer_.Get<half>();
        auto mid_float = mid_float_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum_half = sum_half_buffer_.Get<half>();
        auto tmp = tmp_buffer_.Get<float>();
        auto out = out_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        DataCopy(r, r_gm_, elements_);
        DataCopy(k, k_gm_, elements_);
        DataCopy(rk, rk_gm_, elements_);
        DataCopy(v, v_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Mul(mid1, r, k, elements_);
        PipeBarrier<PIPE_V>();
        Mul(mid2, mid1, rk, elements_);
        PipeBarrier<PIPE_V>();
        Cast(mid_float, mid2, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum, mid_float, tmp, static_cast<int32_t>(elements_));
        PipeBarrier<PIPE_V>();
        Cast(sum_half, sum, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Muls(mid1, v, sum_half.GetValue(0), elements_);
        PipeBarrier<PIPE_V>();
        Add(out, x, mid1, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, r_buffer_, k_buffer_, rk_buffer_;
    TBuf<TPosition::VECCALC> v_buffer_, mid1_buffer_, mid2_buffer_;
    TBuf<TPosition::VECCALC> mid_float_buffer_, sum_buffer_, sum_half_buffer_;
    TBuf<TPosition::VECCALC> tmp_buffer_, out_buffer_;
    GlobalTensor<half> x_gm_, r_gm_, k_gm_, rk_gm_, v_gm_, out_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_sk_output_direct(
    GM_ADDR x, GM_ADDR r, GM_ADDR k, GM_ADDR r_k, GM_ADDR v,
    GM_ADDR out, uint32_t head_size) {
    RwkvSkOutputDirectKernel kernel;
    kernel.Init(x, r, k, r_k, v, out, head_size);
    kernel.Process();
}
