#include "kernel_operator.h"

using namespace AscendC;

// Match torch norm/div semantics for fp16 decode: cast input to fp32, square
// and reduce in fp32, cast the norm to fp16, then divide in fp16.
class RwkvNormalizeKDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR out, uint32_t head_size) {
        elements_ = head_size;
        const uint32_t offset = GetBlockIdx() * head_size;
        x_gm_.SetGlobalBuffer((__gm__ half*)x + offset);
        out_gm_.SetGlobalBuffer((__gm__ half*)out + offset);
        const uint32_t half_bytes =
            ((head_size + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes =
            ((head_size + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_buffer_, half_bytes);
        pipe_.InitBuffer(x_float_buffer_, float_bytes);
        pipe_.InitBuffer(square_buffer_, float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(tmp_buffer_, 4096);
        pipe_.InitBuffer(norm_half_buffer_, 32);
        pipe_.InitBuffer(norm_vector_buffer_, half_bytes);
        pipe_.InitBuffer(out_buffer_, half_bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x = x_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        auto square = square_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto tmp = tmp_buffer_.Get<float>();
        auto norm_half = norm_half_buffer_.Get<half>();
        auto norm_vector = norm_vector_buffer_.Get<half>();
        auto out = out_buffer_.Get<half>();
        DataCopy(x, x_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Cast(x_float, x, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();
        Mul(square, x_float, x_float, elements_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum, square, tmp, static_cast<int32_t>(elements_));
        PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, 1);
        PipeBarrier<PIPE_V>();
        Cast(norm_half, sum, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(norm_vector, norm_half.GetValue(0), elements_);
        PipeBarrier<PIPE_V>();
        Div(out, x, norm_vector, elements_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out, elements_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, x_float_buffer_, square_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, tmp_buffer_, norm_half_buffer_;
    TBuf<TPosition::VECCALC> norm_vector_buffer_;
    TBuf<TPosition::VECCALC> out_buffer_;
    GlobalTensor<half> x_gm_, out_gm_;
    uint32_t elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_normalize_k_direct(
    GM_ADDR x, GM_ADDR out, uint32_t head_size) {
    RwkvNormalizeKDirectKernel kernel;
    kernel.Init(x, out, head_size);
    kernel.Process();
}
