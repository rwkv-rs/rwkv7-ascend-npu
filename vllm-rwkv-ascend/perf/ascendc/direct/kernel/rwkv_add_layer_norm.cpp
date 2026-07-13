#include "kernel_operator.h"

using namespace AscendC;

// B=1 decode tail: fuse the final FFN residual add with final LayerNorm.
class RwkvAddLayerNormDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR add,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR out,
        float epsilon,
        float inv_n,
        uint32_t elements) {
        n_ = elements;
        epsilon_ = epsilon;
        inv_n_ = inv_n;
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        add_gm_.SetGlobalBuffer((__gm__ half*)add);
        weight_gm_.SetGlobalBuffer((__gm__ float*)weight);
        bias_gm_.SetGlobalBuffer((__gm__ float*)bias);
        out_gm_.SetGlobalBuffer((__gm__ half*)out);
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_half_buffer_, half_bytes);
        pipe_.InitBuffer(add_half_buffer_, half_bytes);
        pipe_.InitBuffer(out_half_buffer_, half_bytes);
        pipe_.InitBuffer(x_float_buffer_, float_bytes);
        pipe_.InitBuffer(weight_float_buffer_, float_bytes);
        pipe_.InitBuffer(bias_float_buffer_, float_bytes);
        pipe_.InitBuffer(mean_float_buffer_, float_bytes);
        pipe_.InitBuffer(center_float_buffer_, float_bytes);
        pipe_.InitBuffer(square_float_buffer_, float_bytes);
        pipe_.InitBuffer(std_float_buffer_, float_bytes);
        pipe_.InitBuffer(out_float_buffer_, float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum2_buffer_, 32);
        pipe_.InitBuffer(reduce_tmp_buffer_, 8192);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x_half = x_half_buffer_.Get<half>();
        auto add_half = add_half_buffer_.Get<half>();
        auto out_half = out_half_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        auto weight_float = weight_float_buffer_.Get<float>();
        auto bias_float = bias_float_buffer_.Get<float>();
        auto mean_float = mean_float_buffer_.Get<float>();
        auto center_float = center_float_buffer_.Get<float>();
        auto square_float = square_float_buffer_.Get<float>();
        auto std_float = std_float_buffer_.Get<float>();
        auto out_float = out_float_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum2 = sum2_buffer_.Get<float>();
        auto reduce_tmp = reduce_tmp_buffer_.Get<float>();

        DataCopy(x_half, x_gm_, n_);
        DataCopy(add_half, add_gm_, n_);
        DataCopy(weight_float, weight_gm_, n_);
        DataCopy(bias_float, bias_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        Add(x_half, x_half, add_half, n_);
        PipeBarrier<PIPE_V>();
        Cast(x_float, x_half, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum, x_float, reduce_tmp, static_cast<int32_t>(n_));
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, inv_n_, 1);
        Duplicate(mean_float, sum.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Sub(center_float, x_float, mean_float, n_);
        PipeBarrier<PIPE_V>();
        Mul(square_float, center_float, center_float, n_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum2, square_float, reduce_tmp, static_cast<int32_t>(n_));
        PipeBarrier<PIPE_V>();
        Muls(sum2, sum2, inv_n_, 1);
        Adds(sum2, sum2, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sum2, sum2, 1);
        Duplicate(std_float, sum2.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Div(out_float, center_float, std_float, n_);
        PipeBarrier<PIPE_V>();
        Mul(out_float, out_float, weight_float, n_);
        PipeBarrier<PIPE_V>();
        Add(out_float, out_float, bias_float, n_);
        PipeBarrier<PIPE_V>();
        Cast(out_half, out_float, RoundMode::CAST_RINT, n_);
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, out_half, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_half_buffer_, add_half_buffer_;
    TBuf<TPosition::VECCALC> out_half_buffer_;
    TBuf<TPosition::VECCALC> x_float_buffer_, weight_float_buffer_;
    TBuf<TPosition::VECCALC> bias_float_buffer_, mean_float_buffer_;
    TBuf<TPosition::VECCALC> center_float_buffer_, square_float_buffer_;
    TBuf<TPosition::VECCALC> std_float_buffer_, out_float_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, sum2_buffer_, reduce_tmp_buffer_;
    GlobalTensor<half> x_gm_, add_gm_, out_gm_;
    GlobalTensor<float> weight_gm_, bias_gm_;
    uint32_t n_;
    float epsilon_, inv_n_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_add_layer_norm_direct(
    GM_ADDR x,
    GM_ADDR add,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR out,
    float epsilon,
    float inv_n,
    uint32_t elements) {
    RwkvAddLayerNormDirectKernel kernel;
    kernel.Init(x, add, weight, bias, out, epsilon, inv_n, elements);
    kernel.Process();
}
