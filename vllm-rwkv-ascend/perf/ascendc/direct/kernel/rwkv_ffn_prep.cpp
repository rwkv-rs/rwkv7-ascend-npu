#include "kernel_operator.h"

using namespace AscendC;

// Fuses the exact fp16 residual add, fp32 FFN LayerNorm, and fp16 shift-mix.
// Packed output rows are [residual_sum, normalized, mixed].
class RwkvFfnPrepDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR base,
        GM_ADDR add,
        GM_ADDR previous,
        GM_ADDR mix,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR output,
        float epsilon,
        float inv_n,
        uint32_t elements) {
        n_ = elements;
        epsilon_ = epsilon;
        inv_n_ = inv_n;
        base_gm_.SetGlobalBuffer((__gm__ half*)base);
        add_gm_.SetGlobalBuffer((__gm__ half*)add);
        previous_gm_.SetGlobalBuffer((__gm__ half*)previous);
        mix_gm_.SetGlobalBuffer((__gm__ half*)mix);
        weight_gm_.SetGlobalBuffer((__gm__ float*)weight);
        bias_gm_.SetGlobalBuffer((__gm__ float*)bias);
        for (uint32_t row = 0; row < 3; ++row) {
            output_gm_[row].SetGlobalBuffer((__gm__ half*)output + row * n_);
        }
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(base_half_buffer_, half_bytes);
        pipe_.InitBuffer(add_half_buffer_, half_bytes);
        pipe_.InitBuffer(previous_half_buffer_, half_bytes);
        pipe_.InitBuffer(mix_half_buffer_, half_bytes);
        pipe_.InitBuffer(sum_half_buffer_, half_bytes);
        pipe_.InitBuffer(norm_half_buffer_, half_bytes);
        pipe_.InitBuffer(delta_half_buffer_, half_bytes);
        pipe_.InitBuffer(mixed_half_buffer_, half_bytes);
        pipe_.InitBuffer(x_float_buffer_, float_bytes);
        pipe_.InitBuffer(weight_float_buffer_, float_bytes);
        pipe_.InitBuffer(bias_float_buffer_, float_bytes);
        pipe_.InitBuffer(mean_float_buffer_, float_bytes);
        pipe_.InitBuffer(center_float_buffer_, float_bytes);
        pipe_.InitBuffer(square_float_buffer_, float_bytes);
        pipe_.InitBuffer(std_float_buffer_, float_bytes);
        pipe_.InitBuffer(out_float_buffer_, float_bytes);
        pipe_.InitBuffer(reduce_sum_buffer_, 32);
        pipe_.InitBuffer(reduce_square_buffer_, 32);
        pipe_.InitBuffer(reduce_tmp_buffer_, 8192);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto base = base_half_buffer_.Get<half>();
        auto add = add_half_buffer_.Get<half>();
        auto previous = previous_half_buffer_.Get<half>();
        auto mix = mix_half_buffer_.Get<half>();
        auto residual = sum_half_buffer_.Get<half>();
        auto norm = norm_half_buffer_.Get<half>();
        auto delta = delta_half_buffer_.Get<half>();
        auto mixed = mixed_half_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        auto weight = weight_float_buffer_.Get<float>();
        auto bias = bias_float_buffer_.Get<float>();
        auto mean = mean_float_buffer_.Get<float>();
        auto center = center_float_buffer_.Get<float>();
        auto square = square_float_buffer_.Get<float>();
        auto std = std_float_buffer_.Get<float>();
        auto out_float = out_float_buffer_.Get<float>();
        auto sum = reduce_sum_buffer_.Get<float>();
        auto sum2 = reduce_square_buffer_.Get<float>();
        auto reduce_tmp = reduce_tmp_buffer_.Get<float>();

        DataCopy(base, base_gm_, n_);
        DataCopy(add, add_gm_, n_);
        DataCopy(previous, previous_gm_, n_);
        DataCopy(mix, mix_gm_, n_);
        DataCopy(weight, weight_gm_, n_);
        DataCopy(bias, bias_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Add(residual, base, add, n_);
        PipeBarrier<PIPE_V>();
        Cast(x_float, residual, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum, x_float, reduce_tmp, static_cast<int32_t>(n_));
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, inv_n_, 1);
        Duplicate(mean, sum.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Sub(center, x_float, mean, n_);
        PipeBarrier<PIPE_V>();
        Mul(square, center, center, n_);
        PipeBarrier<PIPE_V>();
        ReduceSum(sum2, square, reduce_tmp, static_cast<int32_t>(n_));
        PipeBarrier<PIPE_V>();
        Muls(sum2, sum2, inv_n_, 1);
        Adds(sum2, sum2, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sum2, sum2, 1);
        Duplicate(std, sum2.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Div(out_float, center, std, n_);
        PipeBarrier<PIPE_V>();
        Mul(out_float, out_float, weight, n_);
        PipeBarrier<PIPE_V>();
        Add(out_float, out_float, bias, n_);
        PipeBarrier<PIPE_V>();
        Cast(norm, out_float, RoundMode::CAST_RINT, n_);
        PipeBarrier<PIPE_V>();
        Sub(delta, previous, norm, n_);
        PipeBarrier<PIPE_V>();
        Mul(delta, delta, mix, n_);
        PipeBarrier<PIPE_V>();
        Add(mixed, norm, delta, n_);

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(output_gm_[0], residual, n_);
        DataCopy(output_gm_[1], norm, n_);
        DataCopy(output_gm_[2], mixed, n_);
        DataCopy(previous_gm_, norm, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> base_half_buffer_, add_half_buffer_;
    TBuf<TPosition::VECCALC> previous_half_buffer_, mix_half_buffer_;
    TBuf<TPosition::VECCALC> sum_half_buffer_, norm_half_buffer_;
    TBuf<TPosition::VECCALC> delta_half_buffer_, mixed_half_buffer_;
    TBuf<TPosition::VECCALC> x_float_buffer_, weight_float_buffer_;
    TBuf<TPosition::VECCALC> bias_float_buffer_, mean_float_buffer_;
    TBuf<TPosition::VECCALC> center_float_buffer_, square_float_buffer_;
    TBuf<TPosition::VECCALC> std_float_buffer_, out_float_buffer_;
    TBuf<TPosition::VECCALC> reduce_sum_buffer_, reduce_square_buffer_;
    TBuf<TPosition::VECCALC> reduce_tmp_buffer_;
    GlobalTensor<half> base_gm_, add_gm_, previous_gm_, mix_gm_;
    GlobalTensor<float> weight_gm_, bias_gm_;
    GlobalTensor<half> output_gm_[3];
    uint32_t n_;
    float epsilon_, inv_n_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_ffn_prep_direct(
    GM_ADDR base,
    GM_ADDR add,
    GM_ADDR previous,
    GM_ADDR mix,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR output,
    float epsilon,
    float inv_n,
    uint32_t elements) {
    RwkvFfnPrepDirectKernel kernel;
    kernel.Init(
        base, add, previous, mix, weight, bias, output, epsilon, inv_n,
        elements);
    kernel.Process();
}
