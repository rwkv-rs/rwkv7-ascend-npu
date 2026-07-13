#include "kernel_operator.h"

using namespace AscendC;

// Fuse per-head GroupNorm affine with the RWKV-7 receptance/key bonus:
//   group_norm(x) + sum(r * k * r_k) * v
class RwkvGroupNormSkDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR r,
        GM_ADDR k,
        GM_ADDR r_k,
        GM_ADDR v,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR g,
        GM_ADDR out,
        float epsilon,
        float inv_n,
        uint32_t head_size) {
        n_ = head_size;
        epsilon_ = epsilon;
        inv_n_ = inv_n;
        const uint32_t offset = GetBlockIdx() * n_;
        x_gm_.SetGlobalBuffer((__gm__ half*)x + offset);
        r_gm_.SetGlobalBuffer((__gm__ half*)r + offset);
        k_gm_.SetGlobalBuffer((__gm__ half*)k + offset);
        rk_gm_.SetGlobalBuffer((__gm__ half*)r_k + offset);
        v_gm_.SetGlobalBuffer((__gm__ half*)v + offset);
        weight_gm_.SetGlobalBuffer((__gm__ float*)weight + offset);
        bias_gm_.SetGlobalBuffer((__gm__ float*)bias + offset);
        g_gm_.SetGlobalBuffer((__gm__ half*)g + offset);
        out_gm_.SetGlobalBuffer((__gm__ half*)out + offset);
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(x_half_buffer_, half_bytes);
        pipe_.InitBuffer(r_half_buffer_, half_bytes);
        pipe_.InitBuffer(k_half_buffer_, half_bytes);
        pipe_.InitBuffer(rk_half_buffer_, half_bytes);
        pipe_.InitBuffer(v_half_buffer_, half_bytes);
        pipe_.InitBuffer(g_half_buffer_, half_bytes);
        pipe_.InitBuffer(mid1_half_buffer_, half_bytes);
        pipe_.InitBuffer(mid2_half_buffer_, half_bytes);
        pipe_.InitBuffer(norm_half_buffer_, half_bytes);
        pipe_.InitBuffer(x_float_buffer_, float_bytes);
        pipe_.InitBuffer(center_float_buffer_, float_bytes);
        pipe_.InitBuffer(square_float_buffer_, float_bytes);
        pipe_.InitBuffer(mean_float_buffer_, float_bytes);
        pipe_.InitBuffer(std_float_buffer_, float_bytes);
        pipe_.InitBuffer(weight_float_buffer_, float_bytes);
        pipe_.InitBuffer(bias_float_buffer_, float_bytes);
        pipe_.InitBuffer(norm_float_buffer_, float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum2_buffer_, 32);
        pipe_.InitBuffer(sum_half_buffer_, 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto x_half = x_half_buffer_.Get<half>();
        auto r_half = r_half_buffer_.Get<half>();
        auto k_half = k_half_buffer_.Get<half>();
        auto rk_half = rk_half_buffer_.Get<half>();
        auto v_half = v_half_buffer_.Get<half>();
        auto g_half = g_half_buffer_.Get<half>();
        auto mid1_half = mid1_half_buffer_.Get<half>();
        auto mid2_half = mid2_half_buffer_.Get<half>();
        auto norm_half = norm_half_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        auto center_float = center_float_buffer_.Get<float>();
        auto square_float = square_float_buffer_.Get<float>();
        auto mean_float = mean_float_buffer_.Get<float>();
        auto std_float = std_float_buffer_.Get<float>();
        auto weight_float = weight_float_buffer_.Get<float>();
        auto bias_float = bias_float_buffer_.Get<float>();
        auto norm_float = norm_float_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum2 = sum2_buffer_.Get<float>();
        auto sum_half = sum_half_buffer_.Get<half>();

        DataCopy(x_half, x_gm_, n_);
        DataCopy(r_half, r_gm_, n_);
        DataCopy(k_half, k_gm_, n_);
        DataCopy(rk_half, rk_gm_, n_);
        DataCopy(v_half, v_gm_, n_);
        DataCopy(weight_float, weight_gm_, n_);
        DataCopy(bias_float, bias_gm_, n_);
        DataCopy(g_half, g_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Cast(x_float, x_half, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum, x_float, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, inv_n_, 1);
        Duplicate(mean_float, sum.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Sub(center_float, x_float, mean_float, n_);
        PipeBarrier<PIPE_V>();
        Mul(square_float, center_float, center_float, n_);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum2, square_float, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(sum2, sum2, inv_n_, 1);
        Adds(sum2, sum2, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sum2, sum2, 1);
        Duplicate(std_float, sum2.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Div(norm_float, center_float, std_float, n_);
        PipeBarrier<PIPE_V>();
        Mul(norm_float, norm_float, weight_float, n_);
        PipeBarrier<PIPE_V>();
        Add(norm_float, norm_float, bias_float, n_);
        PipeBarrier<PIPE_V>();
        Cast(norm_half, norm_float, RoundMode::CAST_RINT, n_);

        Mul(mid1_half, r_half, k_half, n_);
        PipeBarrier<PIPE_V>();
        Mul(mid2_half, mid1_half, rk_half, n_);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum_half, mid2_half, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Axpy(norm_half, v_half, sum_half.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Mul(norm_half, norm_half, g_half, n_);

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_gm_, norm_half, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_half_buffer_, r_half_buffer_, k_half_buffer_;
    TBuf<TPosition::VECCALC> rk_half_buffer_, v_half_buffer_;
    TBuf<TPosition::VECCALC> g_half_buffer_;
    TBuf<TPosition::VECCALC> mid1_half_buffer_, mid2_half_buffer_;
    TBuf<TPosition::VECCALC> norm_half_buffer_;
    TBuf<TPosition::VECCALC> x_float_buffer_, center_float_buffer_;
    TBuf<TPosition::VECCALC> square_float_buffer_, mean_float_buffer_;
    TBuf<TPosition::VECCALC> std_float_buffer_, weight_float_buffer_;
    TBuf<TPosition::VECCALC> bias_float_buffer_, norm_float_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, sum2_buffer_, sum_half_buffer_;
    GlobalTensor<half> x_gm_, r_gm_, k_gm_, rk_gm_, v_gm_;
    GlobalTensor<half> g_gm_, out_gm_;
    GlobalTensor<float> weight_gm_, bias_gm_;
    uint32_t n_;
    float epsilon_, inv_n_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_groupnorm_sk_direct(
    GM_ADDR x,
    GM_ADDR r,
    GM_ADDR k,
    GM_ADDR r_k,
    GM_ADDR v,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR g,
    GM_ADDR out,
    float epsilon,
    float inv_n,
    uint32_t head_size) {
    RwkvGroupNormSkDirectKernel kernel;
    kernel.Init(
        x, r, k, r_k, v, weight, bias, g, out, epsilon, inv_n, head_size);
    kernel.Process();
}
