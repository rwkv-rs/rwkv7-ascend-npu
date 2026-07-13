#include "kernel_operator.h"

using namespace AscendC;

// B=1 decode entry: gather one embedding row, apply pre-LayerNorm, then apply
// layer-0 attention LayerNorm.  Output rows are [pre_norm, attn_norm].
class RwkvEmbeddingNorm2DirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR token_ids,
        GM_ADDR embedding,
        GM_ADDR pre_weight,
        GM_ADDR pre_bias,
        GM_ADDR attn_weight,
        GM_ADDR attn_bias,
        GM_ADDR previous,
        GM_ADDR output,
        float epsilon,
        float inv_n,
        uint32_t hidden) {
        n_ = hidden;
        epsilon_ = epsilon;
        inv_n_ = inv_n;
        token_gm_.SetGlobalBuffer((__gm__ int64_t*)token_ids);
        embedding_gm_.SetGlobalBuffer((__gm__ half*)embedding);
        pre_weight_gm_.SetGlobalBuffer((__gm__ float*)pre_weight);
        pre_bias_gm_.SetGlobalBuffer((__gm__ float*)pre_bias);
        attn_weight_gm_.SetGlobalBuffer((__gm__ float*)attn_weight);
        attn_bias_gm_.SetGlobalBuffer((__gm__ float*)attn_bias);
        previous_gm_.SetGlobalBuffer((__gm__ half*)previous);
        out_pre_gm_.SetGlobalBuffer((__gm__ half*)output);
        out_attn_gm_.SetGlobalBuffer((__gm__ half*)output + n_);
        out_previous_gm_.SetGlobalBuffer((__gm__ half*)output + 2 * n_);
        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(token_buffer_, 32);
        pipe_.InitBuffer(row_half_buffer_, half_bytes);
        pipe_.InitBuffer(pre_half_buffer_, half_bytes);
        pipe_.InitBuffer(attn_half_buffer_, half_bytes);
        pipe_.InitBuffer(previous_half_buffer_, half_bytes);
        pipe_.InitBuffer(pre_weight_buffer_, float_bytes);
        pipe_.InitBuffer(pre_bias_buffer_, float_bytes);
        pipe_.InitBuffer(attn_weight_buffer_, float_bytes);
        pipe_.InitBuffer(attn_bias_buffer_, float_bytes);
        pipe_.InitBuffer(x_float_buffer_, float_bytes);
        pipe_.InitBuffer(mean_float_buffer_, float_bytes);
        pipe_.InitBuffer(center_float_buffer_, float_bytes);
        pipe_.InitBuffer(square_float_buffer_, float_bytes);
        pipe_.InitBuffer(std_float_buffer_, float_bytes);
        pipe_.InitBuffer(out_float_buffer_, float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum2_buffer_, 32);
        pipe_.InitBuffer(reduce_tmp_buffer_, 8192);
        token_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_S>());
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void LayerNorm(
        LocalTensor<half>& input,
        LocalTensor<float>& weight,
        LocalTensor<float>& bias,
        LocalTensor<half>& output,
        LocalTensor<float>& x_float,
        LocalTensor<float>& mean_float,
        LocalTensor<float>& center_float,
        LocalTensor<float>& square_float,
        LocalTensor<float>& std_float,
        LocalTensor<float>& out_float,
        LocalTensor<float>& sum,
        LocalTensor<float>& sum2,
        LocalTensor<float>& reduce_tmp) {
        Cast(x_float, input, RoundMode::CAST_NONE, n_);
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
        Mul(out_float, out_float, weight, n_);
        PipeBarrier<PIPE_V>();
        Add(out_float, out_float, bias, n_);
        PipeBarrier<PIPE_V>();
        Cast(output, out_float, RoundMode::CAST_RINT, n_);
    }

    __aicore__ inline void Process() {
        auto token = token_buffer_.Get<int64_t>();
        auto row_half = row_half_buffer_.Get<half>();
        auto pre_half = pre_half_buffer_.Get<half>();
        auto attn_half = attn_half_buffer_.Get<half>();
        auto previous_half = previous_half_buffer_.Get<half>();
        auto pre_weight = pre_weight_buffer_.Get<float>();
        auto pre_bias = pre_bias_buffer_.Get<float>();
        auto attn_weight = attn_weight_buffer_.Get<float>();
        auto attn_bias = attn_bias_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto mean_float = mean_float_buffer_.Get<float>();
        auto center_float = center_float_buffer_.Get<float>();
        auto square_float = square_float_buffer_.Get<float>();
        auto std_float = std_float_buffer_.Get<float>();
        auto out_float = out_float_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum2 = sum2_buffer_.Get<float>();
        auto reduce_tmp = reduce_tmp_buffer_.Get<float>();

        DataCopy(token, token_gm_, 4);
        SetFlag<HardEvent::MTE2_S>(token_ready_);
        WaitFlag<HardEvent::MTE2_S>(token_ready_);
        const int64_t index = token.GetValue(0);
        DataCopy(row_half, embedding_gm_[index * n_], n_);
        DataCopy(pre_weight, pre_weight_gm_, n_);
        DataCopy(pre_bias, pre_bias_gm_, n_);
        DataCopy(attn_weight, attn_weight_gm_, n_);
        DataCopy(attn_bias, attn_bias_gm_, n_);
        DataCopy(previous_half, previous_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        LayerNorm(
            row_half, pre_weight, pre_bias, pre_half, x_float, mean_float,
            center_float, square_float, std_float, out_float, sum, sum2,
            reduce_tmp);
        PipeBarrier<PIPE_V>();
        LayerNorm(
            pre_half, attn_weight, attn_bias, attn_half, x_float, mean_float,
            center_float, square_float, std_float, out_float, sum, sum2,
            reduce_tmp);

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(out_pre_gm_, pre_half, n_);
        DataCopy(out_attn_gm_, attn_half, n_);
        DataCopy(out_previous_gm_, previous_half, n_);
        DataCopy(previous_gm_, attn_half, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(token_ready_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> token_buffer_;
    TBuf<TPosition::VECCALC> row_half_buffer_, pre_half_buffer_;
    TBuf<TPosition::VECCALC> attn_half_buffer_, previous_half_buffer_;
    TBuf<TPosition::VECCALC> pre_weight_buffer_, pre_bias_buffer_;
    TBuf<TPosition::VECCALC> attn_weight_buffer_, attn_bias_buffer_;
    TBuf<TPosition::VECCALC> x_float_buffer_, mean_float_buffer_;
    TBuf<TPosition::VECCALC> center_float_buffer_, square_float_buffer_;
    TBuf<TPosition::VECCALC> std_float_buffer_, out_float_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, sum2_buffer_, reduce_tmp_buffer_;
    GlobalTensor<int64_t> token_gm_;
    GlobalTensor<half> embedding_gm_, previous_gm_;
    GlobalTensor<half> out_pre_gm_, out_attn_gm_, out_previous_gm_;
    GlobalTensor<float> pre_weight_gm_, pre_bias_gm_;
    GlobalTensor<float> attn_weight_gm_, attn_bias_gm_;
    uint32_t n_;
    float epsilon_, inv_n_;
    event_t token_ready_, input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_embedding_norm2_direct(
    GM_ADDR token_ids,
    GM_ADDR embedding,
    GM_ADDR pre_weight,
    GM_ADDR pre_bias,
    GM_ADDR attn_weight,
    GM_ADDR attn_bias,
    GM_ADDR previous,
    GM_ADDR output,
    float epsilon,
    float inv_n,
    uint32_t hidden) {
    RwkvEmbeddingNorm2DirectKernel kernel;
    kernel.Init(
        token_ids, embedding, pre_weight, pre_bias, attn_weight, attn_bias,
        previous, output, epsilon, inv_n, hidden);
    kernel.Process();
}
