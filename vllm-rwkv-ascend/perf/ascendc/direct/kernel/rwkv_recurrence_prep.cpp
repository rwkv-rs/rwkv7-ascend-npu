#include "kernel_operator.h"

using namespace AscendC;

// B=1, N=64 decode specialization.  One Vector Core owns one head and fuses
// the low-rank postprocess, a/k preparation, and value residual mixing.  The
// packed output rows are [w_exp, k, kk, a_sigmoid, v, g].
class RwkvRecurrencePrepDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR lowrank,
        GM_ADDR bias,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR v_first,
        GM_ADDR k_k,
        GM_ADDR k_a,
        GM_ADDR output,
        uint32_t hidden,
        uint32_t has_value_mix,
        uint32_t head_size) {
        hidden_ = hidden;
        n_ = head_size;
        has_value_mix_ = has_value_mix;
        const uint32_t head_offset = GetBlockIdx() * n_;
        for (uint32_t row = 0; row < 4; ++row) {
            lowrank_gm_[row].SetGlobalBuffer(
                (__gm__ half*)lowrank + row * hidden_ + head_offset);
            bias_gm_[row].SetGlobalBuffer(
                (__gm__ half*)bias + row * hidden_ + head_offset);
        }
        k_gm_.SetGlobalBuffer((__gm__ half*)k + head_offset);
        v_gm_.SetGlobalBuffer((__gm__ half*)v + head_offset);
        v_first_gm_.SetGlobalBuffer((__gm__ half*)v_first + head_offset);
        kk_scale_gm_.SetGlobalBuffer((__gm__ half*)k_k + head_offset);
        ka_scale_gm_.SetGlobalBuffer((__gm__ half*)k_a + head_offset);
        for (uint32_t row = 0; row < 6; ++row) {
            output_gm_[row].SetGlobalBuffer(
                (__gm__ half*)output + row * hidden_ + head_offset);
        }

        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        for (uint32_t row = 0; row < 4; ++row) {
            pipe_.InitBuffer(lowrank_buffer_[row], half_bytes);
            pipe_.InitBuffer(bias_buffer_[row], half_bytes);
        }
        pipe_.InitBuffer(k_buffer_, half_bytes);
        pipe_.InitBuffer(v_buffer_, half_bytes);
        pipe_.InitBuffer(v_first_buffer_, half_bytes);
        pipe_.InitBuffer(kk_scale_buffer_, half_bytes);
        pipe_.InitBuffer(ka_scale_buffer_, half_bytes);
        for (uint32_t row = 0; row < 6; ++row) {
            pipe_.InitBuffer(output_buffer_[row], half_bytes);
        }
        pipe_.InitBuffer(mid1_half_buffer_, half_bytes);
        pipe_.InitBuffer(mid2_half_buffer_, half_bytes);
        pipe_.InitBuffer(sigmoid_half_buffer_, 3 * half_bytes);
        pipe_.InitBuffer(norm_half_buffer_, half_bytes);
        pipe_.InitBuffer(float1_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(float2_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(one_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(sum_buffer_, 32);
        pipe_.InitBuffer(sum_half_buffer_, 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        LocalTensor<half> lowrank[4];
        LocalTensor<half> bias[4];
        LocalTensor<half> out[6];
        for (uint32_t row = 0; row < 4; ++row) {
            lowrank[row] = lowrank_buffer_[row].Get<half>();
            bias[row] = bias_buffer_[row].Get<half>();
            DataCopy(lowrank[row], lowrank_gm_[row], n_);
            DataCopy(bias[row], bias_gm_[row], n_);
        }
        for (uint32_t row = 0; row < 6; ++row) {
            out[row] = output_buffer_[row].Get<half>();
        }
        auto k = k_buffer_.Get<half>();
        auto v = v_buffer_.Get<half>();
        auto v_first = v_first_buffer_.Get<half>();
        auto kk_scale = kk_scale_buffer_.Get<half>();
        auto ka_scale = ka_scale_buffer_.Get<half>();
        auto mid1_half = mid1_half_buffer_.Get<half>();
        auto mid2_half = mid2_half_buffer_.Get<half>();
        auto sigmoid_half = sigmoid_half_buffer_.Get<half>();
        auto norm_half = norm_half_buffer_.Get<half>();
        auto float1 = float1_buffer_.Get<float>();
        auto float2 = float2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        auto sum = sum_buffer_.Get<float>();
        auto sum_half = sum_half_buffer_.Get<half>();

        DataCopy(k, k_gm_, n_);
        DataCopy(v, v_gm_, n_);
        if (has_value_mix_ != 0) DataCopy(v_first, v_first_gm_, n_);
        DataCopy(kk_scale, kk_scale_gm_, n_);
        DataCopy(ka_scale, ka_scale_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        // Batch the three sigmoid inputs so their fp32 vector instructions
        // share one barrier chain instead of serializing three independent
        // 64-element paths.
        Add(sigmoid_half, lowrank[0], bias[0], n_);
        Add(sigmoid_half[n_], lowrank[1], bias[1], n_);
        Add(sigmoid_half[2 * n_], lowrank[3], bias[3], n_);
        Add(out[5], lowrank[2], bias[2], n_);
        PipeBarrier<PIPE_V>();
        const uint32_t sigmoid_elements = 3 * n_;
        Cast(float1, sigmoid_half, RoundMode::CAST_NONE, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Muls(float2, float1, -1.0f, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Exp(float1, float2, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Adds(float2, float1, 1.0f, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Duplicate(one, 1.0f, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Div(float1, one, float2, sigmoid_elements);
        PipeBarrier<PIPE_V>();
        Cast(sigmoid_half, float1, RoundMode::CAST_RINT, sigmoid_elements);

        // w = exp(-0.606531 * sigmoid(...)); a and v-mix retain the fp16
        // rounding point used by the split reference kernels.
        PipeBarrier<PIPE_V>();
        Muls(out[0], sigmoid_half, static_cast<half>(-0.606531f), n_);
        Adds(out[3], sigmoid_half[n_], static_cast<half>(0.0f), n_);
        PipeBarrier<PIPE_V>();
        Cast(float1, out[0], RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        Exp(float2, float1, n_);
        PipeBarrier<PIPE_V>();
        Cast(out[0], float2, RoundMode::CAST_RINT, n_);

        // kk = normalize(k * k_k), exactly matching the half reduction route.
        Mul(norm_half, k, kk_scale, n_);
        PipeBarrier<PIPE_V>();
        Cast(float1, norm_half, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        Mul(float2, float1, float1, n_);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum, float2, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, 1);
        PipeBarrier<PIPE_V>();
        Cast(sum_half, sum, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(norm_half, sum_half.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Mul(out[2], k, kk_scale, n_);
        PipeBarrier<PIPE_V>();
        Div(out[2], out[2], norm_half, n_);

        // k = k * (1 + (a - 1) * k_a).
        Adds(mid1_half, out[3], static_cast<half>(-1.0f), n_);
        PipeBarrier<PIPE_V>();
        Mul(mid2_half, mid1_half, ka_scale, n_);
        PipeBarrier<PIPE_V>();
        Adds(mid1_half, mid2_half, static_cast<half>(1.0f), n_);
        PipeBarrier<PIPE_V>();
        Mul(out[1], k, mid1_half, n_);

        if (has_value_mix_ != 0) {
            Sub(mid1_half, v_first, v, n_);
            PipeBarrier<PIPE_V>();
            Mul(mid1_half, mid1_half, sigmoid_half[2 * n_], n_);
            PipeBarrier<PIPE_V>();
            Add(out[4], v, mid1_half, n_);
        } else {
            Adds(out[4], v, static_cast<half>(0.0f), n_);
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        for (uint32_t row = 0; row < 6; ++row) {
            DataCopy(output_gm_[row], out[row], n_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lowrank_buffer_[4], bias_buffer_[4];
    TBuf<TPosition::VECCALC> k_buffer_, v_buffer_, v_first_buffer_;
    TBuf<TPosition::VECCALC> kk_scale_buffer_, ka_scale_buffer_;
    TBuf<TPosition::VECCALC> output_buffer_[6];
    TBuf<TPosition::VECCALC> mid1_half_buffer_, mid2_half_buffer_;
    TBuf<TPosition::VECCALC> sigmoid_half_buffer_;
    TBuf<TPosition::VECCALC> norm_half_buffer_;
    TBuf<TPosition::VECCALC> float1_buffer_, float2_buffer_, one_buffer_;
    TBuf<TPosition::VECCALC> sum_buffer_, sum_half_buffer_;
    GlobalTensor<half> lowrank_gm_[4], bias_gm_[4], output_gm_[6];
    GlobalTensor<half> k_gm_, v_gm_, v_first_gm_;
    GlobalTensor<half> kk_scale_gm_, ka_scale_gm_;
    uint32_t hidden_, n_, has_value_mix_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_recurrence_prep_direct(
    GM_ADDR lowrank,
    GM_ADDR bias,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR v_first,
    GM_ADDR k_k,
    GM_ADDR k_a,
    GM_ADDR output,
    uint32_t hidden,
    uint32_t has_value_mix,
    uint32_t head_size) {
    RwkvRecurrencePrepDirectKernel kernel;
    kernel.Init(
        lowrank, bias, k, v, v_first, k_k, k_a, output, hidden,
        has_value_mix, head_size);
    kernel.Process();
}
