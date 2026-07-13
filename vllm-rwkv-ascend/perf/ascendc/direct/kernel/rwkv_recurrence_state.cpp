#include "kernel_operator.h"

using namespace AscendC;

// B=1, N=64 decode specialization that fuses recurrence preparation with the
// rank-one fp32 state update.  Two row blocks share a head by redundantly
// preparing its 64-element vectors; this avoids unsafe cross-block barriers
// and removes the standalone prep launch plus w/kk/a global-memory traffic.
class RwkvRecurrenceStateDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR state,
        GM_ADDR lowrank,
        GM_ADDR bias,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR v_first,
        GM_ADDR k_k,
        GM_ADDR k_a,
        GM_ADDR r,
        GM_ADDR state_out,
        GM_ADDR output,
        GM_ADDR prepared,
        uint32_t hidden,
        uint32_t has_value_mix,
        uint32_t row_blocks,
        uint32_t head_size) {
        hidden_ = hidden;
        n_ = head_size;
        nn_ = n_ * n_;
        has_value_mix_ = has_value_mix;
        const uint32_t block = GetBlockIdx();
        head_ = block / row_blocks;
        row_block_ = block % row_blocks;
        const uint32_t rows_per_block = (n_ + row_blocks - 1) / row_blocks;
        row_start_ = row_block_ * rows_per_block;
        const uint32_t remaining = n_ - row_start_;
        row_count_ = remaining < rows_per_block ? remaining : rows_per_block;
        elements_ = row_count_ * n_;
        const uint32_t head_offset = head_ * n_;

        state_gm_.SetGlobalBuffer(
            (__gm__ float*)state + head_ * nn_ + row_start_ * n_);
        state_out_gm_.SetGlobalBuffer(
            (__gm__ float*)state_out + head_ * nn_ + row_start_ * n_);
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
        r_gm_.SetGlobalBuffer((__gm__ half*)r + head_offset);
        output_gm_.SetGlobalBuffer(
            (__gm__ half*)output + head_offset + row_start_);
        for (uint32_t row = 0; row < 3; ++row) {
            prepared_gm_[row].SetGlobalBuffer(
                (__gm__ half*)prepared + row * hidden_ + head_offset);
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
        pipe_.InitBuffer(r_buffer_, half_bytes);
        pipe_.InitBuffer(prepared_buffer_, 6 * half_bytes);
        pipe_.InitBuffer(sigmoid_buffer_, 3 * half_bytes);
        pipe_.InitBuffer(mid1_buffer_, half_bytes);
        pipe_.InitBuffer(mid2_buffer_, half_bytes);
        pipe_.InitBuffer(norm_buffer_, half_bytes);
        pipe_.InitBuffer(float1_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(float2_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(one_buffer_, 3 * float_bytes);
        pipe_.InitBuffer(norm_sum_buffer_, 32);
        pipe_.InitBuffer(norm_sum_half_buffer_, 32);
        pipe_.InitBuffer(state_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_out_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_half_buffer_, elements_ * sizeof(half));
        pipe_.InitBuffer(product_half_buffer_, elements_ * sizeof(half));
        const uint32_t output_bytes =
            ((row_count_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(output_buffer_, output_bytes);
        pipe_.InitBuffer(state_sum_buffer_, row_count_ * 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        LocalTensor<half> lowrank[4];
        LocalTensor<half> bias[4];
        for (uint32_t row = 0; row < 4; ++row) {
            lowrank[row] = lowrank_buffer_[row].Get<half>();
            bias[row] = bias_buffer_[row].Get<half>();
            DataCopy(lowrank[row], lowrank_gm_[row], n_);
            DataCopy(bias[row], bias_gm_[row], n_);
        }
        auto k_raw = k_buffer_.Get<half>();
        auto v_raw = v_buffer_.Get<half>();
        auto v_first = v_first_buffer_.Get<half>();
        auto kk_scale = kk_scale_buffer_.Get<half>();
        auto ka_scale = ka_scale_buffer_.Get<half>();
        auto r = r_buffer_.Get<half>();
        auto prepared = prepared_buffer_.Get<half>();
        auto w = prepared;
        auto k_out = prepared[n_];
        auto kk = prepared[2 * n_];
        auto a = prepared[3 * n_];
        auto v = prepared[4 * n_];
        auto g = prepared[5 * n_];
        auto sigmoid = sigmoid_buffer_.Get<half>();
        auto mid1 = mid1_buffer_.Get<half>();
        auto mid2 = mid2_buffer_.Get<half>();
        auto norm = norm_buffer_.Get<half>();
        auto float1 = float1_buffer_.Get<float>();
        auto float2 = float2_buffer_.Get<float>();
        auto one = one_buffer_.Get<float>();
        auto norm_sum = norm_sum_buffer_.Get<float>();
        auto norm_sum_half = norm_sum_half_buffer_.Get<half>();
        auto state = state_buffer_.Get<float>();
        auto state_out = state_out_buffer_.Get<float>();
        auto state_half = state_half_buffer_.Get<half>();
        auto product_half = product_half_buffer_.Get<half>();
        auto output = output_buffer_.Get<half>();
        auto state_sum = state_sum_buffer_.Get<half>();

        DataCopy(k_raw, k_gm_, n_);
        DataCopy(v_raw, v_gm_, n_);
        if (has_value_mix_ != 0) DataCopy(v_first, v_first_gm_, n_);
        DataCopy(kk_scale, kk_scale_gm_, n_);
        DataCopy(ka_scale, ka_scale_gm_, n_);
        DataCopy(r, r_gm_, n_);
        DataCopy(state, state_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Add(sigmoid, lowrank[0], bias[0], n_);
        Add(sigmoid[n_], lowrank[1], bias[1], n_);
        Add(sigmoid[2 * n_], lowrank[3], bias[3], n_);
        Add(g, lowrank[2], bias[2], n_);
        PipeBarrier<PIPE_V>();
        const uint32_t sigmoid_elements = 3 * n_;
        Cast(float1, sigmoid, RoundMode::CAST_NONE, sigmoid_elements);
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
        Cast(sigmoid, float1, RoundMode::CAST_RINT, sigmoid_elements);
        PipeBarrier<PIPE_V>();

        Muls(w, sigmoid, static_cast<half>(-0.606531f), n_);
        Adds(a, sigmoid[n_], static_cast<half>(0.0f), n_);
        PipeBarrier<PIPE_V>();
        Cast(float1, w, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        Exp(float2, float1, n_);
        PipeBarrier<PIPE_V>();
        Cast(w, float2, RoundMode::CAST_RINT, n_);

        Mul(norm, k_raw, kk_scale, n_);
        PipeBarrier<PIPE_V>();
        Cast(float1, norm, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        Mul(float2, float1, float1, n_);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(norm_sum, float2, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(norm_sum, norm_sum, 1);
        PipeBarrier<PIPE_V>();
        Cast(norm_sum_half, norm_sum, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(norm, norm_sum_half.GetValue(0), n_);
        PipeBarrier<PIPE_V>();
        Mul(kk, k_raw, kk_scale, n_);
        PipeBarrier<PIPE_V>();
        Div(kk, kk, norm, n_);

        Adds(mid1, a, static_cast<half>(-1.0f), n_);
        PipeBarrier<PIPE_V>();
        Mul(mid2, mid1, ka_scale, n_);
        PipeBarrier<PIPE_V>();
        Adds(mid1, mid2, static_cast<half>(1.0f), n_);
        PipeBarrier<PIPE_V>();
        Mul(k_out, k_raw, mid1, n_);

        if (has_value_mix_ != 0) {
            Sub(mid1, v_first, v_raw, n_);
            PipeBarrier<PIPE_V>();
            Mul(mid1, mid1, sigmoid[2 * n_], n_);
            PipeBarrier<PIPE_V>();
            Add(v, v_raw, mid1, n_);
        } else {
            Adds(v, v_raw, static_cast<half>(0.0f), n_);
        }
        PipeBarrier<PIPE_V>();

        Muls(mid1, kk, static_cast<half>(-1.0f), n_);
        Mul(mid2, kk, a, n_);
        Cast(state_half, state, RoundMode::CAST_RINT, elements_);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams half_rows(1, 1, 1, 4, 4, 0);
        Mul(
            product_half, state_half, mid1, 64,
            static_cast<uint8_t>(row_count_), half_rows);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            WholeReduceSum(
                state_sum[row * 16], product_half[row * n_],
                64, 1, 1, 1, 4);
        }
        PipeBarrier<PIPE_V>();
        Cast(float1, w, RoundMode::CAST_NONE, n_);
        Cast(float2, mid2, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams float_rows(1, 1, 1, 8, 8, 0);
        Mul(
            state_out, state, float1, 64,
            static_cast<uint8_t>(row_count_), float_rows);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            Axpy(
                state_out[row * n_], float2,
                static_cast<float>(state_sum.GetValue(row * 16)), n_);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            Axpy(
                state_out[row * n_], k_out,
                v.GetValue(row_start_ + row), n_);
        }
        PipeBarrier<PIPE_V>();
        Cast(state_half, state_out, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();
        Mul(
            product_half, state_half, r, 64,
            static_cast<uint8_t>(row_count_), half_rows);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            WholeReduceSum(
                state_sum[row * 16], product_half[row * n_],
                64, 1, 1, 1, 4);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            output.SetValue(row, state_sum.GetValue(row * 16));
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(state_out_gm_, state_out, elements_);
        DataCopy(output_gm_, output, row_count_);
        if (row_block_ == 0) {
            DataCopy(prepared_gm_[0], k_out, n_);
            DataCopy(prepared_gm_[1], v, n_);
            DataCopy(prepared_gm_[2], g, n_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lowrank_buffer_[4], bias_buffer_[4];
    TBuf<TPosition::VECCALC> k_buffer_, v_buffer_, v_first_buffer_;
    TBuf<TPosition::VECCALC> kk_scale_buffer_, ka_scale_buffer_, r_buffer_;
    TBuf<TPosition::VECCALC> prepared_buffer_, sigmoid_buffer_;
    TBuf<TPosition::VECCALC> mid1_buffer_, mid2_buffer_, norm_buffer_;
    TBuf<TPosition::VECCALC> float1_buffer_, float2_buffer_, one_buffer_;
    TBuf<TPosition::VECCALC> norm_sum_buffer_, norm_sum_half_buffer_;
    TBuf<TPosition::VECCALC> state_buffer_, state_out_buffer_;
    TBuf<TPosition::VECCALC> state_half_buffer_, product_half_buffer_;
    TBuf<TPosition::VECCALC> output_buffer_, state_sum_buffer_;
    GlobalTensor<float> state_gm_, state_out_gm_;
    GlobalTensor<half> lowrank_gm_[4], bias_gm_[4], prepared_gm_[3];
    GlobalTensor<half> k_gm_, v_gm_, v_first_gm_;
    GlobalTensor<half> kk_scale_gm_, ka_scale_gm_, r_gm_, output_gm_;
    uint32_t hidden_, n_, nn_, has_value_mix_, head_, row_block_;
    uint32_t row_start_, row_count_, elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_recurrence_state_direct(
    GM_ADDR state,
    GM_ADDR lowrank,
    GM_ADDR bias,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR v_first,
    GM_ADDR k_k,
    GM_ADDR k_a,
    GM_ADDR r,
    GM_ADDR state_out,
    GM_ADDR output,
    GM_ADDR prepared,
    uint32_t hidden,
    uint32_t has_value_mix,
    uint32_t row_blocks,
    uint32_t head_size) {
    RwkvRecurrenceStateDirectKernel kernel;
    kernel.Init(
        state, lowrank, bias, k, v, v_first, k_k, k_a, r, state_out,
        output, prepared, hidden, has_value_mix, row_blocks, head_size);
    kernel.Process();
}
