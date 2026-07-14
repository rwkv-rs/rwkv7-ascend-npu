#include "kernel_operator.h"

using namespace AscendC;

// Batched N=64 layer-major prefill scan.  Each block owns a disjoint row range
// of one head, keeps that fp32 state tile in UB for the complete prompt, and
// writes the recurrent output for every token.  Projection and state-prep are
// intentionally outside this first kernel so their batched matmuls remain on
// the vendor-tuned path.
class RwkvPrefillScanDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR state,
        GM_ADDR w,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR kk,
        GM_ADDR a,
        GM_ADDR r,
        GM_ADDR output,
        uint32_t tokens,
        uint32_t hidden,
        uint32_t row_blocks,
        uint32_t head_size) {
        tokens_ = tokens;
        hidden_ = hidden;
        n_ = head_size;
        nn_ = n_ * n_;
        heads_ = hidden_ / n_;
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks_per_batch = heads_ * row_blocks;
        batch_ = block / blocks_per_batch;
        const uint32_t batch_block = block % blocks_per_batch;
        head_ = batch_block / row_blocks;
        row_block_ = block % row_blocks;
        const uint32_t rows_per_block = (n_ + row_blocks - 1) / row_blocks;
        row_start_ = row_block_ * rows_per_block;
        const uint32_t remaining = n_ - row_start_;
        row_count_ = remaining < rows_per_block ? remaining : rows_per_block;
        elements_ = row_count_ * n_;
        head_offset_ = head_ * n_;

        state_gm_.SetGlobalBuffer(
            (__gm__ float*)state + (batch_ * heads_ + head_) * nn_
                + row_start_ * n_);
        w_gm_.SetGlobalBuffer((__gm__ half*)w);
        k_gm_.SetGlobalBuffer((__gm__ half*)k);
        v_gm_.SetGlobalBuffer((__gm__ half*)v);
        kk_gm_.SetGlobalBuffer((__gm__ half*)kk);
        a_gm_.SetGlobalBuffer((__gm__ half*)a);
        r_gm_.SetGlobalBuffer((__gm__ half*)r);
        output_gm_.SetGlobalBuffer((__gm__ half*)output);

        const uint32_t half_bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_bytes = ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(vector_buffer_, 6 * half_bytes);
        pipe_.InitBuffer(mid_buffer_, 2 * half_bytes);
        pipe_.InitBuffer(float_buffer_, 2 * float_bytes);
        pipe_.InitBuffer(state_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_out_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_half_buffer_, elements_ * sizeof(half));
        pipe_.InitBuffer(product_buffer_, elements_ * sizeof(half));
        pipe_.InitBuffer(state_sum_buffer_, row_count_ * 32);
        const uint32_t output_bytes =
            ((row_count_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(output_buffer_, output_bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
        output_done_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void Process() {
        auto vectors = vector_buffer_.Get<half>();
        auto w = vectors;
        auto k = vectors[n_];
        auto v = vectors[2 * n_];
        auto kk = vectors[3 * n_];
        auto a = vectors[4 * n_];
        auto r = vectors[5 * n_];
        auto mid = mid_buffer_.Get<half>();
        auto minus_kk = mid;
        auto kk_a = mid[n_];
        auto float_vectors = float_buffer_.Get<float>();
        auto w_float = float_vectors;
        auto kk_a_float = float_vectors[n_];
        auto state = state_buffer_.Get<float>();
        auto state_out = state_out_buffer_.Get<float>();
        auto state_half = state_half_buffer_.Get<half>();
        auto product = product_buffer_.Get<half>();
        auto state_sum = state_sum_buffer_.Get<half>();
        auto output = output_buffer_.Get<half>();

        DataCopy(state, state_gm_, elements_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        const BinaryRepeatParams half_rows(1, 1, 1, 4, 4, 0);
        const BinaryRepeatParams float_rows(1, 1, 1, 8, 8, 0);
        for (uint32_t token = 0; token < tokens_; ++token) {
            const uint32_t offset =
                (batch_ * tokens_ + token) * hidden_ + head_offset_;
            DataCopy(w, w_gm_[offset], n_);
            DataCopy(k, k_gm_[offset], n_);
            DataCopy(v, v_gm_[offset], n_);
            DataCopy(kk, kk_gm_[offset], n_);
            DataCopy(a, a_gm_[offset], n_);
            DataCopy(r, r_gm_[offset], n_);
            SetFlag<HardEvent::MTE2_V>(input_ready_);
            WaitFlag<HardEvent::MTE2_V>(input_ready_);

            Muls(minus_kk, kk, static_cast<half>(-1.0f), n_);
            Mul(kk_a, kk, a, n_);
            Cast(state_half, state, RoundMode::CAST_RINT, elements_);
            PipeBarrier<PIPE_V>();
            Mul(
                product, state_half, minus_kk, 64,
                static_cast<uint8_t>(row_count_), half_rows);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < row_count_; ++row) {
                WholeReduceSum(
                    state_sum[row * 16], product[row * n_],
                    64, 1, 1, 1, 4);
            }
            Cast(w_float, w, RoundMode::CAST_NONE, n_);
            Cast(kk_a_float, kk_a, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
            Mul(
                state_out, state, w_float, 64,
                static_cast<uint8_t>(row_count_), float_rows);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < row_count_; ++row) {
                Axpy(
                    state_out[row * n_], kk_a_float,
                    static_cast<float>(state_sum.GetValue(row * 16)), n_);
                Axpy(
                    state_out[row * n_], k,
                    v.GetValue(row_start_ + row), n_);
            }
            PipeBarrier<PIPE_V>();

            Cast(state_half, state_out, RoundMode::CAST_NONE, elements_);
            PipeBarrier<PIPE_V>();
            Mul(
                product, state_half, r, 64,
                static_cast<uint8_t>(row_count_), half_rows);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < row_count_; ++row) {
                WholeReduceSum(
                    state_sum[row * 16], product[row * n_],
                    64, 1, 1, 1, 4);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < row_count_; ++row) {
                output.SetValue(row, state_sum.GetValue(row * 16));
            }
            Adds(state, state_out, 0.0f, elements_);
            SetFlag<HardEvent::V_MTE3>(output_ready_);
            WaitFlag<HardEvent::V_MTE3>(output_ready_);
            DataCopy(
                output_gm_[offset + row_start_], output, row_count_);
            SetFlag<HardEvent::MTE3_V>(output_done_);
            WaitFlag<HardEvent::MTE3_V>(output_done_);
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(state_gm_, state, elements_);
        SetFlag<HardEvent::MTE3_V>(output_done_);
        WaitFlag<HardEvent::MTE3_V>(output_done_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(output_done_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> vector_buffer_, mid_buffer_, float_buffer_;
    TBuf<TPosition::VECCALC> state_buffer_, state_out_buffer_;
    TBuf<TPosition::VECCALC> state_half_buffer_, product_buffer_;
    TBuf<TPosition::VECCALC> state_sum_buffer_, output_buffer_;
    GlobalTensor<float> state_gm_;
    GlobalTensor<half> w_gm_, k_gm_, v_gm_, kk_gm_, a_gm_, r_gm_;
    GlobalTensor<half> output_gm_;
    uint32_t tokens_, hidden_, n_, nn_, heads_, batch_, head_, row_block_;
    uint32_t row_start_, row_count_, elements_, head_offset_;
    event_t input_ready_, output_ready_, output_done_;
};

extern "C" __global__ __aicore__ void rwkv_prefill_scan_direct(
    GM_ADDR state,
    GM_ADDR w,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR kk,
    GM_ADDR a,
    GM_ADDR r,
    GM_ADDR output,
    uint32_t tokens,
    uint32_t hidden,
    uint32_t row_blocks,
    uint32_t head_size) {
    RwkvPrefillScanDirectKernel kernel;
    kernel.Init(
        state, w, k, v, kk, a, r, output, tokens, hidden, row_blocks,
        head_size);
    kernel.Process();
}
