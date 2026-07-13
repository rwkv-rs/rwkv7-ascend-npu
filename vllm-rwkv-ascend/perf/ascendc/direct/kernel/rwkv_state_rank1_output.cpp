#include "kernel_operator.h"

using namespace AscendC;

// B=1 decode specialization for the RWKV-7 rank-one state transition:
//   state @ ((-kk) outer (kk * a))
//     == (state @ -kk) outer (kk * a)
// One vector core owns one head and also emits state @ r, avoiding the dense
// [N,N] transition, two BMM launches, and intermediate GM traffic.
class RwkvStateRank1OutputDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR state,
        GM_ADDR w,
        GM_ADDR v,
        GM_ADDR k,
        GM_ADDR kk,
        GM_ADDR a,
        GM_ADDR r,
        GM_ADDR state_out,
        GM_ADDR output,
        uint32_t row_blocks,
        uint32_t head_size) {
        n_ = head_size;
        nn_ = n_ * n_;
        const uint32_t block = GetBlockIdx();
        const uint32_t head = block / row_blocks;
        const uint32_t row_block = block % row_blocks;
        const uint32_t rows_per_block = (n_ + row_blocks - 1) / row_blocks;
        row_start_ = row_block * rows_per_block;
        const uint32_t remaining = n_ - row_start_;
        row_count_ = remaining < rows_per_block ? remaining : rows_per_block;
        elements_ = row_count_ * n_;
        state_gm_.SetGlobalBuffer(
            (__gm__ float*)state + head * nn_ + row_start_ * n_);
        state_out_gm_.SetGlobalBuffer(
            (__gm__ float*)state_out + head * nn_ + row_start_ * n_);
        w_gm_.SetGlobalBuffer((__gm__ half*)w + head * n_);
        v_gm_.SetGlobalBuffer((__gm__ half*)v + head * n_);
        k_gm_.SetGlobalBuffer((__gm__ half*)k + head * n_);
        kk_gm_.SetGlobalBuffer((__gm__ half*)kk + head * n_);
        a_gm_.SetGlobalBuffer((__gm__ half*)a + head * n_);
        r_gm_.SetGlobalBuffer((__gm__ half*)r + head * n_);
        output_gm_.SetGlobalBuffer(
            (__gm__ half*)output + head * n_ + row_start_);

        const uint32_t half_vector_bytes =
            ((n_ + 15) / 16 * 16) * sizeof(half);
        const uint32_t float_vector_bytes =
            ((n_ + 7) / 8 * 8) * sizeof(float);
        pipe_.InitBuffer(state_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_out_buffer_, elements_ * sizeof(float));
        pipe_.InitBuffer(state_half_buffer_, elements_ * sizeof(half));
        pipe_.InitBuffer(w_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(v_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(k_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(kk_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(a_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(r_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(direction_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(vk_half_buffer_, half_vector_bytes);
        pipe_.InitBuffer(product_half_buffer_, elements_ * sizeof(half));
        const uint32_t output_bytes =
            ((row_count_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(output_half_buffer_, output_bytes);
        pipe_.InitBuffer(w_float_buffer_, float_vector_bytes);
        pipe_.InitBuffer(direction_float_buffer_, float_vector_bytes);
        pipe_.InitBuffer(sum_half_buffer_, row_count_ * 32);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto state = state_buffer_.Get<float>();
        auto state_out = state_out_buffer_.Get<float>();
        auto state_half = state_half_buffer_.Get<half>();
        auto w_half = w_half_buffer_.Get<half>();
        auto v_half = v_half_buffer_.Get<half>();
        auto k_half = k_half_buffer_.Get<half>();
        auto kk_half = kk_half_buffer_.Get<half>();
        auto a_half = a_half_buffer_.Get<half>();
        auto r_half = r_half_buffer_.Get<half>();
        auto direction_half = direction_half_buffer_.Get<half>();
        auto vk_half = vk_half_buffer_.Get<half>();
        auto product_half = product_half_buffer_.Get<half>();
        auto output_half = output_half_buffer_.Get<half>();
        auto w_float = w_float_buffer_.Get<float>();
        auto direction_float = direction_float_buffer_.Get<float>();
        auto sum_half = sum_half_buffer_.Get<half>();

        DataCopy(state, state_gm_, elements_);
        DataCopy(w_half, w_gm_, n_);
        DataCopy(v_half, v_gm_, n_);
        DataCopy(k_half, k_gm_, n_);
        DataCopy(kk_half, kk_gm_, n_);
        DataCopy(a_half, a_gm_, n_);
        DataCopy(r_half, r_gm_, n_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);

        Cast(w_float, w_half, RoundMode::CAST_NONE, n_);
        Muls(direction_half, kk_half, static_cast<half>(-1.0f), n_);
        Mul(vk_half, kk_half, a_half, n_);
        PipeBarrier<PIPE_V>();
        Cast(direction_float, vk_half, RoundMode::CAST_NONE, n_);
        Cast(state_half, state, RoundMode::CAST_RINT, elements_);
        PipeBarrier<PIPE_V>();

        // The recurrent cache remains fp32, but the direction dot product may
        // use the same fp16 view required by the output projection.  This
        // halves reduction traffic while preserving fp32 state accumulation.
        const BinaryRepeatParams half_row_broadcast(1, 1, 1, 4, 4, 0);
        Mul(
            product_half, state_half, direction_half, 64,
            static_cast<uint8_t>(row_count_), half_row_broadcast);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            const uint32_t offset = row * n_;
            WholeReduceSum(
                sum_half[row * 16], product_half[offset], 64, 1, 1, 1, 4);
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams float_row_broadcast(1, 1, 1, 8, 8, 0);
        Mul(
            state_out, state, w_float, 64,
            static_cast<uint8_t>(row_count_), float_row_broadcast);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            const uint32_t offset = row * n_;
            Axpy(
                state_out[offset], direction_float,
                static_cast<float>(sum_half.GetValue(row * 16)), n_);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            const uint32_t offset = row * n_;
            Axpy(
                state_out[offset], k_half,
                v_half.GetValue(row_start_ + row), n_);
        }
        PipeBarrier<PIPE_V>();
        Cast(state_half, state_out, RoundMode::CAST_NONE, elements_);
        PipeBarrier<PIPE_V>();

        Mul(
            product_half, state_half, r_half, 64,
            static_cast<uint8_t>(row_count_), half_row_broadcast);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            const uint32_t offset = row * n_;
            WholeReduceSum(
                sum_half[row * 16], product_half[offset], 64, 1, 1, 1, 4);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < row_count_; ++row) {
            output_half.SetValue(row, sum_half.GetValue(row * 16));
        }

        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        DataCopy(state_out_gm_, state_out, elements_);
        DataCopy(output_gm_, output_half, row_count_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> state_buffer_, state_out_buffer_;
    TBuf<TPosition::VECCALC> state_half_buffer_;
    TBuf<TPosition::VECCALC> w_half_buffer_, v_half_buffer_, k_half_buffer_;
    TBuf<TPosition::VECCALC> kk_half_buffer_, a_half_buffer_, r_half_buffer_;
    TBuf<TPosition::VECCALC> direction_half_buffer_, vk_half_buffer_;
    TBuf<TPosition::VECCALC> product_half_buffer_, output_half_buffer_;
    TBuf<TPosition::VECCALC> w_float_buffer_;
    TBuf<TPosition::VECCALC> direction_float_buffer_;
    TBuf<TPosition::VECCALC> sum_half_buffer_;
    GlobalTensor<float> state_gm_, state_out_gm_;
    GlobalTensor<half> w_gm_, v_gm_, k_gm_, kk_gm_, a_gm_, r_gm_;
    GlobalTensor<half> output_gm_;
    uint32_t n_, nn_, row_start_, row_count_, elements_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_state_rank1_output_direct(
    GM_ADDR state,
    GM_ADDR w,
    GM_ADDR v,
    GM_ADDR k,
    GM_ADDR kk,
    GM_ADDR a,
    GM_ADDR r,
    GM_ADDR state_out,
    GM_ADDR output,
    uint32_t row_blocks,
    uint32_t head_size) {
    RwkvStateRank1OutputDirectKernel kernel;
    kernel.Init(
        state, w, v, k, kk, a, r, state_out, output, row_blocks, head_size);
    kernel.Process();
}
