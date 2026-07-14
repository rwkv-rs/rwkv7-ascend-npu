#include "kernel_operator.h"

using namespace AscendC;

// Prefill tmix_mix6 boundary.  Each block owns a small row range for one
// output.  Rows 0..5 are [xr,xk,xv,xw,xa,xg]; rows 6..9 duplicate
// [xw,xa,xg,xv] in the packed order consumed by the low-rank BMM.  Producing
// both layouts in one launch removes the eager stack/reorder launch.
class RwkvPrefillShiftMixDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR previous, GM_ADDR mix1, GM_ADDR mix2,
        GM_ADDR mix3, GM_ADDR mix4, GM_ADDR mix5, GM_ADDR mix6,
        GM_ADDR output, uint32_t rows, uint32_t tokens, uint32_t hidden,
        uint32_t row_blocks, uint32_t rows_per_block) {
        rows_ = rows;
        tokens_ = tokens;
        hidden_ = hidden;
        rows_per_block_ = rows_per_block;
        const uint32_t block = GetBlockIdx();
        output_index_ = block / row_blocks;
        row_start_ = (block % row_blocks) * rows_per_block_;

        GM_ADDR mixes[10] = {
            mix1, mix2, mix3, mix4, mix5, mix6,
            mix4, mix5, mix6, mix3};
        x_gm_.SetGlobalBuffer((__gm__ half*)x);
        previous_gm_.SetGlobalBuffer((__gm__ half*)previous);
        mix_gm_.SetGlobalBuffer((__gm__ half*)mixes[output_index_]);
        output_gm_.SetGlobalBuffer(
            (__gm__ half*)output + output_index_ * rows_ * hidden_);

        const uint32_t bytes = hidden_ * sizeof(half);
        pipe_.InitBuffer(x_buffer_, bytes);
        pipe_.InitBuffer(previous_buffer_, bytes);
        pipe_.InitBuffer(mix_buffer_, bytes);
        pipe_.InitBuffer(output_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        input_done_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE2>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
        output_done_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void Process() {
        auto current = x_buffer_.Get<half>();
        auto prior = previous_buffer_.Get<half>();
        auto mix = mix_buffer_.Get<half>();
        auto output = output_buffer_.Get<half>();
        DataCopy(mix, mix_gm_, hidden_);

        const uint32_t candidate_end = row_start_ + rows_per_block_;
        const uint32_t row_end = candidate_end < rows_ ? candidate_end : rows_;
        for (uint32_t row = row_start_; row < row_end; ++row) {
            if (row != row_start_) {
                WaitFlag<HardEvent::V_MTE2>(input_done_);
                WaitFlag<HardEvent::MTE3_V>(output_done_);
            }
            const uint32_t offset = row * hidden_;
            DataCopy(current, x_gm_[offset], hidden_);
            if (row % tokens_ == 0) {
                const uint32_t batch = row / tokens_;
                DataCopy(prior, previous_gm_[batch * hidden_], hidden_);
            } else {
                DataCopy(prior, x_gm_[offset - hidden_], hidden_);
            }
            SetFlag<HardEvent::MTE2_V>(input_ready_);
            WaitFlag<HardEvent::MTE2_V>(input_ready_);

            Sub(prior, prior, current, hidden_);
            PipeBarrier<PIPE_V>();
            Mul(prior, prior, mix, hidden_);
            PipeBarrier<PIPE_V>();
            Add(output, current, prior, hidden_);
            SetFlag<HardEvent::V_MTE2>(input_done_);
            SetFlag<HardEvent::V_MTE3>(output_ready_);
            WaitFlag<HardEvent::V_MTE3>(output_ready_);
            DataCopy(output_gm_[offset], output, hidden_);
            SetFlag<HardEvent::MTE3_V>(output_done_);
        }
        if (row_start_ < row_end) {
            WaitFlag<HardEvent::V_MTE2>(input_done_);
            WaitFlag<HardEvent::MTE3_V>(output_done_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(input_done_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(output_done_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> x_buffer_, previous_buffer_, mix_buffer_;
    TBuf<TPosition::VECCALC> output_buffer_;
    GlobalTensor<half> x_gm_, previous_gm_, mix_gm_, output_gm_;
    uint32_t rows_, tokens_, hidden_, rows_per_block_;
    uint32_t output_index_, row_start_;
    event_t input_ready_, input_done_, output_ready_, output_done_;
};

extern "C" __global__ __aicore__ void rwkv_prefill_shift_mix_direct(
    GM_ADDR x, GM_ADDR previous, GM_ADDR mix1, GM_ADDR mix2,
    GM_ADDR mix3, GM_ADDR mix4, GM_ADDR mix5, GM_ADDR mix6,
    GM_ADDR output, uint32_t rows, uint32_t tokens, uint32_t hidden,
    uint32_t row_blocks, uint32_t rows_per_block) {
    RwkvPrefillShiftMixDirectKernel kernel;
    kernel.Init(
        x, previous, mix1, mix2, mix3, mix4, mix5, mix6, output,
        rows, tokens, hidden, row_blocks, rows_per_block);
    kernel.Process();
}
