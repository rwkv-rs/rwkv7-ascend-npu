#include "kernel_operator.h"

using namespace AscendC;

// Graph-capturable B=1 embedding lookup.  The generic GatherV2 setup is much
// larger than the actual work for one 768-element fp16 row.
class RwkvEmbeddingDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR token_ids,
        GM_ADDR weight,
        GM_ADDR output,
        uint32_t hidden) {
        hidden_ = hidden;
        token_gm_.SetGlobalBuffer((__gm__ int64_t*)token_ids);
        weight_gm_.SetGlobalBuffer((__gm__ half*)weight);
        output_gm_.SetGlobalBuffer((__gm__ half*)output);
        pipe_.InitBuffer(token_buffer_, 32);
        const uint32_t bytes = ((hidden_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(row_buffer_, bytes);
        token_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_S>());
        row_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_MTE3>());
    }

    __aicore__ inline void Process() {
        auto token = token_buffer_.Get<int64_t>();
        auto row = row_buffer_.Get<half>();
        DataCopy(token, token_gm_, 4);
        SetFlag<HardEvent::MTE2_S>(token_ready_);
        WaitFlag<HardEvent::MTE2_S>(token_ready_);
        const int64_t index = token.GetValue(0);
        DataCopy(row, weight_gm_[index * hidden_], hidden_);
        SetFlag<HardEvent::MTE2_MTE3>(row_ready_);
        WaitFlag<HardEvent::MTE2_MTE3>(row_ready_);
        DataCopy(output_gm_, row, hidden_);
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(token_ready_);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE3>(row_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> token_buffer_, row_buffer_;
    GlobalTensor<int64_t> token_gm_;
    GlobalTensor<half> weight_gm_, output_gm_;
    uint32_t hidden_;
    event_t token_ready_, row_ready_;
};

extern "C" __global__ __aicore__ void rwkv_embedding_direct(
    GM_ADDR token_ids,
    GM_ADDR weight,
    GM_ADDR output,
    uint32_t hidden) {
    RwkvEmbeddingDirectKernel kernel;
    kernel.Init(token_ids, weight, output, hidden);
    kernel.Process();
}
