#include "kernel_operator.h"

using namespace AscendC;

// Small B=1 decode concat used by the static mix-projection path.  The generic
// ConcatD kernel costs more than the projections it helps eliminate at 768d.
class RwkvConcat2DirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR first, GM_ADDR second, GM_ADDR out, uint32_t elements) {
        n_ = elements;
        first_gm_.SetGlobalBuffer((__gm__ half*)first);
        second_gm_.SetGlobalBuffer((__gm__ half*)second);
        out_gm_.SetGlobalBuffer((__gm__ half*)out);
        const uint32_t bytes = ((n_ + 15) / 16 * 16) * sizeof(half);
        pipe_.InitBuffer(first_buffer_, bytes);
        pipe_.InitBuffer(second_buffer_, bytes);
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_MTE3>());
    }

    __aicore__ inline void Process() {
        auto first = first_buffer_.Get<half>();
        auto second = second_buffer_.Get<half>();
        DataCopy(first, first_gm_, n_);
        DataCopy(second, second_gm_, n_);
        SetFlag<HardEvent::MTE2_MTE3>(input_ready_);
        WaitFlag<HardEvent::MTE2_MTE3>(input_ready_);
        DataCopy(out_gm_, first, n_);
        DataCopy(out_gm_[n_], second, n_);
        // The old shift state is no longer needed after it has been copied to
        // the packed projection input.  Refresh the cache with the normalized
        // value here instead of launching a separate TensorMove.
        DataCopy(second_gm_, first, n_);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE3>(input_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> first_buffer_, second_buffer_;
    GlobalTensor<half> first_gm_, second_gm_, out_gm_;
    uint32_t n_;
    event_t input_ready_;
};

extern "C" __global__ __aicore__ void rwkv_concat2_direct(
    GM_ADDR first, GM_ADDR second, GM_ADDR out, uint32_t elements) {
    RwkvConcat2DirectKernel kernel;
    kernel.Init(first, second, out, elements);
    kernel.Process();
}
