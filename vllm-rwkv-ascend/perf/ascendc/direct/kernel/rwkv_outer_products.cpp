#include "kernel_operator.h"

using namespace AscendC;

// Compute both RWKV outer products in one launch.  Each vector core owns one
// output head: the first H cores compute v @ k, the next H compute
// (-kk) @ (kk * a).
class RwkvOuterProductsDirectKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR v,
        GM_ADDR k,
        GM_ADDR kk,
        GM_ADDR a,
        GM_ADDR vk,
        GM_ADDR ab,
        uint32_t heads,
        uint32_t head_size) {
        head_size_ = head_size;
        head_elements_ = head_size * head_size;
        const uint32_t output_index = GetBlockIdx();
        const bool is_ab = output_index >= heads;
        const uint32_t head = is_ab ? output_index - heads : output_index;
        if (is_ab) {
            left_gm_.SetGlobalBuffer((__gm__ half*)kk + head * head_size);
            right_gm_.SetGlobalBuffer((__gm__ half*)kk + head * head_size);
            a_gm_.SetGlobalBuffer((__gm__ half*)a + head * head_size);
            ab_gm_.SetGlobalBuffer((__gm__ float*)ab + head * head_elements_);
        } else {
            left_gm_.SetGlobalBuffer((__gm__ half*)v + head * head_size);
            right_gm_.SetGlobalBuffer((__gm__ half*)k + head * head_size);
            out_gm_.SetGlobalBuffer((__gm__ half*)vk + head * head_elements_);
        }
        is_ab_ = is_ab;
        const uint32_t vector_bytes =
            ((head_size + 15) / 16 * 16) * sizeof(half);
        const uint32_t output_bytes = head_elements_ * sizeof(half);
        pipe_.InitBuffer(left_buffer_, vector_bytes);
        pipe_.InitBuffer(right_buffer_, vector_bytes);
        pipe_.InitBuffer(a_buffer_, vector_bytes);
        pipe_.InitBuffer(product_buffer_, vector_bytes);
        pipe_.InitBuffer(out_buffer_, output_bytes);
        pipe_.InitBuffer(out_float_buffer_, head_elements_ * sizeof(float));
        input_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        output_ready_ = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::V_MTE3>());
    }

    __aicore__ inline void Process() {
        auto left = left_buffer_.Get<half>();
        auto right = right_buffer_.Get<half>();
        auto a = a_buffer_.Get<half>();
        auto product = product_buffer_.Get<half>();
        auto out = out_buffer_.Get<half>();
        auto out_float = out_float_buffer_.Get<float>();
        DataCopy(left, left_gm_, head_size_);
        DataCopy(right, right_gm_, head_size_);
        if (is_ab_) DataCopy(a, a_gm_, head_size_);
        SetFlag<HardEvent::MTE2_V>(input_ready_);
        WaitFlag<HardEvent::MTE2_V>(input_ready_);
        if (is_ab_) {
            Mul(product, right, a, head_size_);
            Muls(a, left, static_cast<half>(-1.0f), head_size_);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < head_size_; ++row) {
                Muls(
                    out[row * head_size_],
                    product,
                    a.GetValue(row),
                    head_size_);
            }
        } else {
            for (uint32_t row = 0; row < head_size_; ++row) {
                Muls(
                    out[row * head_size_],
                    right,
                    left.GetValue(row),
                    head_size_);
            }
        }
        if (is_ab_) {
            PipeBarrier<PIPE_V>();
            Cast(out_float, out, RoundMode::CAST_NONE, head_elements_);
        }
        SetFlag<HardEvent::V_MTE3>(output_ready_);
        WaitFlag<HardEvent::V_MTE3>(output_ready_);
        if (is_ab_) {
            DataCopy(ab_gm_, out_float, head_elements_);
        } else {
            DataCopy(out_gm_, out, head_elements_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(input_ready_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(output_ready_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> left_buffer_, right_buffer_, a_buffer_;
    TBuf<TPosition::VECCALC> product_buffer_, out_buffer_, out_float_buffer_;
    GlobalTensor<half> left_gm_, right_gm_, a_gm_, out_gm_;
    GlobalTensor<float> ab_gm_;
    uint32_t head_size_, head_elements_;
    bool is_ab_;
    event_t input_ready_, output_ready_;
};

extern "C" __global__ __aicore__ void rwkv_outer_products_direct(
    GM_ADDR v,
    GM_ADDR k,
    GM_ADDR kk,
    GM_ADDR a,
    GM_ADDR vk,
    GM_ADDR ab,
    uint32_t heads,
    uint32_t head_size) {
    RwkvOuterProductsDirectKernel kernel;
    kernel.Init(v, k, kk, a, vk, ab, heads, head_size);
    kernel.Process();
}
