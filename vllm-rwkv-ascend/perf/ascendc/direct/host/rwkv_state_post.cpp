#include "utils.h"
#include "aclrtlaunch_rwkv_state_post_direct.h"

namespace rwkv7_ascend_direct {
std::vector<at::Tensor> state_post(
    const at::Tensor& state,
    const at::Tensor& w,
    const at::Tensor& term2,
    const at::Tensor& vk) {
    TORCH_CHECK(state.dim() == 4, "state_post expects [B,H,N,N] state");
    TORCH_CHECK(state.size(0) == 1, "state_post currently supports B=1");
    TORCH_CHECK(state.size(2) == state.size(3), "state head must be square");
    TORCH_CHECK(state.scalar_type() == at::kFloat, "state must be fp32");
    TORCH_CHECK(term2.scalar_type() == at::kFloat, "term2 must be fp32");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "w must be fp16");
    TORCH_CHECK(vk.scalar_type() == at::kHalf, "vk must be fp16");
    auto out = at::empty_like(state);
    auto out_half = at::empty(state.sizes(), state.options().dtype(at::kHalf));
    const uint32_t block_dim = static_cast<uint32_t>(state.size(1));
    const uint32_t head_size = static_cast<uint32_t>(state.size(2));
    RWKV7_EXEC_KERNEL(
        rwkv_state_post_direct,
        block_dim,
        state,
        w,
        term2,
        vk,
        out,
        out_half,
        head_size);
    return {out, out_half};
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "state_post(Tensor state, Tensor w, Tensor term2, Tensor vk) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("state_post", TORCH_FN(rwkv7_ascend_direct::state_post));
}
