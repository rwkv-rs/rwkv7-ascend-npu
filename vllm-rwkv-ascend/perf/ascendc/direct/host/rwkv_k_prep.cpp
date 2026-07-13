#include "utils.h"
#include "aclrtlaunch_rwkv_k_prep_direct.h"

namespace rwkv7_ascend_direct {
std::vector<at::Tensor> k_prep(
    const at::Tensor& k,
    const at::Tensor& a,
    const at::Tensor& k_k,
    const at::Tensor& k_a) {
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k_prep requires fp16");
    auto kk_raw = at::empty_like(k);
    auto k_out = at::empty_like(k);
    const uint32_t elements = static_cast<uint32_t>(k.numel());
    constexpr uint32_t block_dim = 2;
    RWKV7_EXEC_KERNEL(
        rwkv_k_prep_direct,
        block_dim,
        k,
        a,
        k_k,
        k_a,
        kk_raw,
        k_out,
        elements);
    return {kk_raw, k_out};
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "k_prep(Tensor k, Tensor a, Tensor k_k, Tensor k_a) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("k_prep", TORCH_FN(rwkv7_ascend_direct::k_prep));
}
