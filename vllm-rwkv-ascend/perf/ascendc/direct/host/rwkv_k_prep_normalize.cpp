#include "utils.h"
#include "aclrtlaunch_rwkv_k_prep_normalize_direct.h"

namespace rwkv7_ascend_direct {
std::vector<at::Tensor> k_prep_normalize(
    const at::Tensor& k,
    const at::Tensor& a,
    const at::Tensor& k_k,
    const at::Tensor& k_a,
    int64_t heads,
    int64_t head_size) {
    auto kk = at::empty_like(k);
    auto k_out = at::empty_like(k);
    auto a_out = at::empty_like(a);
    const uint32_t h = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    const uint32_t block_dim = h + 1;
    RWKV7_EXEC_KERNEL(
        rwkv_k_prep_normalize_direct,
        block_dim,
        k,
        a,
        k_k,
        k_a,
        kk,
        k_out,
        a_out,
        h,
        n);
    return {kk, k_out, a_out};
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "k_prep_normalize(Tensor k, Tensor a, Tensor k_k, Tensor k_a, "
        "int heads, int head_size) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl(
        "k_prep_normalize",
        TORCH_FN(rwkv7_ascend_direct::k_prep_normalize));
}
