#include "utils.h"
#include "aclrtlaunch_rwkv_outer_products_direct.h"

namespace rwkv7_ascend_direct {
std::vector<at::Tensor> outer_products(
    const at::Tensor& v,
    const at::Tensor& k,
    const at::Tensor& kk,
    const at::Tensor& a,
    int64_t heads,
    int64_t head_size) {
    auto options = v.options();
    auto vk = at::empty({1, heads, head_size, head_size}, options);
    auto ab = at::empty(
        {1, heads, head_size, head_size}, v.options().dtype(at::kFloat));
    const uint32_t h = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    const uint32_t block_dim = 2 * h;
    RWKV7_EXEC_KERNEL(
        rwkv_outer_products_direct,
        block_dim,
        v,
        k,
        kk,
        a,
        vk,
        ab,
        h,
        n);
    return {vk, ab};
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "outer_products(Tensor v, Tensor k, Tensor kk, Tensor a, "
        "int heads, int head_size) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl(
        "outer_products", TORCH_FN(rwkv7_ascend_direct::outer_products));
}
