#include "utils.h"
#include "aclrtlaunch_rwkv_sk_output_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor sk_output(
    const at::Tensor& x, const at::Tensor& r, const at::Tensor& k,
    const at::Tensor& r_k, const at::Tensor& v,
    int64_t heads, int64_t head_size) {
    auto out = at::empty_like(x);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    RWKV7_EXEC_KERNEL(
        rwkv_sk_output_direct, block_dim, x, r, k, r_k, v, out, n);
    return out;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "sk_output(Tensor x, Tensor r, Tensor k, Tensor r_k, Tensor v, "
        "int heads, int head_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("sk_output", TORCH_FN(rwkv7_ascend_direct::sk_output));
}
