#include "utils.h"
#include "aclrtlaunch_rwkv_normalize_k_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor normalize_k(
    const at::Tensor& x, int64_t heads, int64_t head_size) {
    auto out = at::empty_like(x);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    RWKV7_EXEC_KERNEL(rwkv_normalize_k_direct, block_dim, x, out, n);
    return out;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("normalize_k(Tensor x, int heads, int head_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("normalize_k", TORCH_FN(rwkv7_ascend_direct::normalize_k));
}
