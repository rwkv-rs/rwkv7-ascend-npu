#include "utils.h"
#include "aclrtlaunch_rwkv_head_scaled_add_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor head_scaled_add(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& v) {
    auto out = at::empty_like(x);
    constexpr uint32_t block_dim = 1;
    const uint32_t heads = static_cast<uint32_t>(x.size(1));
    const uint32_t head_size = static_cast<uint32_t>(x.size(2));
    RWKV7_EXEC_KERNEL(
        rwkv_head_scaled_add_direct,
        block_dim,
        x,
        scale,
        v,
        out,
        heads,
        head_size);
    return out;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("head_scaled_add(Tensor x, Tensor scale, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl(
        "head_scaled_add", TORCH_FN(rwkv7_ascend_direct::head_scaled_add));
}
