#include "utils.h"
#include "aclrtlaunch_rwkv_w_pre_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor w_pre(const at::Tensor& x) {
    auto out = at::empty_like(x);
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    constexpr uint32_t block_dim = 1;
    RWKV7_EXEC_KERNEL(rwkv_w_pre_direct, block_dim, x, out, elements);
    return out;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("w_pre(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("w_pre", TORCH_FN(rwkv7_ascend_direct::w_pre));
}
