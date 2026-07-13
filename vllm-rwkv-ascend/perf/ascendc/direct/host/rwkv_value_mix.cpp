#include "utils.h"
#include "aclrtlaunch_rwkv_value_mix_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor value_mix(
    const at::Tensor& v,
    const at::Tensor& v_first,
    const at::Tensor& mix) {
    auto out = at::empty_like(v);
    const uint32_t elements = static_cast<uint32_t>(v.numel());
    constexpr uint32_t block_dim = 1;
    RWKV7_EXEC_KERNEL(
        rwkv_value_mix_direct, block_dim, v, v_first, mix, out, elements);
    return out;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("value_mix(Tensor v, Tensor v_first, Tensor mix) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("value_mix", TORCH_FN(rwkv7_ascend_direct::value_mix));
}
