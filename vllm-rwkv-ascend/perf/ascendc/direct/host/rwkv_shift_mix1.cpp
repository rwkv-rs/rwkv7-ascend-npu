#include "utils.h"
#include "aclrtlaunch_rwkv_shift_mix1_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor shift_mix1(
    const at::Tensor& x, const at::Tensor& xx, const at::Tensor& mix) {
    TORCH_CHECK(x.scalar_type() == at::kHalf, "shift_mix1 requires fp16");
    TORCH_CHECK(x.is_contiguous() && xx.is_contiguous() && mix.is_contiguous(),
                "shift_mix1 requires contiguous tensors");
    TORCH_CHECK(x.numel() == xx.numel() && x.numel() == mix.numel(),
                "shift_mix1 shape mismatch");
    auto y = at::empty_like(x);
    uint32_t elements = static_cast<uint32_t>(x.numel());
    constexpr uint32_t block_dim = 1;
    RWKV7_EXEC_KERNEL(
        rwkv_shift_mix1_direct, block_dim, x, xx, mix, y, elements);
    return y;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("shift_mix1(Tensor x, Tensor xx, Tensor mix) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("shift_mix1", TORCH_FN(rwkv7_ascend_direct::shift_mix1));
}
