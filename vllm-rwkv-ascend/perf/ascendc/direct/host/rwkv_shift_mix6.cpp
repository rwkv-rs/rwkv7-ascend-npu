#include "utils.h"
#include "aclrtlaunch_rwkv_shift_mix6_direct.h"

namespace rwkv7_ascend_direct {
std::vector<at::Tensor> shift_mix6(
    const at::Tensor& x,
    const at::Tensor& xx,
    const at::Tensor& mix1,
    const at::Tensor& mix2,
    const at::Tensor& mix3,
    const at::Tensor& mix4,
    const at::Tensor& mix5,
    const at::Tensor& mix6) {
    std::vector<at::Tensor> outputs;
    outputs.reserve(7);
    for (int i = 0; i < 7; ++i) outputs.push_back(at::empty_like(x));
    uint32_t elements = static_cast<uint32_t>(x.numel());
    constexpr uint32_t block_dim = 7;
    RWKV7_EXEC_KERNEL(
        rwkv_shift_mix6_direct,
        block_dim,
        x,
        xx,
        mix1,
        mix2,
        mix3,
        mix4,
        mix5,
        mix6,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        elements);
    return outputs;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def(
        "shift_mix6(Tensor x, Tensor xx, Tensor m1, Tensor m2, Tensor m3, "
        "Tensor m4, Tensor m5, Tensor m6) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("shift_mix6", TORCH_FN(rwkv7_ascend_direct::shift_mix6));
}
