#include "utils.h"
#include "aclrtlaunch_rwkv_relu_square_direct.h"

namespace rwkv7_ascend_direct {
at::Tensor relu_square(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kHalf, "relu_square requires fp16");
    auto y = at::empty_like(x);
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    constexpr uint32_t block_dim = 1;
    RWKV7_EXEC_KERNEL(rwkv_relu_square_direct, block_dim, x, y, elements);
    return y;
}
}  // namespace rwkv7_ascend_direct

TORCH_LIBRARY_FRAGMENT(rwkv7_ascend, library) {
    library.def("relu_square(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_ascend, PrivateUse1, library) {
    library.impl("relu_square", TORCH_FN(rwkv7_ascend_direct::relu_square));
}
