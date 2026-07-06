// debug_ops.cpp — isolate whether at::layer_norm / at::group_norm / 4D matmul
// match eager PyTorch on NPU. If these diverge, the v3 port can't be correct.
#include <torch/extension.h>

at::Tensor test_layernorm(at::Tensor x, at::Tensor w, at::Tensor b, int64_t h) {
    return at::layer_norm(x, {h}, w, b, 1e-5);
}
at::Tensor test_groupnorm(at::Tensor x, at::Tensor w, at::Tensor b, int64_t g, double eps) {
    return at::group_norm(x, g, w, b, eps);
}
at::Tensor test_matmul4d(at::Tensor a, at::Tensor b) { return at::matmul(a, b); }
at::Tensor test_bmm(at::Tensor a, at::Tensor b) { return torch::bmm(a, b); }
at::Tensor test_norm2(at::Tensor x) {
    return x / x.norm(2, /*dim=*/-1, /*keepdim=*/true).clamp_min(1e-8);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_layernorm", &test_layernorm);
    m.def("test_groupnorm", &test_groupnorm);
    m.def("test_matmul4d", &test_matmul4d);
    m.def("test_bmm", &test_bmm);
    m.def("test_norm2", &test_norm2);
}
