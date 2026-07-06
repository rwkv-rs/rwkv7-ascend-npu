// call_rwkv_wexp.cpp — torch binding that calls the custom AscendC op aclnnRwkvWexp.
// Uses only the ACL C API (no torch_npu C++ symbols) so it links cleanly.
#include <torch/extension.h>
#include <vector>
#include "acl/acl.h"
#include "aclnn_rwkv_wexp.h"

static aclDataType to_acl_dtype(at::ScalarType t) {
    if (t == at::kHalf)  return ACL_FLOAT16;
    if (t == at::kFloat) return ACL_FLOAT;
    if (t == at::kInt)   return ACL_INT32;
    if (t == at::kLong)  return ACL_INT64;
    return ACL_FLOAT16;
}

static aclTensor* to_acl_tensor(const at::Tensor& x) {
    auto c = x.contiguous();
    int n = c.dim();
    std::vector<int64_t> shape(n), stride(n);
    for (int i = 0; i < n; i++) { shape[i] = c.size(i); stride[i] = c.stride(i); }
    return aclCreateTensor(shape.data(), (uint64_t)n, to_acl_dtype(c.scalar_type()),
                           stride.data(), (int64_t)n, ACL_FORMAT_ND, nullptr, (uint64_t)0, c.data_ptr());
}

at::Tensor rwkv_wexp(at::Tensor x) {
    x = x.contiguous();
    auto y = at::empty_like(x);
    aclTensor* xa = to_acl_tensor(x);
    aclTensor* ya = to_acl_tensor(y);
    uint64_t wsSize = 0;
    aclOpExecutor* ex = nullptr;
    aclnnStatus s1 = aclnnRwkvWexpGetWorkspaceSize(xa, ya, &wsSize, &ex);
    TORCH_CHECK(s1 == 0, "aclnnRwkvWexpGetWorkspaceSize failed: ", (int)s1);
    void* ws = nullptr;
    if (wsSize > 0) {
        aclError rc = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        TORCH_CHECK(rc == ACL_SUCCESS, "aclrtMalloc failed: ", (int)rc);
    }
    aclrtStream stream = nullptr;
    aclrtCtxGetCurrentDefaultStream(&stream);  // torch_npu sets the ACL current stream
    aclnnStatus s2 = aclnnRwkvWexp(ws, wsSize, ex, stream);
    TORCH_CHECK(s2 == 0, "aclnnRwkvWexp failed: ", (int)s2);
    aclDestroyTensor(xa);
    aclDestroyTensor(ya);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv_wexp", &rwkv_wexp, "y = exp(-0.606531*sigmoid(x)) via custom AscendC op");
}
