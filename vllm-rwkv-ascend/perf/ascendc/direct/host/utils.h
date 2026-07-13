#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace rwkv7_ascend_direct {
inline void* ConvertType(const at::Tensor& tensor) {
    return const_cast<void*>(tensor.data_ptr());
}

template <typename T>
T ConvertType(T value) {
    return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args) {
    return std::make_tuple(ConvertType(args)...);
}

#define RWKV7_EXEC_KERNEL(kernel_name, block_dim, ...)                         \
    do {                                                                        \
        auto stream = c10_npu::getCurrentNPUStream().stream(false);             \
        auto parameters = ConvertTypes(__VA_ARGS__);                            \
        auto launch = [stream, block_dim, parameters]() -> int {                \
            std::apply(                                                         \
                [&](auto&&... values) {                                         \
                    ACLRT_LAUNCH_KERNEL(kernel_name)(                            \
                        block_dim, stream, values...);                           \
                },                                                              \
                parameters);                                                    \
            return 0;                                                           \
        };                                                                      \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, launch);              \
    } while (false)
}  // namespace rwkv7_ascend_direct
