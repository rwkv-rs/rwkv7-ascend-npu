// rwkv7_ascend_v3.cpp — Correct + optimized single-call C++ forward for RWKV-7
// on Ascend NPU.
//
// Matches native.py attn_step_batched + ffn_step_batched exactly, incl.:
//   * per-layer LayerNorms (pre_norm on layer 0; attn_norm, ffn_norm every layer)
//     + final LayerNorm, via fused at::layer_norm.
//   * LoRA lora[2] biases (w/a/v have bias=True) via at::linear(x, w, b).
//   * g_norm via fused at::group_norm.
//
// Optimizations (reduce CANN kernel-launch count):
//   1. Batched shift-mix: stack [x_r..x_g] -> 1 mul + 1 add (+ free selects).
//   2. r/k/v via one bmm.
//   3. w_exp in fp16 (sigmoid range is exp-safe).

#include <torch/extension.h>

#ifndef RWKV7_RANK1_ROW_BLOCKS
#define RWKV7_RANK1_ROW_BLOCKS 2
#endif
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_mul.h>
#include <array>
#include <vector>

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX2
#define ACL_API
#include "acl/acl.h"
#include "aclnn_rwkv_shift_mix2.h"
#endif
#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_OPCOMMAND
#include "torch_npu/csrc/framework/OpCommand.h"
#endif
#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_rwkv_shift_mix1_direct.h"
#include "aclrtlaunch_rwkv_shift_mix6_direct.h"
#include "aclrtlaunch_rwkv_state_post_direct.h"
#include "aclrtlaunch_rwkv_k_prep_direct.h"
#include "aclrtlaunch_rwkv_relu_square_direct.h"
#include "aclrtlaunch_rwkv_value_mix_direct.h"
#include "aclrtlaunch_rwkv_head_scaled_add_direct.h"
#include "aclrtlaunch_rwkv_outer_products_direct.h"
#include "aclrtlaunch_rwkv_sk_output_direct.h"
#include "aclrtlaunch_rwkv_normalize_k_direct.h"
#include "aclrtlaunch_rwkv_w_pre_direct.h"
#include "aclrtlaunch_rwkv_k_prep_normalize_direct.h"
#include "aclrtlaunch_rwkv_state_rank1_output_direct.h"
#include "aclrtlaunch_rwkv_groupnorm_sk_direct.h"
#include "aclrtlaunch_rwkv_lowrank_activate_direct.h"
#include "aclrtlaunch_rwkv_lowrank_post_direct.h"
#include "aclrtlaunch_rwkv_recurrence_prep_direct.h"
#include "aclrtlaunch_rwkv_recurrence_state_direct.h"
#include "aclrtlaunch_rwkv_ffn_prep_direct.h"
#include "aclrtlaunch_rwkv_attn_prep_direct.h"
#include "aclrtlaunch_rwkv_embedding_direct.h"
#include "aclrtlaunch_rwkv_embedding_norm2_direct.h"
#include "aclrtlaunch_rwkv_layer_norm_direct.h"
#include "aclrtlaunch_rwkv_add_layer_norm_direct.h"
#include "aclrtlaunch_rwkv_concat2_direct.h"
#endif

static const float EXP_HALF = 0.606531f;
static const double LN_EPS = 1e-5;

#define RWKV7_DIRECT_LINEAR(input, weight) at::linear((input), (weight))

at::Tensor fused_ln(at::Tensor x, at::Tensor w, at::Tensor b, int64_t hidden) {
    return at::layer_norm(x, {hidden}, w, b, LN_EPS);
}


#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX2
static std::vector<aclTensor*> rwkv7_persistent_acl_tensors;
static std::vector<at::Tensor> rwkv7_persistent_tensors;

static aclTensor* rwkv7_acl_tensor(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "AscendC shift-mix requires contiguous tensors");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, "AscendC shift-mix requires fp16");
    const int dimensions = tensor.dim();
    std::vector<int64_t> shape(dimensions);
    std::vector<int64_t> stride(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        shape[i] = tensor.size(i);
        stride[i] = tensor.stride(i);
    }
    aclTensor* acl_tensor = aclCreateTensor(
        shape.data(), static_cast<uint64_t>(dimensions), ACL_FLOAT16,
        stride.data(), 0, ACL_FORMAT_ND, shape.data(),
        static_cast<uint64_t>(dimensions), tensor.data_ptr());
    // aclnn is asynchronous.  Keep descriptors alive through NPUGraph capture;
    // graph replay itself does not execute this host helper or grow the vector.
    rwkv7_persistent_acl_tensors.push_back(acl_tensor);
    return acl_tensor;
}

static std::vector<at::Tensor> rwkv7_ascendc_shift_mix2(
        at::Tensor x, at::Tensor xx, at::Tensor mix1, at::Tensor mix2) {
    x = x.contiguous();
    xx = xx.contiguous();
    mix1 = mix1.contiguous();
    mix2 = mix2.contiguous();
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "AscendC shift-mix is a B=1 decode prototype");
    TORCH_CHECK(x.numel() == xx.numel() && x.numel() == mix1.numel() &&
                x.numel() == mix2.numel(), "AscendC shift-mix shape mismatch");
    auto y1 = at::empty_like(x);
    auto y2 = at::empty_like(x);
    // The direct aclnn bridge bypasses PyTorch's stream-aware allocator.
    // Retain capture-time buffers so asynchronous kernels cannot observe
    // storage recycled by a later layer; replay uses these fixed addresses.
    rwkv7_persistent_tensors.insert(
        rwkv7_persistent_tensors.end(), {x, xx, mix1, mix2, y1, y2});
    aclTensor* x_acl = rwkv7_acl_tensor(x);
    aclTensor* xx_acl = rwkv7_acl_tensor(xx);
    aclTensor* mix1_acl = rwkv7_acl_tensor(mix1);
    aclTensor* mix2_acl = rwkv7_acl_tensor(mix2);
    aclTensor* y1_acl = rwkv7_acl_tensor(y1);
    aclTensor* y2_acl = rwkv7_acl_tensor(y2);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus prepare_status = aclnnRwkvShiftMix2GetWorkspaceSize(
        x_acl, xx_acl, mix1_acl, mix2_acl, y1_acl, y2_acl,
        &workspace_size, &executor);
    TORCH_CHECK(prepare_status == ACL_SUCCESS,
                "aclnnRwkvShiftMix2GetWorkspaceSize failed: ",
                static_cast<int>(prepare_status));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        const aclError allocation_status =
            aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        TORCH_CHECK(allocation_status == ACL_SUCCESS,
                    "AscendC shift-mix workspace allocation failed: ",
                    static_cast<int>(allocation_status));
    }
    aclrtStream stream = nullptr;
    const aclError stream_status = aclrtCtxGetCurrentDefaultStream(&stream);
    TORCH_CHECK(stream_status == ACL_SUCCESS,
                "AscendC shift-mix stream lookup failed: ",
                static_cast<int>(stream_status));
    const aclnnStatus run_status =
        aclnnRwkvShiftMix2(workspace, workspace_size, executor, stream);
    TORCH_CHECK(run_status == ACL_SUCCESS, "aclnnRwkvShiftMix2 failed: ",
                static_cast<int>(run_status));
    if (workspace != nullptr) aclrtFree(workspace);
    return {y1, y2};
}
#endif

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT
static at::Tensor rwkv7_ascendc_shift_mix1_direct(
        const at::Tensor& x, const at::Tensor& xx, const at::Tensor& mix) {
    auto y = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = 1;
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    void* x_data = const_cast<void*>(x.data_ptr());
    void* xx_data = const_cast<void*>(xx.data_ptr());
    void* mix_data = const_cast<void*>(mix.data_ptr());
    void* y_data = const_cast<void*>(y.data_ptr());
    auto launch = [stream, block_dim, x_data, xx_data, mix_data,
                   y_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_shift_mix1_direct)(
            block_dim, stream, x_data, xx_data, mix_data, y_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_shift_mix1_direct", launch);
    return y;
}

static std::vector<at::Tensor> rwkv7_ascendc_shift_mix6_direct(
        const at::Tensor& x,
        const at::Tensor& xx,
        const at::Tensor& mix1,
        const at::Tensor& mix2,
        const at::Tensor& mix3,
        const at::Tensor& mix4,
        const at::Tensor& mix5,
        const at::Tensor& mix6) {
    std::vector<at::Tensor> outputs;
#if defined(RWKV7_USE_RKV_BMM) || defined(RWKV7_USE_LOWRANK_BMM)
    auto packed = at::empty({7, x.numel()}, x.options());
    outputs.reserve(8);
    for (int i = 0; i < 7; ++i) {
        outputs.push_back(packed[i].view(x.sizes()));
    }
    outputs.push_back(packed.view({7, x.size(0), x.size(1)}));
#else
    outputs.reserve(7);
    for (int i = 0; i < 7; ++i) outputs.push_back(at::empty_like(x));
#endif
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = 7;
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    std::array<void*, 15> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(xx.data_ptr()),
        const_cast<void*>(mix1.data_ptr()),
        const_cast<void*>(mix2.data_ptr()),
        const_cast<void*>(mix3.data_ptr()),
        const_cast<void*>(mix4.data_ptr()),
        const_cast<void*>(mix5.data_ptr()),
        const_cast<void*>(mix6.data_ptr()),
        const_cast<void*>(outputs[0].data_ptr()),
        const_cast<void*>(outputs[1].data_ptr()),
        const_cast<void*>(outputs[2].data_ptr()),
        const_cast<void*>(outputs[3].data_ptr()),
        const_cast<void*>(outputs[4].data_ptr()),
        const_cast<void*>(outputs[5].data_ptr()),
        const_cast<void*>(outputs[6].data_ptr())};
    auto launch = [stream, block_dim, pointers, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_shift_mix6_direct)(
            block_dim, stream,
            pointers[0], pointers[1], pointers[2], pointers[3],
            pointers[4], pointers[5], pointers[6], pointers[7],
            pointers[8], pointers[9], pointers[10], pointers[11],
            pointers[12], pointers[13], pointers[14], elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_shift_mix6_direct", launch);
    return outputs;
}

static std::vector<at::Tensor> rwkv7_ascendc_state_post_direct(
        const at::Tensor& state,
        const at::Tensor& w,
        const at::Tensor& term2,
        const at::Tensor& vk,
        int64_t heads,
    int64_t head_size) {
    auto out = at::empty_like(state);
    auto out_half = at::empty(state.sizes(), state.options().dtype(at::kHalf));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    std::array<void*, 6> pointers = {
        const_cast<void*>(state.data_ptr()),
        const_cast<void*>(w.data_ptr()),
        const_cast<void*>(term2.data_ptr()),
        const_cast<void*>(vk.data_ptr()),
        const_cast<void*>(out.data_ptr()),
        const_cast<void*>(out_half.data_ptr())};
    auto launch = [stream, block_dim, pointers, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_state_post_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_state_post_direct", launch);
    return {out, out_half};
}

static std::vector<at::Tensor> rwkv7_ascendc_k_prep_direct(
        const at::Tensor& k,
        const at::Tensor& a,
        const at::Tensor& k_k,
        const at::Tensor& k_a) {
    auto kk_raw = at::empty_like(k);
    auto k_out = at::empty_like(k);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 2;
    const uint32_t elements = static_cast<uint32_t>(k.numel());
    std::array<void*, 6> pointers = {
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(a.data_ptr()),
        const_cast<void*>(k_k.data_ptr()),
        const_cast<void*>(k_a.data_ptr()),
        const_cast<void*>(kk_raw.data_ptr()),
        const_cast<void*>(k_out.data_ptr())};
    auto launch = [stream, pointers, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_k_prep_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_k_prep_direct", launch);
    return {kk_raw, k_out};
}

static at::Tensor rwkv7_ascendc_relu_square_direct(const at::Tensor& x) {
    auto y = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 1;
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    void* x_data = const_cast<void*>(x.data_ptr());
    void* y_data = const_cast<void*>(y.data_ptr());
    auto launch = [stream, x_data, y_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_relu_square_direct)(
            block_dim, stream, x_data, y_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_relu_square_direct", launch);
    return y;
}

static at::Tensor rwkv7_ascendc_value_mix_direct(
        const at::Tensor& v,
        const at::Tensor& v_first,
        const at::Tensor& mix) {
    auto out = at::empty_like(v);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 1;
    const uint32_t elements = static_cast<uint32_t>(v.numel());
    std::array<void*, 4> pointers = {
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(v_first.data_ptr()),
        const_cast<void*>(mix.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, pointers, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_value_mix_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_value_mix_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_head_scaled_add_direct(
        const at::Tensor& x,
        const at::Tensor& scale,
        const at::Tensor& v,
        int64_t heads,
        int64_t head_size) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 1;
    const uint32_t h = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    std::array<void*, 4> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(scale.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, pointers, h, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_head_scaled_add_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], h, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_head_scaled_add_direct", launch);
    return out;
}

static std::vector<at::Tensor> rwkv7_ascendc_outer_products_direct(
        const at::Tensor& v,
        const at::Tensor& k,
        const at::Tensor& kk,
        const at::Tensor& a,
        int64_t heads,
        int64_t head_size) {
    auto options = v.options();
    auto vk = at::empty({1, heads, head_size, head_size}, options);
    auto ab = at::empty(
        {1, heads, head_size, head_size}, options.dtype(at::kFloat));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t h = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    const uint32_t block_dim = 2 * h;
    std::array<void*, 6> pointers = {
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(kk.data_ptr()),
        const_cast<void*>(a.data_ptr()),
        const_cast<void*>(vk.data_ptr()),
        const_cast<void*>(ab.data_ptr())};
    auto launch = [stream, block_dim, pointers, h, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_outer_products_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], h, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_outer_products_direct", launch);
    return {vk, ab};
}

static at::Tensor rwkv7_ascendc_sk_output_direct(
        const at::Tensor& x,
        const at::Tensor& r,
        const at::Tensor& k,
        const at::Tensor& r_k,
        const at::Tensor& v,
        int64_t heads,
        int64_t head_size) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    std::array<void*, 6> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(r.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(r_k.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, block_dim, pointers, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_sk_output_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_sk_output_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_normalize_k_direct(
        const at::Tensor& x, int64_t heads, int64_t head_size) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    void* x_data = const_cast<void*>(x.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, block_dim, x_data, out_data, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_normalize_k_direct)(
            block_dim, stream, x_data, out_data, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_normalize_k_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_w_pre_direct(const at::Tensor& x) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 1;
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    void* x_data = const_cast<void*>(x.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, x_data, out_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_w_pre_direct)(
            block_dim, stream, x_data, out_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_w_pre_direct", launch);
    return out;
}

static std::vector<at::Tensor> rwkv7_ascendc_k_prep_normalize_direct(
        const at::Tensor& k,
        const at::Tensor& a,
        const at::Tensor& k_k,
        const at::Tensor& k_a,
        int64_t heads,
        int64_t head_size) {
    auto kk = at::empty_like(k);
    auto k_out = at::empty_like(k);
    auto a_out = at::empty_like(a);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t h = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    const uint32_t block_dim = h + 1;
    std::array<void*, 7> pointers = {
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(a.data_ptr()),
        const_cast<void*>(k_k.data_ptr()),
        const_cast<void*>(k_a.data_ptr()),
        const_cast<void*>(kk.data_ptr()),
        const_cast<void*>(k_out.data_ptr()),
        const_cast<void*>(a_out.data_ptr())};
    auto launch = [stream, block_dim, pointers, h, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_k_prep_normalize_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], pointers[6], h, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_k_prep_normalize_direct", launch);
    return {kk, k_out, a_out};
}

static std::vector<at::Tensor> rwkv7_ascendc_state_rank1_output_direct(
        const at::Tensor& state,
        const at::Tensor& w,
        const at::Tensor& v,
        const at::Tensor& k,
        const at::Tensor& kk,
        const at::Tensor& a,
        const at::Tensor& r,
        int64_t heads,
        int64_t head_size) {
#ifdef RWKV7_USE_INPLACE_STATE
    auto state_out = state;
#else
    auto state_out = at::empty_like(state);
#endif
    auto output = at::empty_like(r);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t row_blocks = RWKV7_RANK1_ROW_BLOCKS;
    const uint32_t block_dim = static_cast<uint32_t>(heads) * row_blocks;
    const uint32_t n = static_cast<uint32_t>(head_size);
    std::array<void*, 9> pointers = {
        const_cast<void*>(state.data_ptr()),
        const_cast<void*>(w.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(kk.data_ptr()),
        const_cast<void*>(a.data_ptr()),
        const_cast<void*>(r.data_ptr()),
        const_cast<void*>(state_out.data_ptr()),
        const_cast<void*>(output.data_ptr())};
    auto launch = [stream, block_dim, pointers, row_blocks, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_state_rank1_output_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], pointers[6], pointers[7],
            pointers[8], row_blocks, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_state_rank1_output_direct", launch);
    return {state_out, output};
}

static at::Tensor rwkv7_ascendc_groupnorm_sk_direct(
        const at::Tensor& x,
        const at::Tensor& r,
        const at::Tensor& k,
        const at::Tensor& r_k,
        const at::Tensor& v,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const at::Tensor& g,
        int64_t heads,
        int64_t head_size) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t n = static_cast<uint32_t>(head_size);
    const float epsilon = static_cast<float>(head_size) * 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(head_size);
    std::array<void*, 9> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(r.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(r_k.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(weight.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(g.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, block_dim, pointers, epsilon, inv_n, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_groupnorm_sk_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], pointers[6], pointers[7],
            pointers[8], epsilon, inv_n, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_groupnorm_sk_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_lowrank_activate_direct(
        const at::Tensor& x) {
    auto out = x;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 2;
    const uint32_t elements = static_cast<uint32_t>(x.numel() / 4);
    void* x_data = const_cast<void*>(x.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, x_data, out_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_lowrank_activate_direct)(
            block_dim, stream, x_data, out_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_lowrank_activate_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_lowrank_post_direct(
        const at::Tensor& x, const at::Tensor& bias) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 4;
    const uint32_t elements = static_cast<uint32_t>(x.numel() / 4);
    void* x_data = const_cast<void*>(x.data_ptr());
    void* bias_data = const_cast<void*>(bias.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, x_data, bias_data, out_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_lowrank_post_direct)(
            block_dim, stream, x_data, bias_data, out_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_lowrank_post_direct", launch);
    return out;
}

static std::vector<at::Tensor> rwkv7_ascendc_recurrence_prep_direct(
        const at::Tensor& lowrank,
        const at::Tensor& bias,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& v_first,
        const at::Tensor& k_k,
        const at::Tensor& k_a,
        bool has_value_mix,
        int64_t heads,
        int64_t head_size) {
    auto packed = at::empty({6, k.numel()}, k.options());
    std::vector<at::Tensor> outputs;
    outputs.reserve(6);
    for (int i = 0; i < 6; ++i) outputs.push_back(packed[i].view(k.sizes()));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t block_dim = static_cast<uint32_t>(heads);
    const uint32_t hidden = static_cast<uint32_t>(k.numel());
    const uint32_t n = static_cast<uint32_t>(head_size);
    const uint32_t mix = has_value_mix ? 1U : 0U;
    std::array<void*, 14> pointers = {
        const_cast<void*>(lowrank.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(v_first.data_ptr()),
        const_cast<void*>(k_k.data_ptr()),
        const_cast<void*>(k_a.data_ptr()),
        const_cast<void*>(outputs[0].data_ptr()),
        const_cast<void*>(outputs[1].data_ptr()),
        const_cast<void*>(outputs[2].data_ptr()),
        const_cast<void*>(outputs[3].data_ptr()),
        const_cast<void*>(outputs[4].data_ptr()),
        const_cast<void*>(outputs[5].data_ptr()),
        const_cast<void*>(packed.data_ptr())};
    auto launch = [stream, block_dim, pointers, hidden, mix, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_recurrence_prep_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], pointers[6], pointers[13],
            hidden, mix, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_recurrence_prep_direct", launch);
    return outputs;
}

static std::vector<at::Tensor> rwkv7_ascendc_recurrence_state_direct(
        const at::Tensor& state,
        const at::Tensor& lowrank,
        const at::Tensor& bias,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& v_first,
        const at::Tensor& k_k,
        const at::Tensor& k_a,
        const at::Tensor& r,
        bool has_value_mix,
        int64_t heads,
        int64_t head_size) {
    TORCH_CHECK(
        head_size == 64 && RWKV7_RANK1_ROW_BLOCKS == 2,
        "fused recurrence-state requires N=64 and two row blocks");
    TORCH_CHECK(
        state.scalar_type() == at::kFloat &&
            state.numel() == heads * head_size * head_size,
        "fused recurrence-state requires B=1 fp32 [H,N,N] state");
    TORCH_CHECK(
        k.scalar_type() == at::kHalf && k.numel() == heads * head_size,
        "fused recurrence-state requires fp16 B=1 token inputs");
#ifdef RWKV7_USE_INPLACE_STATE
    auto state_out = state;
#else
    auto state_out = at::empty_like(state);
#endif
    auto output = at::empty_like(r);
    auto packed = at::empty({3, k.numel()}, k.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t row_blocks = RWKV7_RANK1_ROW_BLOCKS;
    const uint32_t block_dim = static_cast<uint32_t>(heads) * row_blocks;
    const uint32_t hidden = static_cast<uint32_t>(k.numel());
    const uint32_t mix = has_value_mix ? 1U : 0U;
    const uint32_t n = static_cast<uint32_t>(head_size);
    std::array<void*, 12> pointers = {
        const_cast<void*>(state.data_ptr()),
        const_cast<void*>(lowrank.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(k.data_ptr()),
        const_cast<void*>(v.data_ptr()),
        const_cast<void*>(v_first.data_ptr()),
        const_cast<void*>(k_k.data_ptr()),
        const_cast<void*>(k_a.data_ptr()),
        const_cast<void*>(r.data_ptr()),
        const_cast<void*>(state_out.data_ptr()),
        const_cast<void*>(output.data_ptr()),
        const_cast<void*>(packed.data_ptr())};
    auto launch = [
        stream, block_dim, pointers, hidden, mix, row_blocks, n]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_recurrence_state_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], pointers[4], pointers[5], pointers[6], pointers[7],
            pointers[8], pointers[9], pointers[10], pointers[11], hidden, mix,
            row_blocks, n);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_recurrence_state_direct", launch);
    return {
        state_out,
        output,
        packed[0].view(k.sizes()),
        packed[1].view(v.sizes()),
        packed[2].view(k.sizes())};
}

static std::vector<at::Tensor> rwkv7_ascendc_ffn_prep_direct(
        const at::Tensor& base,
        const at::Tensor& add,
        const at::Tensor& previous,
        const at::Tensor& mix,
        const at::Tensor& weight,
        const at::Tensor& bias) {
    auto packed = at::empty({3, base.numel()}, base.options());
    std::vector<at::Tensor> outputs;
    outputs.reserve(3);
    for (int i = 0; i < 3; ++i) {
        outputs.push_back(packed[i].view(base.sizes()));
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t elements = static_cast<uint32_t>(base.numel());
    const float epsilon = 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(elements);
    std::array<void*, 7> pointers = {
        const_cast<void*>(base.data_ptr()),
        const_cast<void*>(add.data_ptr()),
        const_cast<void*>(previous.data_ptr()),
        const_cast<void*>(mix.data_ptr()),
        const_cast<void*>(weight.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(packed.data_ptr())};
    auto launch = [stream, pointers, epsilon, inv_n, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_ffn_prep_direct)(
            1, stream, pointers[0], pointers[1], pointers[2], pointers[3],
            pointers[4], pointers[5], pointers[6], epsilon, inv_n, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_ffn_prep_direct", launch);
    return outputs;
}

static std::vector<at::Tensor> rwkv7_ascendc_attn_prep_direct(
        const at::Tensor& x,
        const at::Tensor& add,
        const at::Tensor& previous,
        const at::Tensor& weight,
        const at::Tensor& bias) {
    auto packed = at::empty({3, x.numel()}, x.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    const float epsilon = 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(elements);
    std::array<void*, 6> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(add.data_ptr()),
        const_cast<void*>(previous.data_ptr()),
        const_cast<void*>(weight.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(packed.data_ptr())};
    auto launch = [stream, pointers, epsilon, inv_n, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_attn_prep_direct)(
            1, stream, pointers[0], pointers[1], pointers[2], pointers[3],
            pointers[4], pointers[5], epsilon, inv_n, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_attn_prep_direct", launch);
    return {packed};
}

static at::Tensor rwkv7_ascendc_layer_norm_direct(
        const at::Tensor& x,
        const at::Tensor& weight,
        const at::Tensor& bias) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    constexpr uint32_t block_dim = 1;
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    const float epsilon = 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(elements);
    std::array<void*, 4> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(weight.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, pointers, epsilon, inv_n, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_layer_norm_direct)(
            block_dim, stream, pointers[0], pointers[1], pointers[2],
            pointers[3], epsilon, inv_n, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_layer_norm_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_add_layer_norm_direct(
        const at::Tensor& x,
        const at::Tensor& add,
        const at::Tensor& weight,
        const at::Tensor& bias) {
    auto out = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t elements = static_cast<uint32_t>(x.numel());
    const float epsilon = 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(elements);
    std::array<void*, 5> pointers = {
        const_cast<void*>(x.data_ptr()),
        const_cast<void*>(add.data_ptr()),
        const_cast<void*>(weight.data_ptr()),
        const_cast<void*>(bias.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, pointers, epsilon, inv_n, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_add_layer_norm_direct)(
            1, stream, pointers[0], pointers[1], pointers[2], pointers[3],
            pointers[4], epsilon, inv_n, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_add_layer_norm_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_concat2_direct(
        const at::Tensor& first, const at::Tensor& second) {
    auto out = at::empty({first.size(0), first.size(1) * 2}, first.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t elements = static_cast<uint32_t>(first.numel());
    void* first_data = const_cast<void*>(first.data_ptr());
    void* second_data = const_cast<void*>(second.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, first_data, second_data, out_data, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_concat2_direct)(
            1, stream, first_data, second_data, out_data, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_concat2_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_embedding_direct(
        const at::Tensor& token_ids, const at::Tensor& weight) {
    TORCH_CHECK(token_ids.numel() == 1, "direct embedding requires one token");
    TORCH_CHECK(weight.dim() == 2, "embedding weight must be rank two");
    auto out = at::empty({1, weight.size(1)}, weight.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t hidden = static_cast<uint32_t>(weight.size(1));
    void* token_data = const_cast<void*>(token_ids.data_ptr());
    void* weight_data = const_cast<void*>(weight.data_ptr());
    void* out_data = const_cast<void*>(out.data_ptr());
    auto launch = [stream, token_data, weight_data, out_data, hidden]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_embedding_direct)(
            1, stream, token_data, weight_data, out_data, hidden);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("rwkv_embedding_direct", launch);
    return out;
}

static at::Tensor rwkv7_ascendc_embedding_norm2_direct(
        const at::Tensor& token_ids,
        const at::Tensor& embedding,
        const at::Tensor& pre_weight,
        const at::Tensor& pre_bias,
        const at::Tensor& attn_weight,
        const at::Tensor& attn_bias,
        const at::Tensor& previous) {
    TORCH_CHECK(token_ids.numel() == 1, "direct embedding requires one token");
    const int64_t hidden = embedding.size(1);
    auto out = at::empty({3, hidden}, embedding.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t elements = static_cast<uint32_t>(hidden);
    const float epsilon = 1.0e-5f;
    const float inv_n = 1.0f / static_cast<float>(hidden);
    std::array<void*, 8> pointers = {
        const_cast<void*>(token_ids.data_ptr()),
        const_cast<void*>(embedding.data_ptr()),
        const_cast<void*>(pre_weight.data_ptr()),
        const_cast<void*>(pre_bias.data_ptr()),
        const_cast<void*>(attn_weight.data_ptr()),
        const_cast<void*>(attn_bias.data_ptr()),
        const_cast<void*>(previous.data_ptr()),
        const_cast<void*>(out.data_ptr())};
    auto launch = [stream, pointers, epsilon, inv_n, elements]() -> int {
        ACLRT_LAUNCH_KERNEL(rwkv_embedding_norm2_direct)(
            1, stream, pointers[0], pointers[1], pointers[2], pointers[3],
            pointers[4], pointers[5], pointers[6], pointers[7], epsilon,
            inv_n, elements);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(
        "rwkv_embedding_norm2_direct", launch);
    return out;
}

#endif

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT
#define RWKV7_RELU_SQUARE(x) rwkv7_ascendc_relu_square_direct((x))
#define RWKV7_VALUE_MIX(v, v_first, mix) \
    rwkv7_ascendc_value_mix_direct((v), (v_first), (mix))
#define RWKV7_HEAD_SCALED_ADD(x, scale, v, H, N) \
    rwkv7_ascendc_head_scaled_add_direct( \
        (x).view({1, (H), (N)}), (scale), (v).view({1, (H), (N)}), (H), (N)) \
        .view({1, (H) * (N)})
#define RWKV7_SK_OUTPUT(x, r, k, r_k, v, H, N) \
    rwkv7_ascendc_sk_output_direct( \
        (x).view({1, (H), (N)}), (r), (k), (r_k), (v), (H), (N)) \
        .view({1, (H) * (N)})
#define RWKV7_NORMALIZE_K(x, B, H, N) \
    rwkv7_ascendc_normalize_k_direct((x), (H), (N)).view({(B), (H) * (N)})
#define RWKV7_W_PRE(x) rwkv7_ascendc_w_pre_direct((x))
#ifdef RWKV7_USE_DPLR_STATE
#define RWKV7_LAYER_NORM(x, weight, bias, hidden) \
    rwkv7_ascendc_layer_norm_direct((x), (weight), (bias))
#define RWKV7_OUTER_PRODUCTS(v, k, kk, a, B, H, N)
#else
#define RWKV7_LAYER_NORM(x, weight, bias, hidden) \
    fused_ln((x), (weight), (bias), (hidden))
#define RWKV7_OUTER_PRODUCTS(v, k, kk, a, B, H, N) \
    auto _outer_products = rwkv7_ascendc_outer_products_direct( \
        (v), (k), (kk), (a), (H), (N)); \
    auto vk = _outer_products[0]; \
    auto ab = _outer_products[1];
#endif
#ifdef RWKV7_USE_MIX_PROJECT
#define RWKV7_RKV_PROJECTIONS(xr, xk, xv, rw, kw, vw, rkv_weight)
#elif defined(RWKV7_USE_RKV_BMM)
#define RWKV7_RKV_PROJECTIONS(xr, xk, xv, rw, kw, vw, rkv_weight) \
    auto _rkv = at::bmm((_sm6[7]).narrow(0, 0, 3), (rkv_weight)); \
    auto r = _rkv[0]; \
    auto k = _rkv[1]; \
    auto v = _rkv[2];
#else
#define RWKV7_RKV_PROJECTIONS(xr, xk, xv, rw, kw, vw, rkv_weight) \
    auto r = at::linear((xr), (rw)); \
    auto k = at::linear((xk), (kw)); \
    auto v = at::linear((xv), (vw));
#endif
#ifdef RWKV7_USE_MIX_PROJECT
#ifdef RWKV7_USE_RECURRENCE_PREP
#define RWKV7_LOWRANK_PROJECTIONS( \
        li, xw, xa, xg, xv, w0, w2, w2b, a0, a2, a2b, g0, g2, \
        v0, v2, v2b, lr1_weight, lr2_weight, lr_bias) \
    auto _lr1_activated = rwkv7_ascendc_lowrank_activate_direct(_lr1); \
    auto _lr2 = at::bmm(_lr1_activated, (lr2_weight));
#else
#define RWKV7_LOWRANK_PROJECTIONS( \
        li, xw, xa, xg, xv, w0, w2, w2b, a0, a2, a2b, g0, g2, \
        v0, v2, v2b, lr1_weight, lr2_weight, lr_bias) \
    auto _lr1_activated = rwkv7_ascendc_lowrank_activate_direct(_lr1); \
    auto _lr2 = at::bmm(_lr1_activated, (lr2_weight)); \
    auto _lr_post = rwkv7_ascendc_lowrank_post_direct(_lr2, (lr_bias)); \
    auto w_raw = _lr_post[0]; \
    auto a_raw = _lr_post[1]; \
    auto g_sig = _lr_post[2]; \
    auto vm_raw = _lr_post[3];
#endif
#define RWKV7_W_EXP(x) (x)
#define RWKV7_VM_SIGMOID(x) (x)
#elif defined(RWKV7_USE_LOWRANK_BMM)
#define RWKV7_LOWRANK_PROJECTIONS( \
        li, xw, xa, xg, xv, w0, w2, w2b, a0, a2, a2b, g0, g2, \
        v0, v2, v2b, lr1_weight, lr2_weight, lr_bias) \
    auto _lr1 = at::bmm((_sm6[7]).narrow(0, 3, 4), (lr1_weight)); \
    auto _lr1_activated = rwkv7_ascendc_lowrank_activate_direct(_lr1); \
    auto _lr2 = at::bmm(_lr1_activated, (lr2_weight)); \
    auto _lr_post = rwkv7_ascendc_lowrank_post_direct(_lr2, (lr_bias)); \
    auto w_raw = _lr_post[0]; \
    auto a_raw = _lr_post[1]; \
    auto g_sig = _lr_post[2]; \
    auto vm_raw = _lr_post[3];
#define RWKV7_W_EXP(x) (x)
#define RWKV7_VM_SIGMOID(x) (x)
#else
#define RWKV7_LOWRANK_PROJECTIONS( \
        li, xw, xa, xg, xv, w0, w2, w2b, a0, a2, a2b, g0, g2, \
        v0, v2, v2b, lr1_weight, lr2_weight, lr_bias) \
    auto w_raw = at::linear(at::tanh(at::linear((xw), (w0))), (w2), (w2b)); \
    auto a_raw = at::linear(at::linear((xa), (a0)), (a2), (a2b)); \
    auto g_sig = at::linear(at::sigmoid(at::linear((xg), (g0))), (g2)); \
    at::Tensor vm_raw; \
    if ((li) > 0) vm_raw = at::linear(at::linear((xv), (v0)), (v2), (v2b));
#define RWKV7_W_EXP(x) at::exp(RWKV7_W_PRE((x)))
#define RWKV7_VM_SIGMOID(x) at::sigmoid((x))
#endif
#define RWKV7_K_PREP_NORMALIZE(k, a, k_k, k_a, B, H, N, hidden) \
    auto _k_prep = rwkv7_ascendc_k_prep_normalize_direct( \
        (k), (a), (k_k).view({1, (hidden)}), (k_a).view({1, (hidden)}), \
        (H), (N)); \
    auto kk = _k_prep[0].view({(B), (hidden)}); \
    (k) = _k_prep[1]; \
    auto a_sig = _k_prep[2];
#if defined(RWKV7_USE_FUSED_RECURRENCE_STATE)
#undef RWKV7_K_PREP_NORMALIZE
#define RWKV7_K_PREP_NORMALIZE(k, a, k_k, k_a, B, H, N, hidden)
#define RWKV7_RECURRENCE_PREP( \
        li, lowrank, bias, k, v, v_first, k_k, k_a, H, N)
#define RWKV7_VALUE_PREP(li, v, v_first, vm_raw)
#define RWKV7_DECAY_PREP(w_raw)
#define RWKV7_INIT_V_FIRST(li, v, v_first)
#elif defined(RWKV7_USE_RECURRENCE_PREP)
#undef RWKV7_K_PREP_NORMALIZE
#define RWKV7_K_PREP_NORMALIZE(k, a, k_k, k_a, B, H, N, hidden)
#define RWKV7_RECURRENCE_PREP( \
        li, lowrank, bias, k, v, v_first, k_k, k_a, H, N) \
    auto _recurrence_prep = rwkv7_ascendc_recurrence_prep_direct( \
        (lowrank), (bias), (k), (v), (v_first), \
        (k_k).view({1, (H) * (N)}), (k_a).view({1, (H) * (N)}), \
        (li) > 0, (H), (N)); \
    auto w_exp = _recurrence_prep[0]; \
    (k) = _recurrence_prep[1]; \
    auto kk = _recurrence_prep[2]; \
    auto a_sig = _recurrence_prep[3]; \
    (v) = _recurrence_prep[4]; \
    auto g_sig = _recurrence_prep[5];
#define RWKV7_VALUE_PREP(li, v, v_first, vm_raw)
#define RWKV7_DECAY_PREP(w_raw)
#define RWKV7_INIT_V_FIRST(li, v, v_first) \
    if ((li) == 0) (v_first) = (v);
#else
#define RWKV7_RECURRENCE_PREP( \
        li, lowrank, bias, k, v, v_first, k_k, k_a, H, N)
#define RWKV7_VALUE_PREP(li, v, v_first, vm_raw) \
    if ((li) > 0) { \
        auto vm = RWKV7_VM_SIGMOID((vm_raw)); \
        (v) = RWKV7_VALUE_MIX((v), (v_first), vm); \
    }
#define RWKV7_DECAY_PREP(w_raw) auto w_exp = RWKV7_W_EXP((w_raw));
#define RWKV7_INIT_V_FIRST(li, v, v_first) \
    if ((li) == 0) (v_first) = (v).clone();
#endif
#ifdef RWKV7_USE_DPLR_STATE
#ifdef RWKV7_USE_FUSED_RECURRENCE_STATE
#define RWKV7_STATE_UPDATE( \
        state, w_exp, ab, vk, v, k, kk, a, r, B, H, N, dtype, \
        li, lowrank, bias, v_first, k_k, k_a) \
    auto _state_update = rwkv7_ascendc_recurrence_state_direct( \
        (state), (lowrank), (bias), (k), (v), (v_first), \
        (k_k).view({1, (H) * (N)}), (k_a).view({1, (H) * (N)}), \
        (r), (li) > 0, (H), (N)); \
    (state) = _state_update[0]; \
    auto _state_output = _state_update[1]; \
    (k) = _state_update[2]; \
    (v) = _state_update[3]; \
    auto g_sig = _state_update[4]; \
    if ((li) == 0) (v_first) = (v);
#else
#define RWKV7_STATE_UPDATE( \
        state, w_exp, ab, vk, v, k, kk, a, r, B, H, N, dtype, \
        li, lowrank, bias, v_first, k_k, k_a) \
    auto _state_update = rwkv7_ascendc_state_rank1_output_direct( \
        (state), (w_exp), (v), (k), (kk), (a), (r), (H), (N)); \
    (state) = _state_update[0]; \
    auto _state_output = _state_update[1];
#endif
#define RWKV7_STATE_OUTPUT(state_half, r, B, H, N, hidden) \
    (_state_output).view({(B), (hidden)})
#define RWKV7_GROUPNORM_SK_OUTPUT(x, r, k, r_k, v, weight, bias, g, H, N) \
    rwkv7_ascendc_groupnorm_sk_direct( \
        (x), (r), (k), (r_k), (v), (weight), (bias), (g), (H), (N))
#ifdef RWKV7_USE_INPLACE_STATE
#define RWKV7_STORE_STATE(destination, state)
#else
#define RWKV7_STORE_STATE(destination, state) (destination).copy_((state));
#endif
#else
#define RWKV7_STATE_UPDATE( \
        state, w_exp, ab, vk, v, k, kk, a, r, B, H, N, dtype, \
        li, lowrank, bias, v_first, k_k, k_a) \
    auto _state_term2 = at::matmul((state), (ab)); \
    auto _state_update = rwkv7_ascendc_state_post_direct( \
        (state), (w_exp), _state_term2, (vk), (H), (N)); \
    (state) = _state_update[0]; \
    auto state_half = _state_update[1];
#define RWKV7_STATE_OUTPUT(state_half, r, B, H, N, hidden) \
    at::matmul((state_half), (r).view({(B), (H), (N), 1})).view( \
        {(B), (hidden)})
#define RWKV7_GROUPNORM_SK_OUTPUT(x, r, k, r_k, v, weight, bias, g, H, N) \
    (RWKV7_SK_OUTPUT( \
        at::group_norm((x), (H), (weight), (bias), (double)(N) * 1e-5), \
        (r), (k), (r_k), (v), (H), (N)) * (g))
#define RWKV7_STORE_STATE(destination, state) (destination).copy_((state));
#endif
#else
#define RWKV7_LAYER_NORM(x, weight, bias, hidden) \
    fused_ln((x), (weight), (bias), (hidden))
#define RWKV7_RELU_SQUARE(x) at::relu((x)).pow(2)
#define RWKV7_VALUE_MIX(v, v_first, mix) \
    ((v) + ((v_first) - (v)) * (mix))
#define RWKV7_HEAD_SCALED_ADD(x, scale, v, H, N) \
    ((x) + ((scale) * (v).view({1, (H), (N)})).view({1, (H) * (N)}))
#define RWKV7_SK_OUTPUT(x, r, k, r_k, v, H, N) \
    ([&]() { \
        auto _sk = ((r).view({1, (H), (N)}) * (k).view({1, (H), (N)}) * \
                    (r_k).view({1, (H), (N)})).sum(-1, true); \
        return RWKV7_HEAD_SCALED_ADD((x), _sk, (v), (H), (N)); \
    }())
#define RWKV7_NORMALIZE_K(x, B, H, N) \
    (((x) / (x).norm(2, -1, true).clamp_min(1e-8)).view({(B), (H) * (N)}))
#define RWKV7_W_PRE(x) ((-EXP_HALF) * at::sigmoid((x)))
#define RWKV7_OUTER_PRODUCTS(v, k, kk, a, B, H, N) \
    auto vk = at::matmul( \
        (v).view({(B), (H), (N), 1}), (k).view({(B), (H), 1, (N)})); \
    auto ab = at::matmul( \
        (-(kk)).view({(B), (H), (N), 1}), \
        ((kk) * (a)).view({(B), (H), 1, (N)}));
#define RWKV7_RKV_PROJECTIONS(xr, xk, xv, rw, kw, vw, rkv_weight) \
    auto r = at::linear((xr), (rw)); \
    auto k = at::linear((xk), (kw)); \
    auto v = at::linear((xv), (vw));
#define RWKV7_LOWRANK_PROJECTIONS( \
        li, xw, xa, xg, xv, w0, w2, w2b, a0, a2, a2b, g0, g2, \
        v0, v2, v2b, lr1_weight, lr2_weight, lr_bias) \
    auto w_raw = at::linear(at::tanh(at::linear((xw), (w0))), (w2), (w2b)); \
    auto a_raw = at::linear(at::linear((xa), (a0)), (a2), (a2b)); \
    auto g_sig = at::linear(at::sigmoid(at::linear((xg), (g0))), (g2)); \
    at::Tensor vm_raw; \
    if ((li) > 0) vm_raw = at::linear(at::linear((xv), (v0)), (v2), (v2b));
#define RWKV7_W_EXP(x) at::exp(RWKV7_W_PRE((x)))
#define RWKV7_VM_SIGMOID(x) at::sigmoid((x))
#define RWKV7_K_PREP_NORMALIZE(k, a, k_k, k_a, B, H, N, hidden) \
    auto a_sig = at::sigmoid((a)); \
    auto kk_raw = ((k) * (k_k).view({1, (hidden)})).view({(B), (H), (N)}); \
    (k) = (k) * (1 + (a_sig - 1) * (k_a).view({1, (hidden)})); \
    auto kk = RWKV7_NORMALIZE_K(kk_raw, B, H, N);
#define RWKV7_RECURRENCE_PREP( \
        li, lowrank, bias, k, v, v_first, k_k, k_a, H, N)
#define RWKV7_VALUE_PREP(li, v, v_first, vm_raw) \
    if ((li) > 0) { \
        auto vm = RWKV7_VM_SIGMOID((vm_raw)); \
        (v) = RWKV7_VALUE_MIX((v), (v_first), vm); \
    }
#define RWKV7_DECAY_PREP(w_raw) auto w_exp = RWKV7_W_EXP((w_raw));
#define RWKV7_INIT_V_FIRST(li, v, v_first) \
    if ((li) == 0) (v_first) = (v).clone();
#define RWKV7_STATE_UPDATE( \
        state, w_exp, ab, vk, v, k, kk, a, r, B, H, N, dtype, \
        li, lowrank, bias, v_first, k_k, k_a) \
    (state) = (state) * (w_exp).view({(B), (H), 1, (N)}).to(at::kFloat) + \
        at::matmul((state), (ab).to(at::kFloat)) + (vk).to(at::kFloat); \
    auto state_half = (state).to((dtype));
#define RWKV7_STATE_OUTPUT(state_half, r, B, H, N, hidden) \
    at::matmul((state_half), (r).view({(B), (H), (N), 1})).view( \
        {(B), (hidden)})
#define RWKV7_GROUPNORM_SK_OUTPUT(x, r, k, r_k, v, weight, bias, g, H, N) \
    (RWKV7_SK_OUTPUT( \
        at::group_norm((x), (H), (weight), (bias), (double)(N) * 1e-5), \
        (r), (k), (r_k), (v), (H), (N)) * (g))
#define RWKV7_STORE_STATE(destination, state) (destination).copy_((state));
#endif

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_OPCOMMAND
static at::Tensor rwkv7_ascendc_shift_mix1(
        const at::Tensor& x, const at::Tensor& xx, const at::Tensor& mix) {
    auto y = at::empty_like(x);
    at_npu::native::OpCommand command;
    command.Name("RwkvShiftMix1")
        .Input(x)
        .Input(xx)
        .Input(mix)
        .Output(y)
        .Run();
    return y;
}
#endif

#ifdef RWKV7_USE_ADDCMUL_SHIFT_MIX
// Benchmark-only negative experiment: faster, but fp16 FMA rounding can amplify
// through recurrent state. Production builds intentionally leave this undefined.
#define RWKV7_SHIFT_MIX(base, delta, mix, hidden) \
    at::addcmul((base), (delta), (mix).view({1, (hidden)}), 1.0)
#else
#define RWKV7_SHIFT_MIX(base, delta, mix, hidden) \
    ((base) + (delta) * (mix).view({1, (hidden)}))
#endif

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT
#define RWKV7_SHIFT_INPUT(previous, base) (previous).clone()
#else
#define RWKV7_SHIFT_INPUT(previous, base) ((previous) - (base))
#endif

#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX2
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto _sm_rw = rwkv7_ascendc_shift_mix2((base), (delta), (mr), (mw)); \
    auto _sm_kv = rwkv7_ascendc_shift_mix2((base), (delta), (mk), (mv)); \
    auto _sm_ag = rwkv7_ascendc_shift_mix2((base), (delta), (ma), (mg)); \
    auto xr = _sm_rw[0]; auto xw = _sm_rw[1]; \
    auto xk = _sm_kv[0]; auto xv = _sm_kv[1]; \
    auto xa = _sm_ag[0]; auto xg = _sm_ag[1];
#define RWKV7_FFN_SHIFT_MIX(base, delta, mix, hidden) \
    rwkv7_ascendc_shift_mix2((base), (delta), (mix), (mix))[0]
#elif defined(RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT)
#ifdef RWKV7_USE_LOWRANK_BMM
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto _sm6 = rwkv7_ascendc_shift_mix6_direct( \
        (base), (delta), (mr), (mk), (mv), (mw), (ma), (mg)); \
    auto xr = _sm6[0]; auto xk = _sm6[1]; \
    auto xv = _sm6[2]; auto xw = _sm6[3]; \
    auto xa = _sm6[4]; auto xg = _sm6[5];
#elif defined(RWKV7_USE_RKV_BMM)
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto _sm6 = rwkv7_ascendc_shift_mix6_direct( \
        (base), (delta), (mr), (mk), (mv), (mw), (ma), (mg)); \
    auto xr = _sm6[0]; auto xk = _sm6[1]; \
    auto xv = _sm6[2]; auto xw = _sm6[3]; \
    auto xa = _sm6[4]; auto xg = _sm6[5];
#else
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto _sm6 = rwkv7_ascendc_shift_mix6_direct( \
        (base), (delta), (mr), (mw), (mk), (mv), (ma), (mg)); \
    auto xr = _sm6[0]; auto xw = _sm6[1]; \
    auto xk = _sm6[2]; auto xv = _sm6[3]; \
    auto xa = _sm6[4]; auto xg = _sm6[5];
#endif
#define RWKV7_FFN_SHIFT_MIX(base, delta, mix, hidden) \
    rwkv7_ascendc_shift_mix1_direct((base), (delta), (mix))
#elif defined(RWKV7_USE_ASCENDC_SHIFT_MIX1_OPCOMMAND)
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto xr = rwkv7_ascendc_shift_mix1((base), (delta), (mr)); \
    auto xw = rwkv7_ascendc_shift_mix1((base), (delta), (mw)); \
    auto xk = rwkv7_ascendc_shift_mix1((base), (delta), (mk)); \
    auto xv = rwkv7_ascendc_shift_mix1((base), (delta), (mv)); \
    auto xa = rwkv7_ascendc_shift_mix1((base), (delta), (ma)); \
    auto xg = rwkv7_ascendc_shift_mix1((base), (delta), (mg));
#define RWKV7_FFN_SHIFT_MIX(base, delta, mix, hidden) \
    rwkv7_ascendc_shift_mix1((base), (delta), (mix))
#elif defined(RWKV7_USE_FOREACH_SHIFT_MIX)
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    std::vector<at::Tensor> _sm_bases{(base), (base), (base), (base), (base), (base)}; \
    std::vector<at::Tensor> _sm_deltas{(delta), (delta), (delta), (delta), (delta), (delta)}; \
    std::vector<at::Tensor> _sm_mixes{(mr), (mw), (mk), (mv), (ma), (mg)}; \
    auto _sm_products = at::_foreach_mul(_sm_deltas, _sm_mixes); \
    auto _sm_outputs = at::_foreach_add(_sm_bases, _sm_products); \
    auto xr = _sm_outputs[0]; auto xw = _sm_outputs[1]; \
    auto xk = _sm_outputs[2]; auto xv = _sm_outputs[3]; \
    auto xa = _sm_outputs[4]; auto xg = _sm_outputs[5];
#define RWKV7_FFN_SHIFT_MIX(base, delta, mix, hidden) \
    RWKV7_SHIFT_MIX((base), (delta), (mix), (hidden))
#else
#define RWKV7_ATTN_SHIFT_MIX(base, delta, hidden, mr, mw, mk, mv, ma, mg) \
    auto xr = RWKV7_SHIFT_MIX((base), (delta), (mr), (hidden)); \
    auto xw = RWKV7_SHIFT_MIX((base), (delta), (mw), (hidden)); \
    auto xk = RWKV7_SHIFT_MIX((base), (delta), (mk), (hidden)); \
    auto xv = RWKV7_SHIFT_MIX((base), (delta), (mv), (hidden)); \
    auto xa = RWKV7_SHIFT_MIX((base), (delta), (ma), (hidden)); \
    auto xg = RWKV7_SHIFT_MIX((base), (delta), (mg), (hidden));
#define RWKV7_FFN_SHIFT_MIX(base, delta, mix, hidden) \
    RWKV7_SHIFT_MIX((base), (delta), (mix), (hidden))
#endif

#ifdef RWKV7_USE_FUSED_EMBED_NORM2
#define RWKV7_LAYER0_ATTN_NORM(base, weight, bias, hidden) \
    token_embed[1].view({1, (hidden)})
#define RWKV7_LAYER0_MIX_INPUT(h, previous, hidden) \
    token_embed.narrow(0, 1, 2).view({1, 2 * (hidden)})
#else
#define RWKV7_LAYER0_ATTN_NORM(base, weight, bias, hidden) \
    RWKV7_LAYER_NORM((base), (weight), (bias), (hidden))
#define RWKV7_LAYER0_MIX_INPUT(h, previous, hidden) \
    rwkv7_ascendc_concat2_direct((h), (previous))
#endif

#ifdef RWKV7_USE_MIX_PROJECT
#define RWKV7_STORE_ATTN_PREVIOUS(destination, value)
#ifdef RWKV7_USE_FUSED_NEXT_ATTN
#define RWKV7_ATTN_PREP( \
        li, next, base, previous, weight, bias, hidden, \
        mr, mw, mk, mv, ma, mg, \
        mix_project_weight) \
    auto h = ((li) == 0) \
        ? RWKV7_LAYER0_ATTN_NORM((base), (weight), (bias), (hidden)) \
        : (next)[1].view({1, (hidden)}); \
    auto _mix_project_input = ((li) == 0) \
        ? RWKV7_LAYER0_MIX_INPUT(h, (previous), (hidden)) \
        : (next).narrow(0, 1, 2).view({1, 2 * (hidden)}); \
    auto _mix_project = RWKV7_DIRECT_LINEAR( \
        _mix_project_input, (mix_project_weight)); \
    auto r = _mix_project.narrow(1, 0, (hidden)); \
    auto k = _mix_project.narrow(1, (hidden), (hidden)); \
    auto v = _mix_project.narrow(1, 2 * (hidden), (hidden)); \
    int64_t _mix_rank = (_mix_project.size(1) - 3 * (hidden)) / 4; \
    auto _lr1 = _mix_project.narrow( \
        1, 3 * (hidden), 4 * _mix_rank).view({4, 1, _mix_rank});
#else
#define RWKV7_ATTN_PREP( \
        li, next, base, previous, weight, bias, hidden, \
        mr, mw, mk, mv, ma, mg, \
        mix_project_weight) \
    auto h = RWKV7_LAYER_NORM((base), (weight), (bias), (hidden)); \
    auto _attn_previous = (previous); \
    auto _mix_project_input = rwkv7_ascendc_concat2_direct( \
        h, _attn_previous); \
    auto _mix_project = RWKV7_DIRECT_LINEAR( \
        _mix_project_input, (mix_project_weight)); \
    auto r = _mix_project.narrow(1, 0, (hidden)); \
    auto k = _mix_project.narrow(1, (hidden), (hidden)); \
    auto v = _mix_project.narrow(1, 2 * (hidden), (hidden)); \
    int64_t _mix_rank = (_mix_project.size(1) - 3 * (hidden)) / 4; \
    auto _lr1 = _mix_project.narrow( \
        1, 3 * (hidden), 4 * _mix_rank).view({4, 1, _mix_rank});
#endif
#else
#define RWKV7_STORE_ATTN_PREVIOUS(destination, value) \
    (destination).copy_((value));
#define RWKV7_ATTN_PREP( \
        li, next, base, previous, weight, bias, hidden, \
        mr, mw, mk, mv, ma, mg, \
        mix_project_weight) \
    auto h = RWKV7_LAYER_NORM((base), (weight), (bias), (hidden)); \
    auto _attn_delta = RWKV7_SHIFT_INPUT((previous), h); \
    RWKV7_ATTN_SHIFT_MIX( \
        h, _attn_delta, (hidden), (mr), (mw), (mk), (mv), (ma), (mg))
#endif
#ifdef RWKV7_USE_FUSED_FFN_PREP
#define RWKV7_STORE_FFN_PREVIOUS(destination, value)
#define RWKV7_FFN_PREP( \
        base, add, previous, weight, bias, mix, hidden) \
    auto _ffn_prep = rwkv7_ascendc_ffn_prep_direct( \
        (base), (add), (previous), (mix), (weight), (bias)); \
    (x) = _ffn_prep[0]; \
    auto h2 = _ffn_prep[1]; \
    auto k_ffn = _ffn_prep[2];
#else
#define RWKV7_STORE_FFN_PREVIOUS(destination, value) \
    (destination).copy_((value));
#define RWKV7_FFN_PREP( \
        base, add, previous, weight, bias, mix, hidden) \
    (x) = (base) + (add); \
    auto h2 = RWKV7_LAYER_NORM((x), (weight), (bias), (hidden)); \
    auto _ffn_delta = RWKV7_SHIFT_INPUT((previous), h2); \
    auto k_ffn = RWKV7_FFN_SHIFT_MIX(h2, _ffn_delta, (mix), (hidden));
#endif

#ifdef RWKV7_USE_FUSED_EMBED_NORM2
#define RWKV7_LAYER0_RESIDUAL(x, pre_weight, pre_bias, hidden) \
    token_embed[0].view({1, (hidden)})
#define RWKV7_INPUT_BATCH(token_embed) 1
#define RWKV7_INPUT_X(token_embed, hidden) \
    (token_embed)[0].view({1, (hidden)})
#else
#define RWKV7_LAYER0_RESIDUAL(x, pre_weight, pre_bias, hidden) \
    RWKV7_LAYER_NORM((x), (pre_weight), (pre_bias), (hidden))
#define RWKV7_INPUT_BATCH(token_embed) (token_embed).size(0)
#define RWKV7_INPUT_X(token_embed, hidden) (token_embed)
#endif

#ifdef RWKV7_USE_FUSED_NEXT_ATTN
#define RWKV7_RESIDUAL(li, x, next, pre_weight, pre_bias, hidden) \
    auto residual = ((li) == 0) \
        ? RWKV7_LAYER0_RESIDUAL( \
            (x), (pre_weight), (pre_bias), (hidden)) \
        : (next)[0].view({1, (hidden)});
#ifdef RWKV7_USE_FUSED_FINAL_NORM
#define RWKV7_FINISH_LAYER( \
        li, L, x, ffn_out, next, next_previous, next_weight, next_bias) \
    if ((li) + 1 < (L)) { \
        (next) = rwkv7_ascendc_attn_prep_direct( \
            (x), (ffn_out), (next_previous), (next_weight), (next_bias))[0]; \
    } else { \
        _last_ffn_out = (ffn_out); \
    }
#else
#define RWKV7_FINISH_LAYER( \
        li, L, x, ffn_out, next, next_previous, next_weight, next_bias) \
    if ((li) + 1 < (L)) { \
        (next) = rwkv7_ascendc_attn_prep_direct( \
            (x), (ffn_out), (next_previous), (next_weight), (next_bias))[0]; \
    } else { \
        (x) = (x) + (ffn_out); \
    }
#endif
#else
#define RWKV7_RESIDUAL(li, x, next, pre_weight, pre_bias, hidden) \
    auto residual = ((li) == 0) \
        ? RWKV7_LAYER_NORM( \
            (x), (pre_weight), (pre_bias), (hidden)) \
        : (x);
#define RWKV7_FINISH_LAYER( \
        li, L, x, ffn_out, next, next_previous, next_weight, next_bias) \
    (x) = (x) + (ffn_out);
#endif

#ifdef RWKV7_USE_FUSED_FINAL_NORM
#define RWKV7_FINAL_NORM(x, add, weight, bias, hidden) \
    rwkv7_ascendc_add_layer_norm_direct((x), (add), (weight), (bias))
#define RWKV7_FINAL_HIDDEN(x, add) ((x) + (add))
#else
#define RWKV7_FINAL_NORM(x, add, weight, bias, hidden) \
    RWKV7_LAYER_NORM((x), (weight), (bias), (hidden))
#define RWKV7_FINAL_HIDDEN(x, add) (x)
#endif

#define RWKV7_BODY \
    int64_t L = r_weights.size(); \
    int64_t B = RWKV7_INPUT_BATCH(token_embed); \
    int64_t hidden = H * N; \
    auto dtype = token_embed.scalar_type(); \
    auto x = RWKV7_INPUT_X(token_embed, hidden); \
    at::Tensor _next_attn; \
    at::Tensor _last_ffn_out; \
    for (int64_t li = 0; li < L; li++) { \
        RWKV7_RESIDUAL( \
            li, x, _next_attn, pre_norm_w[0], pre_norm_b[0], hidden) \
        auto x_prev = xpa_all[li]; \
        auto state = state_all[li]; \
        RWKV7_ATTN_PREP( \
            li, _next_attn, residual, x_prev, \
            attn_norm_w[li], attn_norm_b[li], hidden, \
            x_r_list[li], x_w_list[li], x_k_list[li], x_v_list[li], \
            x_a_list[li], x_g_list[li], mix_project_weights[li]) \
        RWKV7_RKV_PROJECTIONS( \
            xr, xk, xv, r_weights[li], k_weights[li], v_weights[li], \
            rkv_bmm_weights[li]) \
        RWKV7_LOWRANK_PROJECTIONS( \
            li, xw, xa, xg, xv, w0_list[li], w2_list[li], w2b_list[li], \
            a0_list[li], a2_list[li], a2b_list[li], g0_list[li], \
            g2_list[li], v0_list[li], v2_list[li], v2b_list[li], \
            lowrank_first_weights[li], lowrank_second_weights[li], \
            lowrank_biases[li]) \
        RWKV7_K_PREP_NORMALIZE( \
            k, a_raw, k_k_list[li], k_a_list[li], B, H, N, hidden) \
        RWKV7_RECURRENCE_PREP( \
            li, _lr2, lowrank_biases[li], k, v, v_first, k_k_list[li], \
            k_a_list[li], H, N) \
        RWKV7_INIT_V_FIRST(li, v, v_first) \
        RWKV7_VALUE_PREP(li, v, v_first, vm_raw) \
        RWKV7_DECAY_PREP(w_raw) \
        RWKV7_OUTER_PRODUCTS(v, k, kk, a_sig, B, H, N) \
        RWKV7_STATE_UPDATE( \
            state, w_exp, ab, vk, v, k, kk, a_sig, r, B, H, N, dtype, \
            li, _lr2, lowrank_biases[li], v_first, k_k_list[li], \
            k_a_list[li]) \
        RWKV7_STORE_STATE(state_all[li], state) \
        auto out = RWKV7_STATE_OUTPUT(state_half, r, B, H, N, hidden); \
        out = RWKV7_GROUPNORM_SK_OUTPUT( \
            out, r, k, r_k_list[li], v, g_norm_w_list[li], \
            g_norm_b_list[li], g_sig, H, N); \
        auto attn_out = RWKV7_DIRECT_LINEAR(out, o_weights[li]); \
        RWKV7_STORE_ATTN_PREVIOUS(xpa_all[li], h) \
        RWKV7_FFN_PREP( \
            residual, attn_out, xpf_all[li], ffn_norm_w[li], \
            ffn_norm_b[li], ffn_xk_list[li], hidden) \
        RWKV7_STORE_FFN_PREVIOUS(xpf_all[li], h2) \
        auto ffn_hidden = RWKV7_DIRECT_LINEAR( \
            k_ffn, ffn_key_weights[li]); \
        auto ffn_out = RWKV7_DIRECT_LINEAR( \
            RWKV7_RELU_SQUARE(ffn_hidden), ffn_value_weights[li]); \
        RWKV7_FINISH_LAYER( \
            li, L, x, ffn_out, _next_attn, \
            xpa_all[(li + 1) % L], \
            attn_norm_w[(li + 1) % L], attn_norm_b[(li + 1) % L]) \
    }

#define RWKV7_ARGS \
    at::Tensor token_embed, \
    std::vector<at::Tensor> r_weights, std::vector<at::Tensor> k_weights, \
    std::vector<at::Tensor> v_weights, \
    std::vector<at::Tensor> rkv_bmm_weights, \
    std::vector<at::Tensor> o_weights, \
    std::vector<at::Tensor> ffn_key_weights, std::vector<at::Tensor> ffn_value_weights, \
    std::vector<at::Tensor> w0_list, std::vector<at::Tensor> w2_list, \
    std::vector<at::Tensor> a0_list, std::vector<at::Tensor> a2_list, \
    std::vector<at::Tensor> g0_list, std::vector<at::Tensor> g2_list, \
    std::vector<at::Tensor> v0_list, std::vector<at::Tensor> v2_list, \
    std::vector<at::Tensor> w2b_list, std::vector<at::Tensor> a2b_list, std::vector<at::Tensor> v2b_list, \
    std::vector<at::Tensor> x_r_list, std::vector<at::Tensor> x_w_list, \
    std::vector<at::Tensor> x_k_list, std::vector<at::Tensor> x_v_list, \
    std::vector<at::Tensor> x_a_list, std::vector<at::Tensor> x_g_list, \
    std::vector<at::Tensor> k_k_list, std::vector<at::Tensor> k_a_list, \
    std::vector<at::Tensor> r_k_list, \
    std::vector<at::Tensor> g_norm_w_list, std::vector<at::Tensor> g_norm_b_list, \
    std::vector<at::Tensor> ffn_xk_list, \
    std::vector<at::Tensor> attn_norm_w, std::vector<at::Tensor> attn_norm_b, \
    std::vector<at::Tensor> ffn_norm_w, std::vector<at::Tensor> ffn_norm_b, \
    std::vector<at::Tensor> pre_norm_w, std::vector<at::Tensor> pre_norm_b, \
    std::vector<at::Tensor> lowrank_first_weights, \
    std::vector<at::Tensor> lowrank_second_weights, \
    std::vector<at::Tensor> lowrank_biases, \
    std::vector<at::Tensor> mix_project_weights, \
    at::Tensor state_all, at::Tensor xpa_all, at::Tensor xpf_all, \
    at::Tensor v_first, int64_t H, int64_t N

#define RWKV7_LEGACY_ARGS \
    at::Tensor token_embed, \
    std::vector<at::Tensor> r_weights, std::vector<at::Tensor> k_weights, \
    std::vector<at::Tensor> v_weights, std::vector<at::Tensor> o_weights, \
    std::vector<at::Tensor> ffn_key_weights, \
    std::vector<at::Tensor> ffn_value_weights, \
    std::vector<at::Tensor> w0_list, std::vector<at::Tensor> w2_list, \
    std::vector<at::Tensor> a0_list, std::vector<at::Tensor> a2_list, \
    std::vector<at::Tensor> g0_list, std::vector<at::Tensor> g2_list, \
    std::vector<at::Tensor> v0_list, std::vector<at::Tensor> v2_list, \
    std::vector<at::Tensor> w2b_list, std::vector<at::Tensor> a2b_list, \
    std::vector<at::Tensor> v2b_list, \
    std::vector<at::Tensor> x_r_list, std::vector<at::Tensor> x_w_list, \
    std::vector<at::Tensor> x_k_list, std::vector<at::Tensor> x_v_list, \
    std::vector<at::Tensor> x_a_list, std::vector<at::Tensor> x_g_list, \
    std::vector<at::Tensor> k_k_list, std::vector<at::Tensor> k_a_list, \
    std::vector<at::Tensor> r_k_list, \
    std::vector<at::Tensor> g_norm_w_list, \
    std::vector<at::Tensor> g_norm_b_list, \
    std::vector<at::Tensor> ffn_xk_list, \
    std::vector<at::Tensor> attn_norm_w, \
    std::vector<at::Tensor> attn_norm_b, \
    std::vector<at::Tensor> ffn_norm_w, std::vector<at::Tensor> ffn_norm_b, \
    std::vector<at::Tensor> pre_norm_w, std::vector<at::Tensor> pre_norm_b, \
    at::Tensor state_all, at::Tensor xpa_all, at::Tensor xpf_all, \
    at::Tensor v_first, int64_t H, int64_t N

at::Tensor rwkv7_decode_full(RWKV7_ARGS,
    at::Tensor lm_head_weight, at::Tensor final_norm_w, at::Tensor final_norm_b) {
    at::NoGradGuard nograd;  // inference: skip autograd graph build, lower per-op overhead
    RWKV7_BODY
    auto x_norm = RWKV7_FINAL_NORM(
        x, _last_ffn_out, final_norm_w, final_norm_b, hidden);
    return RWKV7_DIRECT_LINEAR(x_norm, lm_head_weight);
}

// Preserve the original Python call contract.  Direct benchmark builds pass
// the four packed groups and resolve the overload above; serving and existing
// callers continue to resolve this legacy-arity overload unchanged.
at::Tensor rwkv7_decode_full_legacy(
        RWKV7_LEGACY_ARGS,
        at::Tensor lm_head_weight,
        at::Tensor final_norm_w,
        at::Tensor final_norm_b) {
    std::vector<at::Tensor> empty;
    return rwkv7_decode_full(
        token_embed, r_weights, k_weights, v_weights, empty, o_weights,
        ffn_key_weights, ffn_value_weights, w0_list, w2_list, a0_list,
        a2_list, g0_list, g2_list, v0_list, v2_list, w2b_list, a2b_list,
        v2b_list, x_r_list, x_w_list, x_k_list, x_v_list, x_a_list,
        x_g_list, k_k_list, k_a_list, r_k_list, g_norm_w_list,
        g_norm_b_list, ffn_xk_list, attn_norm_w, attn_norm_b, ffn_norm_w,
        ffn_norm_b, pre_norm_w, pre_norm_b, empty, empty, empty, empty,
        state_all, xpa_all, xpf_all, v_first, H, N, lm_head_weight,
        final_norm_w, final_norm_b);
}

at::Tensor rwkv7_hidden(RWKV7_ARGS) {
    RWKV7_BODY
    return RWKV7_FINAL_HIDDEN(x, _last_ffn_out);
}

// Debug: run ONLY layer 0, return [attn_out, x_after_attn, ffn_out, x_final].
std::vector<at::Tensor> rwkv7_layer0_pieces(RWKV7_ARGS) {
    int64_t B = token_embed.size(0);
    int64_t hidden = H * N;
    auto dtype = token_embed.scalar_type();
    auto x = token_embed;
    int64_t li = 0;
    at::Tensor residual = fused_ln(x, pre_norm_w[0], pre_norm_b[0], hidden);
    auto h = fused_ln(residual, attn_norm_w[0], attn_norm_b[0], hidden);
    auto x_prev = xpa_all[0];
    auto state = state_all[0];
    auto xx = RWKV7_SHIFT_INPUT(x_prev, h);
    RWKV7_ATTN_SHIFT_MIX(h, xx, hidden, x_r_list[0], x_w_list[0], x_k_list[0], x_v_list[0], x_a_list[0], x_g_list[0])
    auto r = at::linear(xr, r_weights[0]);
    auto k = at::linear(xk, k_weights[0]);
    auto v = at::linear(xv, v_weights[0]);
    auto w_raw = at::linear(at::tanh(at::linear(xw, w0_list[0])), w2_list[0], w2b_list[0]);
    auto a_sig = at::sigmoid(at::linear(at::linear(xa, a0_list[0]), a2_list[0], a2b_list[0]));
    auto g_sig = at::linear(at::sigmoid(at::linear(xg, g0_list[0])), g2_list[0]);
    auto kk_raw = (k * k_k_list[0].view({1, hidden})).view({B, H, N});
    auto kk = (kk_raw / kk_raw.norm(2, -1, true).clamp_min(1e-8)).view({B, hidden});
    k = k * (1 + (a_sig - 1) * k_a_list[0].view({1, hidden}));
    auto w_exp = at::exp((-EXP_HALF) * at::sigmoid(w_raw));
    auto vk = at::matmul(v.view({B, H, N, 1}), k.view({B, H, 1, N}));
    auto ab = at::matmul((-kk).view({B, H, N, 1}), (kk * a_sig).view({B, H, 1, N}));
    state = state * w_exp.view({B, H, 1, N}).to(at::kFloat) + at::matmul(state, ab.to(at::kFloat)) + vk.to(at::kFloat);
    auto out = at::matmul(state.to(dtype), r.view({B, H, N, 1})).view({B, hidden});
    out = at::group_norm(
        out, H, g_norm_w_list[0].to(dtype), g_norm_b_list[0].to(dtype),
        (double)(N) * 1e-5);
    auto sk = (r.view({B, H, N}) * k.view({B, H, N}) * r_k_list[0].view({1, H, N})).sum(-1, true);
    out = out + (sk * v.view({B, H, N})).view({B, hidden});
    auto attn_out = at::linear(out * g_sig, o_weights[0]);
    auto x_after = residual + attn_out;
    auto h2 = fused_ln(x_after, ffn_norm_w[0], ffn_norm_b[0], hidden);
    auto xx_ffn = RWKV7_SHIFT_INPUT(xpf_all[0], h2);
    auto k_ffn = RWKV7_FFN_SHIFT_MIX(h2, xx_ffn, ffn_xk_list[0], hidden);
    auto ffn_out = at::linear(at::relu(at::linear(k_ffn, ffn_key_weights[0])).pow(2), ffn_value_weights[0]);
    auto x_final = x_after + ffn_out;
    return {attn_out, x_after, ffn_out, x_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_decode_full", &rwkv7_decode_full, "RWKV7 full decode step v3");
    m.def(
        "rwkv7_decode_full", &rwkv7_decode_full_legacy,
        "RWKV7 full decode step v3 (legacy weight contract)");
    m.def("rwkv7_hidden", &rwkv7_hidden, "RWKV7 pre-norm hidden state (debug)");
    m.def("rwkv7_layer0_pieces", &rwkv7_layer0_pieces, "layer0 attn/x/ffn/final (debug)");
#ifdef RWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT
    m.def(
        "rwkv7_embedding", &rwkv7_ascendc_embedding_direct,
        "B=1 graph-capturable embedding lookup");
#ifdef RWKV7_USE_FUSED_EMBED_NORM2
    m.def(
        "rwkv7_embedding_norm2", &rwkv7_ascendc_embedding_norm2_direct,
        "B=1 fused embedding and two layer norms");
#endif
#endif
}
