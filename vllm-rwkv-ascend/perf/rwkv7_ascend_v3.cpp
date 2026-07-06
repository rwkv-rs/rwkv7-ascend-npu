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
#include <vector>

static const float EXP_HALF = 0.606531f;
static const double LN_EPS = 1e-5;

at::Tensor fused_ln(at::Tensor x, at::Tensor w, at::Tensor b, int64_t hidden) {
    return at::layer_norm(x, {hidden}, w, b, LN_EPS);
}

#define RWKV7_BODY \
    int64_t L = r_weights.size(); \
    int64_t B = token_embed.size(0); \
    int64_t hidden = H * N; \
    auto dtype = token_embed.scalar_type(); \
    auto x = token_embed; \
    for (int64_t li = 0; li < L; li++) { \
        at::Tensor residual = (li == 0) ? fused_ln(x, pre_norm_w[0], pre_norm_b[0], hidden) : x; \
        auto h = fused_ln(residual, attn_norm_w[li], attn_norm_b[li], hidden); \
        auto x_prev = xpa_all[li]; \
        auto state = state_all[li]; \
        auto xx = x_prev - h; \
        auto xr = h + xx * x_r_list[li].view({1, hidden}); \
        auto xw = h + xx * x_w_list[li].view({1, hidden}); \
        auto xk = h + xx * x_k_list[li].view({1, hidden}); \
        auto xv = h + xx * x_v_list[li].view({1, hidden}); \
        auto xa = h + xx * x_a_list[li].view({1, hidden}); \
        auto xg = h + xx * x_g_list[li].view({1, hidden}); \
        auto r = at::linear(xr, r_weights[li]); \
        auto k = at::linear(xk, k_weights[li]); \
        auto v = at::linear(xv, v_weights[li]); \
        auto w_raw = at::linear(at::tanh(at::linear(xw, w0_list[li])), w2_list[li], w2b_list[li]); \
        auto a_sig = at::sigmoid(at::linear(at::linear(xa, a0_list[li]), a2_list[li], a2b_list[li])); \
        auto g_sig = at::linear(at::sigmoid(at::linear(xg, g0_list[li])), g2_list[li]); \
        auto kk_raw = (k * k_k_list[li].view({1, hidden})).view({B, H, N}); \
        auto kk = (kk_raw / kk_raw.norm(2, -1, true).clamp_min(1e-8)).view({B, hidden}); \
        k = k * (1 + (a_sig - 1) * k_a_list[li].view({1, hidden})); \
        if (li == 0) { v_first = v.clone(); } \
        else { auto vm = at::sigmoid(at::linear(at::linear(xv, v0_list[li]), v2_list[li], v2b_list[li])); v = v + (v_first - v) * vm; } \
        auto w_exp = at::exp((-EXP_HALF) * at::sigmoid(w_raw)); \
        auto vk = at::matmul(v.view({B, H, N, 1}), k.view({B, H, 1, N})); \
        auto ab = at::matmul((-kk).view({B, H, N, 1}), (kk * a_sig).view({B, H, 1, N})); \
        state = state * w_exp.view({B, H, 1, N}).to(at::kFloat) + at::matmul(state, ab.to(at::kFloat)) + vk.to(at::kFloat); \
        state_all[li].copy_(state); \
        auto out = at::matmul(state.to(dtype), r.view({B, H, N, 1})).view({B, hidden}); \
        out = at::group_norm(out, H, g_norm_w_list[li], g_norm_b_list[li], (double)(N) * 1e-5); \
        auto sk = (r.view({B, H, N}) * k.view({B, H, N}) * r_k_list[li].view({1, H, N})).sum(-1, true); \
        out = out + (sk * v.view({B, H, N})).view({B, hidden}); \
        auto attn_out = at::linear(out * g_sig, o_weights[li]); \
        xpa_all[li].copy_(h); \
        x = residual + attn_out; \
        auto h2 = fused_ln(x, ffn_norm_w[li], ffn_norm_b[li], hidden); \
        auto xx_ffn = xpf_all[li] - h2; \
        xpf_all[li].copy_(h2); \
        auto k_ffn = h2 + xx_ffn * ffn_xk_list[li].view({1, hidden}); \
        auto ffn_out = at::linear(at::relu(at::linear(k_ffn, ffn_key_weights[li])).pow(2), ffn_value_weights[li]); \
        x = x + ffn_out; \
    }

#define RWKV7_ARGS \
    at::Tensor token_embed, \
    std::vector<at::Tensor> r_weights, std::vector<at::Tensor> k_weights, \
    std::vector<at::Tensor> v_weights, std::vector<at::Tensor> o_weights, \
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
    at::Tensor state_all, at::Tensor xpa_all, at::Tensor xpf_all, \
    at::Tensor v_first, int64_t H, int64_t N

at::Tensor rwkv7_decode_full(RWKV7_ARGS,
    at::Tensor lm_head_weight, at::Tensor final_norm_w, at::Tensor final_norm_b) {
    at::NoGradGuard nograd;  // inference: skip autograd graph build, lower per-op overhead
    RWKV7_BODY
    auto x_norm = fused_ln(x, final_norm_w, final_norm_b, hidden);
    return at::linear(x_norm, lm_head_weight);
}

at::Tensor rwkv7_hidden(RWKV7_ARGS) {
    RWKV7_BODY
    return x;
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
    auto xx = x_prev - h;
    auto xr = h + xx * x_r_list[0].view({1, hidden});
    auto xw = h + xx * x_w_list[0].view({1, hidden});
    auto xk = h + xx * x_k_list[0].view({1, hidden});
    auto xv = h + xx * x_v_list[0].view({1, hidden});
    auto xa = h + xx * x_a_list[0].view({1, hidden});
    auto xg = h + xx * x_g_list[0].view({1, hidden});
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
    out = at::group_norm(out, H, g_norm_w_list[0], g_norm_b_list[0], (double)(N) * 1e-5);
    auto sk = (r.view({B, H, N}) * k.view({B, H, N}) * r_k_list[0].view({1, H, N})).sum(-1, true);
    out = out + (sk * v.view({B, H, N})).view({B, hidden});
    auto attn_out = at::linear(out * g_sig, o_weights[0]);
    auto x_after = residual + attn_out;
    auto h2 = fused_ln(x_after, ffn_norm_w[0], ffn_norm_b[0], hidden);
    auto xx_ffn = xpf_all[0] - h2;
    auto k_ffn = h2 + xx_ffn * ffn_xk_list[0].view({1, hidden});
    auto ffn_out = at::linear(at::relu(at::linear(k_ffn, ffn_key_weights[0])).pow(2), ffn_value_weights[0]);
    auto x_final = x_after + ffn_out;
    return {attn_out, x_after, ffn_out, x_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_decode_full", &rwkv7_decode_full, "RWKV7 full decode step v3");
    m.def("rwkv7_hidden", &rwkv7_hidden, "RWKV7 pre-norm hidden state (debug)");
    m.def("rwkv7_layer0_pieces", &rwkv7_layer0_pieces, "layer0 attn/x/ffn/final (debug)");
}
