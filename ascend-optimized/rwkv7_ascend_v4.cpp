// rwkv7_ascend_v4.cpp — v3 + fused shift-mix via aclnnRwkvShiftMix6 (BN=1, 6 outputs).
// Replaces the 13-op per-layer shift-mix (sub + 6mul + 6add) with 1 aclnn kernel
// call (+ 1 eager sub for xx). Proven 2.39x on the shift-mix block in isolation.
#include <torch/extension.h>
#include <vector>
#include "fused_shiftmix6.h"

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
        auto _sm = fused_shiftmix6(h, x_prev, x_r_list[li], x_w_list[li], x_k_list[li], x_v_list[li], x_a_list[li], x_g_list[li]); \
        auto xr = _sm[0]; auto xw = _sm[1]; auto xk = _sm[2]; auto xv = _sm[3]; auto xa = _sm[4]; auto xg = _sm[5]; \
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
        auto out = at::matmul(state.to(dtype), r.view({B, H, N, 1})).view({B, hidden}); \
        out = at::group_norm(out, H, g_norm_w_list[li], g_norm_b_list[li], (double)(N) * 1e-5); \
        auto sk = (r.view({B, H, N}) * k.view({B, H, N}) * r_k_list[li].view({1, H, N})).sum(-1, true); \
        out = out + (sk * v.view({B, H, N})).view({B, hidden}); \
        auto attn_out = at::linear(out * g_sig, o_weights[li]); \
        x = residual + attn_out; \
        auto h2 = fused_ln(x, ffn_norm_w[li], ffn_norm_b[li], hidden); \
        auto xx_ffn = xpf_all[li] - h2; \
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
    at::NoGradGuard nograd;
    RWKV7_BODY
    auto x_norm = fused_ln(x, final_norm_w, final_norm_b, hidden);
    return at::linear(x_norm, lm_head_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_decode_full", &rwkv7_decode_full, "RWKV7 full decode step v4 (fused shift-mix)");
}
