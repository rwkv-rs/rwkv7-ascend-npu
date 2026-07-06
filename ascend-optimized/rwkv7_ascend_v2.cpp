// rwkv7_ascend_v2.cpp — Optimized single-call C++ forward for RWKV-7 on Ascend NPU.
//
// Optimizations over v1 (all reduce CANN kernel-launch count, the dominant cost):
//   1. Batched shift-mix: stack [x_r..x_g] -> 1 broadcast mul + 1 add + free selects
//      (12 elementwise ops -> 3).  Biggest single win.
//   2. r/k/v via one bmm: stack inputs [3,B,hidden] x stacked weights [3,hidden,hidden]
//      (3 linears -> 1 batched matmul).
//   3. Fewer dtype conversions: w_exp computed in fp16 (sigmoid range is safe),
//      let fp32 state promotion handle accumulation.
//
// Build: torch.utils.cpp_extension.load(name="rwkv7_ascend_v2", sources=[...])

#include <torch/extension.h>
#include <vector>

at::Tensor l2_norm_last(at::Tensor x) {
    auto n = x.norm(2, /*dim=*/-1, /*keepdim=*/true).clamp_min(1e-8);
    return x / n;
}

at::Tensor manual_group_norm(at::Tensor x, at::Tensor weight, at::Tensor bias,
                             int64_t H, int64_t N, float eps) {
    int64_t B = x.size(0);
    auto x_view = x.view({B, H, N});
    auto mean = x_view.mean(/*dim=*/-1, /*keepdim=*/true);
    auto centered = x_view - mean;
    auto var = centered.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normed = centered * torch::rsqrt(var + eps);
    auto out = normed.view({B, H * N});
    return out * weight + bias;
}

at::Tensor rwkv7_decode_full(
    at::Tensor token_embed,
    std::vector<at::Tensor> r_weights, std::vector<at::Tensor> k_weights,
    std::vector<at::Tensor> v_weights, std::vector<at::Tensor> o_weights,
    std::vector<at::Tensor> ffn_key_weights, std::vector<at::Tensor> ffn_value_weights,
    std::vector<at::Tensor> w0_list, std::vector<at::Tensor> w2_list,
    std::vector<at::Tensor> a0_list, std::vector<at::Tensor> a2_list,
    std::vector<at::Tensor> g0_list, std::vector<at::Tensor> g2_list,
    std::vector<at::Tensor> v0_list, std::vector<at::Tensor> v2_list,
    std::vector<at::Tensor> x_r_list, std::vector<at::Tensor> x_w_list,
    std::vector<at::Tensor> x_k_list, std::vector<at::Tensor> x_v_list,
    std::vector<at::Tensor> x_a_list, std::vector<at::Tensor> x_g_list,
    std::vector<at::Tensor> k_k_list, std::vector<at::Tensor> k_a_list,
    std::vector<at::Tensor> r_k_list,
    std::vector<at::Tensor> g_norm_w_list, std::vector<at::Tensor> g_norm_b_list,
    std::vector<at::Tensor> ffn_xk_list,
    at::Tensor state_all, at::Tensor xpa_all, at::Tensor xpf_all,
    at::Tensor v_first,
    int64_t H, int64_t N,
    at::Tensor lm_head_weight, at::Tensor norm_weight
) {
    int64_t L = r_weights.size();
    int64_t B = token_embed.size(0);
    int64_t hidden = H * N;
    float EXP_HALF = 0.606531f;
    auto dtype = token_embed.scalar_type();

    auto x = token_embed;

    for (int64_t li = 0; li < L; li++) {
        auto x_prev = xpa_all[li];
        auto state = state_all[li];

        // --- Batched shift-mix ---
        // mix [6, hidden], xx [B, hidden] -> x6 [6, B, hidden] in 1 mul + 1 add.
        // Use explicit .expand to force the leading dim to 6 (avoids the
        // [1,6,hidden] broadcast ambiguity that broke v2's first cut).
        auto mix = torch::stack(
            {x_r_list[li], x_w_list[li], x_k_list[li],
             x_v_list[li], x_a_list[li], x_g_list[li]}, 0);   // [6, hidden]
        auto xx = x_prev - x;                                  // [B, hidden]
        auto x_exp = x.unsqueeze(0).expand({6, B, hidden});    // [6, B, hidden]
        auto xx_exp = xx.unsqueeze(0).expand({6, B, hidden});  // [6, B, hidden]
        auto mix_exp = mix.view({6, 1, hidden}).expand({6, B, hidden});
        auto x6 = x_exp + xx_exp * mix_exp;                    // [6, B, hidden]
        auto xr = x6.select(0, 0);
        auto xw = x6.select(0, 1);
        auto xk = x6.select(0, 2);
        auto xv = x6.select(0, 3);
        auto xa = x6.select(0, 4);
        auto xg = x6.select(0, 5);

        // --- r/k/v via single batched matmul ---
        // x_rkv [3, B, hidden], W_rkv [3, hidden, hidden] (weight.T for Linear)
        auto x_rkv = torch::stack({xr, xk, xv}, 0);            // [3, B, hidden]
        auto r_w_t = r_weights[li].t().unsqueeze(0);           // [1, hidden, hidden]
        auto k_w_t = k_weights[li].t().unsqueeze(0);
        auto v_w_t = v_weights[li].t().unsqueeze(0);
        auto W_rkv = torch::cat({r_w_t, k_w_t, v_w_t}, 0);     // [3, hidden, hidden]
        auto rkv = torch::bmm(x_rkv, W_rkv);                   // [3, B, hidden]
        auto r = rkv.select(0, 0);
        auto k = rkv.select(0, 1);
        auto v = rkv.select(0, 2);

        auto w_raw = at::linear(at::tanh(at::linear(xw, w0_list[li])), w2_list[li]);
        auto a_sig = at::sigmoid(at::linear(at::linear(xa, a0_list[li]), a2_list[li]));
        auto g_sig = at::sigmoid(at::linear(at::sigmoid(at::linear(xg, g0_list[li])), g2_list[li]));

        // kk normalization
        auto kk_raw = (k * k_k_list[li].view({1, hidden})).view({B, H, N});
        auto kk = l2_norm_last(kk_raw).view({B, hidden});
        k = k * (1 + (a_sig - 1) * k_a_list[li].view({1, hidden}));

        if (li == 0) {
            v_first = v.clone();
        } else {
            auto v_mix = at::sigmoid(at::linear(at::linear(xv, v0_list[li]), v2_list[li]));
            v = v + (v_first - v) * v_mix;
        }

        // WKV state update — w_exp in fp16 (sigmoid output in [0,1], exp safe)
        auto w_exp = at::exp((-EXP_HALF) * at::sigmoid(w_raw));
        auto vk = at::matmul(v.view({B, H, N, 1}), k.view({B, H, 1, N}));
        auto ab = at::matmul((-kk).view({B, H, N, 1}), (kk * a_sig).view({B, H, 1, N}));
        state = state * w_exp.view({B, H, 1, N}).to(at::kFloat) +
                at::matmul(state, ab.to(at::kFloat)) + vk.to(at::kFloat);

        auto out = at::matmul(state.to(dtype), r.view({B, H, N, 1})).view({B, hidden});
        out = manual_group_norm(out, g_norm_w_list[li], g_norm_b_list[li], H, N, (float)(N) * 1e-5f);
        auto sk = (r.view({B, H, N}) * k.view({B, H, N}) * r_k_list[li].view({1, H, N}))
                     .sum(/*dim=*/-1, /*keepdim=*/true);
        out = out + (sk * v.view({B, H, N})).view({B, hidden});
        auto attn_out = at::linear(out * g_sig, o_weights[li]);

        // --- CMix (FFN) ---
        auto x_prev_ffn = xpf_all[li];
        auto xx_ffn = x_prev_ffn - x;
        auto k_ffn = x + xx_ffn * ffn_xk_list[li].view({1, hidden});
        auto k_act = at::relu(at::linear(k_ffn, ffn_key_weights[li])).pow(2);
        auto ffn_out = at::linear(k_act, ffn_value_weights[li]);

        x = x + attn_out + ffn_out;
    }

    auto logits_norm = x * norm_weight;
    auto logits = at::linear(logits_norm, lm_head_weight);
    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_decode_full", &rwkv7_decode_full, "RWKV7 full decode step v2");
}
