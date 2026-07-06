// rwkv7_prof.cpp — instrumented forward, reports per-section NPU time (us) over all layers.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <vector>
#include <chrono>
#include <map>

static const float EXP_HALF = 0.606531f;
static const double LN_EPS = 1e-5;

static inline void synchere() { c10_npu::npuSynchronizeDevice(); }

std::vector<at::Tensor> rwkv7_prof(
    at::Tensor token_embed,
    std::vector<at::Tensor> r_weights, std::vector<at::Tensor> k_weights,
    std::vector<at::Tensor> v_weights, std::vector<at::Tensor> o_weights,
    std::vector<at::Tensor> ffn_key_weights, std::vector<at::Tensor> ffn_value_weights,
    std::vector<at::Tensor> w0_list, std::vector<at::Tensor> w2_list,
    std::vector<at::Tensor> a0_list, std::vector<at::Tensor> a2_list,
    std::vector<at::Tensor> g0_list, std::vector<at::Tensor> g2_list,
    std::vector<at::Tensor> v0_list, std::vector<at::Tensor> v2_list,
    std::vector<at::Tensor> w2b_list, std::vector<at::Tensor> a2b_list, std::vector<at::Tensor> v2b_list,
    std::vector<at::Tensor> x_r_list, std::vector<at::Tensor> x_w_list,
    std::vector<at::Tensor> x_k_list, std::vector<at::Tensor> x_v_list,
    std::vector<at::Tensor> x_a_list, std::vector<at::Tensor> x_g_list,
    std::vector<at::Tensor> k_k_list, std::vector<at::Tensor> k_a_list,
    std::vector<at::Tensor> r_k_list,
    std::vector<at::Tensor> g_norm_w_list, std::vector<at::Tensor> g_norm_b_list,
    std::vector<at::Tensor> ffn_xk_list,
    std::vector<at::Tensor> attn_norm_w, std::vector<at::Tensor> attn_norm_b,
    std::vector<at::Tensor> ffn_norm_w, std::vector<at::Tensor> ffn_norm_b,
    std::vector<at::Tensor> pre_norm_w, std::vector<at::Tensor> pre_norm_b,
    at::Tensor state_all, at::Tensor xpa_all, at::Tensor xpf_all,
    at::Tensor v_first, int64_t H, int64_t N) {
    int64_t L = r_weights.size();
    int64_t B = token_embed.size(0);
    int64_t hidden = H * N;
    auto dtype = token_embed.scalar_type();
    auto x = token_embed;
    std::map<std::string,double> ts;
    auto T0=[&](){ return std::chrono::steady_clock::now(); };
    auto US=[&](auto a,auto b){ return std::chrono::duration<double,std::micro>(b-a).count(); };
    for (int64_t li = 0; li < L; li++) {
        at::Tensor residual; at::Tensor h, x_prev, state, xx, xr, xw, xk, xv, xa, xg;
        at::Tensor r, k, v, w_raw, a_sig, g_sig, kk_raw, kk, w_exp, vk, ab, out, sk, attn_out, h2, xx_f, kf, fo;
        auto tA=T0();
        residual = (li==0)? at::layer_norm(x,{hidden},pre_norm_w[0],pre_norm_b[0],LN_EPS) : x;
        h = at::layer_norm(residual,{hidden},attn_norm_w[li],attn_norm_b[li],LN_EPS);
        x_prev=xpa_all[li]; state=state_all[li];
        synchere(); ts["norms"]+=US(tA,T0());

        xx = x_prev - h;
        xr = h + xx*x_r_list[li].view({1,hidden});
        xw = h + xx*x_w_list[li].view({1,hidden});
        xk = h + xx*x_k_list[li].view({1,hidden});
        xv = h + xx*x_v_list[li].view({1,hidden});
        xa = h + xx*x_a_list[li].view({1,hidden});
        xg = h + xx*x_g_list[li].view({1,hidden});
        synchere(); ts["shiftmix"]+=US(tA,T0());

        r = at::linear(xr, r_weights[li]);
        k = at::linear(xk, k_weights[li]);
        v = at::linear(xv, v_weights[li]);
        synchere(); ts["proj_rkv"]+=US(tA,T0());

        w_raw = at::linear(at::tanh(at::linear(xw, w0_list[li])), w2_list[li], w2b_list[li]);
        a_sig = at::sigmoid(at::linear(at::linear(xa, a0_list[li]), a2_list[li], a2b_list[li]));
        g_sig = at::linear(at::sigmoid(at::linear(xg, g0_list[li])), g2_list[li]);
        synchere(); ts["loras"]+=US(tA,T0());

        kk_raw = (k*k_k_list[li].view({1,hidden})).view({B,H,N});
        kk = (kk_raw/kk_raw.norm(2,-1,true).clamp_min(1e-8)).view({B,hidden});
        k = k*(1+(a_sig-1)*k_a_list[li].view({1,hidden}));
        if(li==0){v_first=v.clone();} else {auto vm=at::sigmoid(at::linear(at::linear(xv,v0_list[li]),v2_list[li],v2b_list[li])); v=v+(v_first-v)*vm;}
        synchere(); ts["kk+vmod"]+=US(tA,T0());

        w_exp = at::exp((-EXP_HALF)*at::sigmoid(w_raw));
        vk = at::matmul(v.view({B,H,N,1}), k.view({B,H,1,N}));
        ab = at::matmul((-kk).view({B,H,N,1}), (kk*a_sig).view({B,H,1,N}));
        state = state*w_exp.view({B,H,1,N}).to(at::kFloat) + at::matmul(state, ab.to(at::kFloat)) + vk.to(at::kFloat);
        synchere(); ts["wkv"]+=US(tA,T0());

        out = at::matmul(state.to(dtype), r.view({B,H,N,1})).view({B,hidden});
        out = at::group_norm(out, H, g_norm_w_list[li], g_norm_b_list[li], (double)(N)*1e-5);
        sk = (r.view({B,H,N})*k.view({B,H,N})*r_k_list[li].view({1,H,N})).sum(-1,true);
        out = out + (sk*v.view({B,H,N})).view({B,hidden});
        attn_out = at::linear(out*g_sig, o_weights[li]);
        x = residual + attn_out;
        synchere(); ts["gn+sk+o"]+=US(tA,T0());

        h2 = at::layer_norm(x,{hidden},ffn_norm_w[li],ffn_norm_b[li],LN_EPS);
        xx_f = xpf_all[li]-h2;
        kf = h2 + xx_f*ffn_xk_list[li].view({1,hidden});
        fo = at::linear(at::relu(at::linear(kf, ffn_key_weights[li])).pow(2), ffn_value_weights[li]);
        x = x + fo;
        synchere(); ts["ffn"]+=US(tA,T0());
    }
    std::vector<at::Tensor> result;
    for (auto &kv : ts) { result.push_back(at::tensor({kv.second}, at::kFloat)); result.push_back(at::tensor({(double)(kv.second/L)})); }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_prof", &rwkv7_prof, "profiled RWKV7 forward");
}
