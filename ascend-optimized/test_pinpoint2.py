"""Exhaustively diff every intermediate: faithful attn_step_batched (ref) vs mirror."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu
import torch.nn.functional as F
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.native import _init_state_batched, EXP_HALF

H, N, dev = 12, 64, "npu:0"
cfg = RWKV7Config(vocab_size=65536, hidden_size=H*N, num_hidden_layers=2,
                  num_heads=H, head_dim=N, intermediate_size=4*H*N)
torch.manual_seed(0)
model = NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
base = model.model; hidden = H*N; B=1
emb = base.embeddings(torch.tensor([42], device=dev))
layer0 = base.layers[0]; L0 = layer0.attn

with torch.no_grad():
    state, xpa, xpf, v_first = _init_state_batched(model, 1, dev, torch.float16)
    residual = layer0.pre_norm(emb)
    x = layer0.attn_norm(residual)         # attn input
    x_prev = xpa[0]
    st = state[0]

def C(name, ref, mir):
    ref=ref.float().cpu(); mir=mir.float().cpu()
    cos=F.cosine_similarity(ref.flatten().unsqueeze(0), mir.flatten().unsqueeze(0)).item()
    print("  %-10s cos=%+.5f maxabs=%.5f %s" % (name, cos, (ref-mir).abs().max().item(),
          "  <<<" if cos<0.99 else ""), flush=True)

with torch.no_grad():
    # ---- REF: exact copy of attn_step_batched ----
    layer=L0; layer_id=0
    xx = x_prev - x
    xr = x + xx * layer.x_r.reshape(1, hidden)
    xw = x + xx * layer.x_w.reshape(1, hidden)
    xk = x + xx * layer.x_k.reshape(1, hidden)
    xv = x + xx * layer.x_v.reshape(1, hidden)
    xa = x + xx * layer.x_a.reshape(1, hidden)
    xg = x + xx * layer.x_g.reshape(1, hidden)
    r = layer.r_proj(xr)
    w = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw)))
    k = layer.k_proj(xk)
    v = layer.v_proj(xv)
    a = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xa)))
    g = torch.sigmoid(layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xg))))
    kk = F.normalize((k * layer.k_k.reshape(1, hidden)).view(B, H, N), dim=-1, p=2).view(B, hidden)
    k = k * (1 + (a - 1) * layer.k_a.reshape(1, hidden))
    vf = v
    w = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    state_r = st * w.view(B, H, 1, N) + st @ ab.float() + vk.float()
    out_r = state_r.to(x.dtype) @ r.view(B, H, N, 1)
    out_r = out_r.view(B, hidden)
    out_r = F.group_norm(out_r, num_groups=H, weight=layer.g_norm.weight, bias=layer.g_norm.bias, eps=N*1e-5)
    sk = (r.view(B, H, N) * k.view(B, H, N) * layer.r_k.reshape(1, H, N)).sum(dim=-1, keepdim=True)
    out_r = out_r + (sk * v.view(B, H, N)).view(B, hidden)
    out_r = layer.o_proj(out_r * g)

    # ---- MIRROR: my C++ logic ----
    xxm = x_prev - x
    xrm = x + xxm * layer.x_r.reshape(1, hidden)
    xwm = x + xxm * layer.x_w.reshape(1, hidden)
    xkm = x + xxm * layer.x_k.reshape(1, hidden)
    xvm = x + xxm * layer.x_v.reshape(1, hidden)
    xam = x + xxm * layer.x_a.reshape(1, hidden)
    xgm = x + xxm * layer.x_g.reshape(1, hidden)
    rm = layer.r_proj(xrm)
    wm = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xwm)))
    km = layer.k_proj(xkm)
    vm = layer.v_proj(xvm)
    am = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xam)))
    gm = torch.sigmoid(layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xgm))))
    kkm = F.normalize((km * layer.k_k.reshape(1, hidden)).view(B, H, N), dim=-1, p=2).view(B, hidden)
    km = km * (1 + (am - 1) * layer.k_a.reshape(1, hidden))
    wm = torch.exp(-EXP_HALF * torch.sigmoid(wm.float()))
    vkm = vm.view(B, H, N, 1) @ km.view(B, H, 1, N)
    abm = (-kkm).view(B, H, N, 1) @ (kkm * am).view(B, H, 1, N)
    state_m = st * wm.view(B, H, 1, N) + st @ abm.float() + vkm.float()
    out_m = state_m.to(x.dtype) @ rm.view(B, H, N, 1)
    out_m = out_m.view(B, hidden)
    out_m = F.group_norm(out_m, num_groups=H, weight=layer.g_norm.weight, bias=layer.g_norm.bias, eps=N*1e-5)
    skm = (rm.view(B, H, N) * km.view(B, H, N) * layer.r_k.reshape(1, H, N)).sum(dim=-1, keepdim=True)
    out_m = out_m + (skm * vm.view(B, H, N)).view(B, hidden)
    out_m = layer.o_proj(out_m * gm)

print("REF vs MIRROR intermediates:", flush=True)
C("r", r, rm); C("k(orig-before-mod)", layer.k_proj(xkm), layer.k_proj(xkm))  # placeholder
C("w_raw", layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw))), wm if False else layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xwm))))
C("kk", F.normalize((layer.k_proj(xk)*layer.k_k.reshape(1,hidden)).view(B,H,N),dim=-1,p=2).view(B,hidden), kkm)
C("w(dec)", torch.exp(-EXP_HALF*torch.sigmoid(layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw))).float())), wm)
C("vk", vk, vkm); C("ab", ab, abm); C("state", state_r, state_m)
C("out_raw", (state_r.to(x.dtype) @ r.view(B,H,N,1)).view(B,hidden), (state_m.to(x.dtype) @ rm.view(B,H,N,1)).view(B,hidden))
C("out_gn", F.group_norm((state_r.to(x.dtype) @ r.view(B,H,N,1)).view(B,hidden),num_groups=H,weight=layer.g_norm.weight,bias=layer.g_norm.bias,eps=N*1e-5), F.group_norm((state_m.to(x.dtype) @ rm.view(B,H,N,1)).view(B,hidden),num_groups=H,weight=layer.g_norm.weight,bias=layer.g_norm.bias,eps=N*1e-5))
C("sk", sk, skm)
C("FINAL_out", out_r, out_m)
