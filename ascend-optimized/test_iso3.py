"""Dump every intermediate: MODULE call path vs INLINE mirror path, 1 layer."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu
import torch.nn.functional as F
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM, NativeRWKV7Attention
from rwkv7_hf.native import _init_state_batched, EXP_HALF

H, N, dev = 12, 64, "npu:0"
cfg = RWKV7Config(vocab_size=65536, hidden_size=H*N, num_hidden_layers=1,
                  num_heads=H, head_dim=N, intermediate_size=4*H*N)
torch.manual_seed(0)
model = NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
base=model.model; hidden=H*N; B=1
emb=base.embeddings(torch.tensor([42],device=dev))
layer0=base.layers[0]; L0=layer0.attn

with torch.no_grad():
    state,xpa,xpf,v_first=_init_state_batched(model,1,dev,torch.float16)
    residual = layer0.pre_norm(emb)
    h = layer0.attn_norm(residual)

    # --- MODULE path: monkeypatch attn_step_batched internals is hard;
    #     instead re-run the module and capture via a hook on each sub-module ---
    # We'll capture inputs/outputs of r_proj etc. via forward hooks.
    cache = {}
    def mk(name):
        def hook(mod, inp, out, key=name):
            cache[key] = out.detach()
        return hook
    hs=[]
    hs.append(L0.r_proj.register_forward_hook(mk("r")))
    hs.append(L0.k_proj.register_forward_hook(mk("k_raw")))
    hs.append(L0.v_proj.register_forward_hook(mk("v")))
    hs.append(L0.o_proj.register_forward_hook(mk("attn_out")))
    hs.append(layer0.ffn_norm.register_forward_hook(mk("h2")))
    hs.append(layer0.ffn.key.register_forward_hook(mk("ffn_key")))
    a_mod, _, _, _ = L0(h, xpa[0], v_first, state[0])
    for hh in hs: hh.remove()

    # --- INLINE path (mirror) ---
    layer=L0; x_prev=xpa[0]; st=state[0]
    xx = x_prev - h
    xr = h + xx*layer.x_r.reshape(1,hidden)
    xw = h + xx*layer.x_w.reshape(1,hidden)
    xk = h + xx*layer.x_k.reshape(1,hidden)
    xv = h + xx*layer.x_v.reshape(1,hidden)
    xa = h + xx*layer.x_a.reshape(1,hidden)
    xg = h + xx*layer.x_g.reshape(1,hidden)
    r = layer.r_proj(xr)
    k = layer.k_proj(xk)
    v = layer.v_proj(xv)
    w = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw)))
    a = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xa)))
    g = torch.sigmoid(layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xg))))
    kk = F.normalize((k*layer.k_k.reshape(1,hidden)).view(B,H,N),dim=-1,p=2).view(B,hidden)
    k = k*(1+(a-1)*layer.k_a.reshape(1,hidden))
    w = torch.exp(-EXP_HALF*torch.sigmoid(w.float()))
    vk = v.view(B,H,N,1)@k.view(B,H,1,N)
    ab = (-kk).view(B,H,N,1)@(kk*a).view(B,H,1,N)
    stt = st*w.view(B,H,1,N) + st@ab.float() + vk.float()
    out = stt.to(h.dtype)@r.view(B,H,N,1); out=out.view(B,hidden)
    out = F.group_norm(out,num_groups=H,weight=layer.g_norm.weight,bias=layer.g_norm.bias,eps=N*1e-5)
    sk = (r.view(B,H,N)*k.view(B,H,N)*layer.r_k.reshape(1,H,N)).sum(-1,keepdim=True)
    out = out+(sk*v.view(B,H,N)).view(B,hidden)
    attn_out_i = layer.o_proj(out*g)

def C(name, mod_out, inl):
    mod_out=mod_out.float().cpu(); inl=inl.float().cpu()
    cos=F.cosine_similarity(mod_out.flatten().unsqueeze(0),inl.flatten().unsqueeze(0)).item()
    print("  %-10s cos=%+.6f maxabs=%.6f%s"%(name,cos,(mod_out-inl).abs().max().item(),"  <<<"if cos<0.999 else ""),flush=True)

print("MODULE vs INLINE intermediates:",flush=True)
C("r", cache["r"], r)
C("k_raw", cache["k_raw"], layer.k_proj(xk))   # k before mod; inline k got overwritten
C("v", cache["v"], v)
C("attn_out", cache["attn_out"], attn_out_i)
