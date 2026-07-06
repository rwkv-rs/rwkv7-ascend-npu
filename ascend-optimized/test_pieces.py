"""Compare C++ layer0 pieces (attn_out, x_after_attn, ffn_out, x_final) vs Python."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu
import torch.nn.functional as F
from torch.utils.cpp_extension import load
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
mod3 = load(name="rwkv7_ascend_v3", sources=["/root/rwkv7_ascend_v3.cpp"],
            verbose=False, extra_cflags=["-O3","-std=c++17"])
base = model.model; hidden=H*N; B=1
emb = base.embeddings(torch.tensor([42],device=dev))
layer0 = base.layers[0]; L0=layer0.attn; FF=layer0.ffn

# extract (minimal: only what layer0 needs, but pass all for the signature)
rw=[L0.r_proj.weight.data];kw=[L0.k_proj.weight.data];vw=[L0.v_proj.weight.data];ow=[L0.o_proj.weight.data]
fkw=[FF.key.weight.data];fvw=[FF.value.weight.data]
w0l=[L0.w_lora.lora[0].weight.data];w2l=[L0.w_lora.lora[2].weight.data]
a0l=[L0.a_lora.lora[0].weight.data];a2l=[L0.a_lora.lora[2].weight.data]
g0l=[L0.g_lora.lora[0].weight.data];g2l=[L0.g_lora.lora[2].weight.data]
v0l=[w0l[0]];v2l=[w2l[0]]
w2b=[L0.w_lora.lora[2].bias.data];a2b=[L0.a_lora.lora[2].bias.data];v2b=[torch.zeros(hidden,dtype=torch.float16,device=dev)]
xr=[L0.x_r.data.reshape(-1)];xw=[L0.x_w.data.reshape(-1)];xk=[L0.x_k.data.reshape(-1)]
xv=[L0.x_v.data.reshape(-1)];xa=[L0.x_a.data.reshape(-1)];xg=[L0.x_g.data.reshape(-1)]
kk_l=[L0.k_k.data.reshape(-1)];ka_l=[L0.k_a.data.reshape(-1)];rk_l=[L0.r_k.data.reshape(-1)]
gnw=[L0.g_norm.weight.data];gnb=[L0.g_norm.bias.data];fxk=[FF.x_k.data.reshape(-1)]
anw=[layer0.attn_norm.weight.data];anb=[layer0.attn_norm.bias.data]
fnw=[layer0.ffn_norm.weight.data];fnb=[layer0.ffn_norm.bias.data]
pnw=[layer0.pre_norm.weight.data];pnb=[layer0.pre_norm.bias.data]
sa=torch.zeros(2,B,H,N,N,dtype=torch.float32,device=dev)
xp=torch.zeros(2,B,hidden,dtype=torch.float16,device=dev)
xf=torch.zeros(2,B,hidden,dtype=torch.float16,device=dev)
vf=torch.zeros(B,hidden,dtype=torch.float16,device=dev)

# --- Python layer0 pieces (mirror = faithful attn_step_batched) ---
with torch.no_grad():
    state, xpa, xpf, v_first = _init_state_batched(model, 1, dev, torch.float16)
    residual = layer0.pre_norm(emb)
    h = layer0.attn_norm(residual)
    layer=L0; x_prev=xpa[0]; st=state[0]
    xx = x_prev - h
    xr_i = h + xx*layer.x_r.reshape(1,hidden)
    xw_i = h + xx*layer.x_w.reshape(1,hidden)
    xk_i = h + xx*layer.x_k.reshape(1,hidden)
    xv_i = h + xx*layer.x_v.reshape(1,hidden)
    xa_i = h + xx*layer.x_a.reshape(1,hidden)
    xg_i = h + xx*layer.x_g.reshape(1,hidden)
    r = layer.r_proj(xr_i)
    w = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw_i)))
    k = layer.k_proj(xk_i); v = layer.v_proj(xv_i)
    a = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xa_i)))
    g = torch.sigmoid(layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xg_i))))
    kk = F.normalize((k*layer.k_k.reshape(1,hidden)).view(B,H,N),dim=-1,p=2).view(B,hidden)
    k = k*(1+(a-1)*layer.k_a.reshape(1,hidden))
    w = torch.exp(-EXP_HALF*torch.sigmoid(w.float()))
    vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
    ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
    stt = st*w.view(B,H,1,N) + st@ab.float() + vk.float()
    out = stt.to(h.dtype) @ r.view(B,H,N,1); out=out.view(B,hidden)
    out = F.group_norm(out,num_groups=H,weight=layer.g_norm.weight,bias=layer.g_norm.bias,eps=N*1e-5)
    sk = (r.view(B,H,N)*k.view(B,H,N)*layer.r_k.reshape(1,H,N)).sum(-1,keepdim=True)
    out = out + (sk*v.view(B,H,N)).view(B,hidden)
    py_attn = layer.o_proj(out*g)
    py_x_after = residual + py_attn
    h2 = layer0.ffn_norm(py_x_after)
    xx_f = xpf[0]-h2
    kf = h2 + xx_f*FF.x_k.reshape(1,-1)
    kf = torch.relu(FF.key(kf))**2
    py_ffn = FF.value(kf)
    py_x_final = py_x_after + py_ffn

W=(rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,
   kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)
with torch.no_grad():
    try:
        pieces = mod3.rwkv7_layer0_pieces(emb,*W,sa,xp,xf,vf,H,N)
        names=["attn_out","x_after","ffn_out","x_final"]
        for i,name in enumerate(names):
            ref=[py_attn,py_x_after,py_ffn,py_x_final][i][0].float().cpu()
            got=pieces[i][0].float().cpu()
            cos=F.cosine_similarity(ref.flatten().unsqueeze(0),got.flatten().unsqueeze(0)).item()
            print("  %-10s cos=%+.5f maxabs=%.5f%s"%(name,cos,(ref-got).abs().max().item(),"  <<<"if cos<0.99 else ""),flush=True)
    except Exception as e:
        print("FAILED: %s: %s"%(type(e).__name__,str(e)[:400]),flush=True)
