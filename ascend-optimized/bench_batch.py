"""Batch-decode benchmark: C++ forward aggregate tok/s vs batch size B.
B>1 turns projections into GEMM (amortizes launch overhead) -> aggregate tok/s rises."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu, time
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

H, N, dev = 12, 64, "npu:0"
cfg = RWKV7Config(vocab_size=65536, hidden_size=H*N, num_hidden_layers=12,
                  num_heads=H, head_dim=N, intermediate_size=4*H*N)
torch.manual_seed(0)
model = NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
mod = load(name="rwkv7_ascend_v3", sources=["/root/rwkv7_ascend_v3.cpp"],
           verbose=False, extra_cflags=["-O3","-std=c++17"])
base=model.model; L=12; hidden=H*N
rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
w2b,a2b,v2b=[],[],[]
xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
anw,anb,fnw,fnb,pnw,pnb=[],[],[],[],[],[]
for li,layer in enumerate(base.layers):
    a=layer.attn; f=layer.ffn
    rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data)
    vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
    fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
    w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
    a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
    g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
    if hasattr(a,'v_lora') and a.v_lora is not None:
        v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data); v2b.append(a.v_lora.lora[2].bias.data)
    else:
        v0l.append(w0l[-1]); v2l.append(w2l[-1]); v2b.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
    w2b.append(a.w_lora.lora[2].bias.data); a2b.append(a.a_lora.lora[2].bias.data)
    xr.append(a.x_r.data.reshape(-1)); xw.append(a.x_w.data.reshape(-1))
    xk.append(a.x_k.data.reshape(-1)); xv.append(a.x_v.data.reshape(-1))
    xa.append(a.x_a.data.reshape(-1)); xg.append(a.x_g.data.reshape(-1))
    kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1)); rk_l.append(a.r_k.data.reshape(-1))
    gnw.append(a.g_norm.weight.data); gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden,dtype=torch.float16,device=dev)); fxk.append(f.x_k.data.reshape(-1))
    anw.append(layer.attn_norm.weight.data); anb.append(layer.attn_norm.bias.data)
    fnw.append(layer.ffn_norm.weight.data); fnb.append(layer.ffn_norm.bias.data)
    if hasattr(layer,'pre_norm'): pnw.append(layer.pre_norm.weight.data); pnb.append(layer.pre_norm.bias.data)
    else: pnw.append(torch.ones(hidden,dtype=torch.float16,device=dev)); pnb.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
fnorm_w=base.norm.weight.data; fnorm_b=base.norm.bias.data; lm_w=model.lm_head.weight.data
W=(rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)

def call(B):
    sa=torch.zeros(L,B,H,N,N,dtype=torch.float32,device=dev)
    xp=torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    xf=torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    vf=torch.zeros(B,hidden,dtype=torch.float16,device=dev)
    emb=base.embeddings(torch.full((B,),42,device=dev))  # [B, hidden]
    return mod.rwkv7_decode_full(emb,*W,sa,xp,xf,vf,H,N,lm_w,fnorm_w,fnorm_b)

print("B | ms/step | aggregate tok/s | per-seq tok/s", flush=True)
with torch.no_grad():
    for B in [1,2,4,8,16]:
        try:
            for _ in range(5): call(B)
            torch.npu.synchronize(); t0=time.time()
            for _ in range(30): call(B)
            torch.npu.synchronize()
            ms=(time.time()-t0)/30*1000
            agg=1000/ms*B
            print("%2d | %.2f   | %5.0f            | %5.0f"%(B,ms,agg,1000/ms), flush=True)
        except Exception as e:
            print("%2d | FAILED %s: %s"%(B,type(e).__name__,str(e)[:120]), flush=True)
