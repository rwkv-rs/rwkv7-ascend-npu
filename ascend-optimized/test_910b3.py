"""910B3 verification: C++ forward correctness (cos) + batch decode aggregate tok/s."""
import os, time
os.environ["RWKV7_NATIVE_MODEL"]="1"; os.environ["TORCHDYNAMO_DISABLE"]="1"
import torch, torch_npu
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PKG="/root/rwkv7-ascend"
H,N,dev=12,64,"npu:0"
cfg=RWKV7Config(vocab_size=65536,hidden_size=H*N,num_hidden_layers=12,num_heads=H,head_dim=N,intermediate_size=4*H*N)
torch.manual_seed(0); model=NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters(): torch.nn.init.normal_(p,std=0.02)
mod=load(name="rwkv7_ascend_v3",sources=[f"{PKG}/rwkv7_ascend_v3.cpp"],verbose=False,extra_cflags=["-O3","-std=c++17"])
base=model.model; L=12; hidden=H*N
# extract weights (same as test_v3)
rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
w2b,a2b,v2b=[],[],[]
xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
anw,anb,fnw,fnb,pnw,pnb=[],[],[],[],[],[]
for li,layer in enumerate(base.layers):
    a=layer.attn;f=layer.ffn
    rw.append(a.r_proj.weight.data);kw.append(a.k_proj.weight.data);vw.append(a.v_proj.weight.data);ow.append(a.o_proj.weight.data)
    fkw.append(f.key.weight.data);fvw.append(f.value.weight.data)
    w0l.append(a.w_lora.lora[0].weight.data);w2l.append(a.w_lora.lora[2].weight.data)
    a0l.append(a.a_lora.lora[0].weight.data);a2l.append(a.a_lora.lora[2].weight.data)
    g0l.append(a.g_lora.lora[0].weight.data);g2l.append(a.g_lora.lora[2].weight.data)
    if hasattr(a,'v_lora') and a.v_lora is not None:
        v0l.append(a.v_lora.lora[0].weight.data);v2l.append(a.v_lora.lora[2].weight.data);v2b.append(a.v_lora.lora[2].bias.data)
    else:
        v0l.append(w0l[-1]);v2l.append(w2l[-1]);v2b.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
    w2b.append(a.w_lora.lora[2].bias.data);a2b.append(a.a_lora.lora[2].bias.data)
    xr.append(a.x_r.data.reshape(-1));xw.append(a.x_w.data.reshape(-1));xk.append(a.x_k.data.reshape(-1));xv.append(a.x_v.data.reshape(-1));xa.append(a.x_a.data.reshape(-1));xg.append(a.x_g.data.reshape(-1))
    kk_l.append(a.k_k.data.reshape(-1));ka_l.append(a.k_a.data.reshape(-1));rk_l.append(a.r_k.data.reshape(-1))
    gnw.append(a.g_norm.weight.data);gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden,dtype=torch.float16,device=dev));fxk.append(f.x_k.data.reshape(-1))
    anw.append(layer.attn_norm.weight.data);anb.append(layer.attn_norm.bias.data);fnw.append(layer.ffn_norm.weight.data);fnb.append(layer.ffn_norm.bias.data)
    if hasattr(layer,'pre_norm'): pnw.append(layer.pre_norm.weight.data);pnb.append(layer.pre_norm.bias.data)
    else: pnw.append(torch.ones(hidden,dtype=torch.float16,device=dev));pnb.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
fnw_=base.norm.weight.data;fnb_=base.norm.bias.data;lmw=model.lm_head.weight.data
W=(rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)

def call(B):
    sa=torch.zeros(L,B,H,N,N,dtype=torch.float32,device=dev)
    xp=torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    xf=torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    vf=torch.zeros(B,hidden,dtype=torch.float16,device=dev)
    emb=base.embeddings(torch.full((B,),42,device=dev))
    return mod.rwkv7_decode_full(emb,*W,sa,xp,xf,vf,H,N,lmw,fnw_,fnb_), sa,xp,xf,vf

# correctness B=1
ids=torch.tensor([[42]],device=dev)
with torch.no_grad(): py=model(ids).logits[0,-1].float().cpu()
out,sa,xp,xf,vf=call(1)
c=out[0].float().cpu()
cos=F.cosine_similarity(py.unsqueeze(0),c.unsqueeze(0)).item()
print("CORRECTNESS B=1 cos=%.5f maxabs=%.4e argmax_match=%d"%(cos,(py-c).abs().max().item(),int(py.argmax()==c.argmax())),flush=True)

# batch decode speed
with torch.no_grad():
    for B in [1,8,16,32]:
        for _ in range(3): call(B)
        torch.npu.synchronize();t0=time.time()
        for _ in range(30): call(B)
        torch.npu.synchronize();ms=(time.time()-t0)/30*1000
        print("B=%2d  %.2f ms/step  agg=%5.0f tok/s"%(B,ms,1000/ms*B),flush=True)
