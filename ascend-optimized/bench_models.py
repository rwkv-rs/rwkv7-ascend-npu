"""Larger-model batch scaling: aggregate tok/s for 0.1B/0.4B/1.5B-scale at various B.
Larger model -> more launches but higher compute density; batch amortizes launches."""
import os, sys
os.environ["RWKV7_NATIVE_MODEL"]="1"; os.environ["TORCHDYNAMO_DISABLE"]="1"
import torch, torch_npu, time
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
dev="npu:0"
mod=load(name="rwkv7_ascend_v3",sources=["/root/rwkv7_ascend_v3.cpp"],verbose=False,extra_cflags=["-O3","-std=c++17"])

def mem_mb():
    try:
        free,total=torch.npu.mem_get_info()
        return (total-free)//(1024*1024)
    except: return -1

def extract(model,H,N,L):
    base=model.model; hidden=H*N
    lists=[[] for _ in range(35)]
    (rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,
     xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)=lists
    z=torch.zeros(hidden,dtype=torch.float16,device=dev)
    o=torch.ones(hidden,dtype=torch.float16,device=dev)
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
            v0l.append(w0l[-1]);v2l.append(w2l[-1]);v2b.append(z.clone())
        w2b.append(a.w_lora.lora[2].bias.data);a2b.append(a.a_lora.lora[2].bias.data)
        xr.append(a.x_r.data.reshape(-1));xw.append(a.x_w.data.reshape(-1));xk.append(a.x_k.data.reshape(-1));xv.append(a.x_v.data.reshape(-1));xa.append(a.x_a.data.reshape(-1));xg.append(a.x_g.data.reshape(-1))
        kk_l.append(a.k_k.data.reshape(-1));ka_l.append(a.k_a.data.reshape(-1));rk_l.append(a.r_k.data.reshape(-1))
        gnw.append(a.g_norm.weight.data);gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else z.clone());fxk.append(f.x_k.data.reshape(-1))
        anw.append(layer.attn_norm.weight.data);anb.append(layer.attn_norm.bias.data);fnw.append(layer.ffn_norm.weight.data);fnb.append(layer.ffn_norm.bias.data)
        if hasattr(layer,'pre_norm'): pnw.append(layer.pre_norm.weight.data);pnb.append(layer.pre_norm.bias.data)
        else: pnw.append(o.clone());pnb.append(z.clone())
    fnorm_w=base.norm.weight.data;fnorm_b=base.norm.bias.data;lm_w=model.lm_head.weight.data
    return lists, fnorm_w, fnorm_b, lm_w, base

configs=[("0.1B",12,64,12),("0.4B",28,64,28),("1.5B",32,64,32)]
print("model | B | ms/step | agg tok/s | mem(MB)", flush=True)
for name,H,N,L in configs:
    try:
        torch.manual_seed(0)
        cfg=RWKV7Config(vocab_size=65536,hidden_size=H*N,num_hidden_layers=L,num_heads=H,head_dim=N,intermediate_size=4*H*N)
        model=NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
        with torch.no_grad():
            for p in model.parameters(): torch.nn.init.normal_(p,std=0.02)
        vecs,fnw,fnb,lmw,base=extract(model,H,N,L)
        emb_fn=base.embeddings
        for B in [1,8,16]:
            try:
                sa=torch.zeros(L,B,H,N,N,dtype=torch.float32,device=dev)
                xp=torch.zeros(L,B,H*N,dtype=torch.float16,device=dev)
                xf=torch.zeros(L,B,H*N,dtype=torch.float16,device=dev)
                vf=torch.zeros(B,H*N,dtype=torch.float16,device=dev)
                emb=emb_fn(torch.full((B,),42,device=dev))
                with torch.no_grad():
                    for _ in range(3): mod.rwkv7_decode_full(emb,*vecs,sa,xp,xf,vf,H,N,lmw,fnw,fnb)
                    torch.npu.synchronize();t0=time.time()
                    for _ in range(20): mod.rwkv7_decode_full(emb,*vecs,sa,xp,xf,vf,H,N,lmw,fnw,fnb)
                    torch.npu.synchronize()
                    ms=(time.time()-t0)/20*1000
                print("%-5s | %2d | %.2f   | %5.0f     | %d"%(name,B,ms,1000/ms*B,mem_mb()), flush=True)
                del sa,xp,xf,vf,emb; torch.npu.empty_cache()
            except Exception as e:
                print("%-5s | B=%d FAILED %s: %s"%(name,B,type(e).__name__,str(e)[:100]), flush=True); break
        del model,vecs; torch.npu.empty_cache()
    except Exception as e:
        print("%-5s setup FAILED %s: %s"%(name,type(e).__name__,str(e)[:150]), flush=True)
