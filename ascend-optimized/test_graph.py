"""Test NPUGraph capture of the C++ forward — the CUDA-graph-equivalent for NPU.
If replay is fast, this is the path to Albatross-level perf (no per-op launch overhead)."""
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
    for p in model.parameters(): torch.nn.init.normal_(p, std=0.02)
mod3 = load(name="rwkv7_ascend_v3", sources=["/root/rwkv7_ascend_v3.cpp"],
            verbose=False, extra_cflags=["-O3","-std=c++17"])
base=model.model; L=12; B=1; hidden=H*N
rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
w2b,a2b,v2b=[],[],[]
xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
anw,anb,fnw,fnb,pnw,pnb=[],[],[],[],[],[]
for li,layer in enumerate(base.layers):
    a=layer.attn; f=layer.ffn
    rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data); vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
    fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
    w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
    a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
    g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
    if hasattr(a,'v_lora') and a.v_lora is not None:
        v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data); v2b.append(a.v_lora.lora[2].bias.data)
    else:
        v0l.append(w0l[-1]); v2l.append(w2l[-1]); v2b.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
    w2b.append(a.w_lora.lora[2].bias.data); a2b.append(a.a_lora.lora[2].bias.data)
    xr.append(a.x_r.data.reshape(-1)); xw.append(a.x_w.data.reshape(-1)); xk.append(a.x_k.data.reshape(-1))
    xv.append(a.x_v.data.reshape(-1)); xa.append(a.x_a.data.reshape(-1)); xg.append(a.x_g.data.reshape(-1))
    kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1)); rk_l.append(a.r_k.data.reshape(-1))
    gnw.append(a.g_norm.weight.data); gnb.append(a.g_norm.bias.data); fxk.append(f.x_k.data.reshape(-1))
    anw.append(layer.attn_norm.weight.data); anb.append(layer.attn_norm.bias.data)
    fnw.append(layer.ffn_norm.weight.data); fnb.append(layer.ffn_norm.bias.data)
    if hasattr(layer,'pre_norm'): pnw.append(layer.pre_norm.weight.data); pnb.append(layer.pre_norm.bias.data)
    else: pnw.append(torch.ones(hidden,dtype=torch.float16,device=dev)); pnb.append(torch.zeros(hidden,dtype=torch.float16,device=dev))
fnorm_w=base.norm.weight.data; fnorm_b=base.norm.bias.data; lm_w=model.lm_head.weight.data

# STATIC buffers for graph capture
emb_s = base.embeddings(torch.tensor([42],device=dev)).clone()
sa_s = torch.zeros(L,B,H,N,N,dtype=torch.float32,device=dev)
xp_s = torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
xf_s = torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
vf_s = torch.zeros(B,hidden,dtype=torch.float16,device=dev)

def call():
    return mod3.rwkv7_decode_full(emb_s,rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,
        w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,
        anw,anb,fnw,fnb,pnw,pnb,sa_s,xp_s,xf_s,vf_s,H,N,lm_w,fnorm_w,fnorm_b)

# baseline (no graph)
with torch.no_grad():
    for _ in range(5): call()
torch.npu.synchronize(); t0=time.time()
with torch.no_grad():
    for _ in range(50): call()
torch.npu.synchronize()
base_ms=(time.time()-t0)/50*1000
print("Baseline (no graph): %.2f ms = %.0f tok/s"%(base_ms,1000/base_ms),flush=True)

# NPUGraph capture
try:
    with torch.no_grad():
        for _ in range(5): out=call()  # warmup on side stream too
    g = torch.npu.NPUGraph()
    s = torch.npu.Stream(device=dev)
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(3): out=call()
        g.capture_begin()
        out = call()
        g.capture_end()
    torch.npu.current_stream().wait_stream(s)
    torch.npu.synchronize()
    # replay bench
    t0=time.time()
    for _ in range(100): g.replay()
    torch.npu.synchronize()
    g_ms=(time.time()-t0)/100*1000
    print("NPUGraph replay:     %.2f ms = %.0f tok/s  (%.2fx)"%(g_ms,1000/g_ms,base_ms/g_ms),flush=True)
    # correctness of replay
    out_graph = out
    out_eager = call()
    cos=F.cosine_similarity(out_graph[0].float().cpu().unsqueeze(0), out_eager[0].float().cpu().unsqueeze(0)).item()
    print("replay vs eager cos=%.5f"%cos,flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("NPUGraph FAILED: %s: %s"%(type(e).__name__,str(e)[:400]),flush=True)
