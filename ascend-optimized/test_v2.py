"""Correctness + perf harness: Python vs v1 vs v2 C++ forward on Ascend."""
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
# small init so fp16 stays finite
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

mod1 = load(name="rwkv7_ascend", sources=["/root/rwkv7_ascend.cpp"],
            verbose=False, extra_cflags=["-O3", "-std=c++17"])
mod2 = load(name="rwkv7_ascend_v2", sources=["/root/rwkv7_ascend_v2.cpp"],
            verbose=False, extra_cflags=["-O3", "-std=c++17"])

base = model.model; L = len(base.layers); B = 1; hidden = H * N

# extract weights
rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
for layer in base.layers:
    a=layer.attn; f=layer.ffn
    rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data)
    vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
    fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
    w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
    a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
    g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
    if hasattr(a,'v_lora') and a.v_lora is not None:
        v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data)
    else:
        v0l.append(w0l[-1]); v2l.append(w2l[-1])
    xr.append(a.x_r.data.reshape(-1)); xw.append(a.x_w.data.reshape(-1))
    xk.append(a.x_k.data.reshape(-1)); xv.append(a.x_v.data.reshape(-1))
    xa.append(a.x_a.data.reshape(-1)); xg.append(a.x_g.data.reshape(-1))
    kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1))
    rk_l.append(a.r_k.data.reshape(-1))
    gnw.append(a.g_norm.weight.data)
    gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden,dtype=torch.float16,device=dev))
    fxk.append(f.x_k.data.reshape(-1))
norm_w = base.norm.weight.data if hasattr(base.norm,'weight') else torch.ones(hidden,dtype=torch.float16,device=dev)
lm_w = model.lm_head.weight.data

def fresh_state():
    sa = torch.zeros(L,B,H,N,N,dtype=torch.float32,device=dev)
    xp = torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    xf = torch.zeros(L,B,hidden,dtype=torch.float16,device=dev)
    vf = torch.zeros(B,hidden,dtype=torch.float16,device=dev)
    return sa,xp,xf,vf

ids = torch.tensor([[42]],device=dev)
emb = base.embeddings(torch.tensor([42],device=dev)).unsqueeze(0)

def cpp_call(mod, sa,xp,xf,vf):
    return mod.rwkv7_decode_full(emb,rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,
        xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,sa,xp,xf,vf,H,N,lm_w,norm_w)

with torch.no_grad():
    py_logits = model(ids).logits[0,-1].float().cpu()
    c1 = cpp_call(mod1, *fresh_state())[0,-1].float().cpu()
    try:
        c2 = cpp_call(mod2, *fresh_state())[0,-1].float().cpu()
    except Exception as e:
        c2 = None
        print("  [v2 correctness FAILED: %s]" % type(e).__name__, flush=True)

def cmp(name, ref, got):
    cos = F.cosine_similarity(ref.unsqueeze(0), got.unsqueeze(0)).item()
    mx = (ref-got).abs().max().item()
    print("  %-6s cos=%.5f maxabs=%.4f argmax_match=%d" % (name, cos, mx, int(ref.argmax()==got.argmax())), flush=True)

print("CORRECTNESS (vs Python):", flush=True)
print("  Python finite=%s" % bool(torch.isfinite(py_logits).all()), flush=True)
cmp("v1", py_logits, c1)
if c2 is not None:
    cmp("v2", py_logits, c2)
    cmp("v1-v2", c1, c2)

def bench(fn, n=30):
    torch.npu.synchronize(); t0=time.time()
    with torch.no_grad():
        for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.time()-t0)/n*1000

with torch.no_grad():
    model(ids); model(ids)
py_ms = bench(lambda: model(ids))
args = fresh_state()
c1_ms = bench(lambda: cpp_call(mod1, *fresh_state()))
print("\nPERF (0.1B-scale, fp16, B=1):", flush=True)
print("  Python : %.2f ms = %5.0f tok/s" % (py_ms, 1000/py_ms), flush=True)
print("  C++ v1 : %.2f ms = %5.0f tok/s (%.2fx)" % (c1_ms, 1000/c1_ms, py_ms/c1_ms), flush=True)
if c2 is not None:
    c2_ms = bench(lambda: cpp_call(mod2, *fresh_state()))
    print("  C++ v2 : %.2f ms = %5.0f tok/s (%.2fx vs py, %.2fx vs v1)" % (c2_ms, 1000/c2_ms, py_ms/c2_ms, c1_ms/c2_ms), flush=True)
