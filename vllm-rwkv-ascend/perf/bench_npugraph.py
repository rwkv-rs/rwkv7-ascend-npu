"""NPUGraph vs eager bench for the RWKV7-Ascend C++ forward (0.1B + 1.5B, multi-batch).

The C++ forward (`rwkv7_decode_full`) is launch-overhead-bound: each step issues ~960
CANN kernels and eager latency is B-INDEPENDENT (~16ms for 0.1B whether B=1 or B=64).
`torch.npu.NPUGraph` collapses the whole step into one device-side replay. This bench
measures the speedup (tok/s + ms/step) across model sizes and batch sizes.

Run on a 910B3 (CANN 8.5.0 + torch_npu):
    source /usr/local/Ascend/cann-8.5.0/set_env.sh
    python perf/bench_npugraph.py
"""
import os
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1"); os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch, torch_npu, time
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

DEV = "npu:0"; VOCAB = 65536
mod = load(name="rwkv7_ascend_v3", sources=[os.environ.get("RWKV7_CPP_PATH", "/root/rwkv7_ascend_v3.cpp")],
           verbose=False, extra_cflags=["-O3", "-std=c++17"])


def build(H, N, L):
    cfg = RWKV7Config(vocab_size=VOCAB, hidden_size=H*N, num_hidden_layers=L,
                      num_heads=H, head_dim=N, intermediate_size=4*H*N)
    torch.manual_seed(0)
    model = NativeRWKV7ForCausalLM(cfg).half().to(DEV).eval()
    with torch.no_grad():
        for p in model.parameters(): torch.nn.init.normal_(p, std=0.02)
    base = model.model; hidden = H*N
    rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
    w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
    w2b,a2b,v2b=[],[],[]
    xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
    kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
    anw,anb,fnw,fnb,pnw,pnb=[],[],[],[],[],[]
    for layer in base.layers:
        a=layer.attn; f=layer.ffn
        rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data); vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
        fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
        w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
        a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
        g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
        if hasattr(a, "v_lora") and a.v_lora is not None:
            v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data); v2b.append(a.v_lora.lora[2].bias.data)
        else:
            v0l.append(w0l[-1]); v2l.append(w2l[-1]); v2b.append(torch.zeros(hidden, dtype=torch.float16, device=DEV))
        w2b.append(a.w_lora.lora[2].bias.data); a2b.append(a.a_lora.lora[2].bias.data)
        xr.append(a.x_r.data.reshape(-1)); xw.append(a.x_w.data.reshape(-1)); xk.append(a.x_k.data.reshape(-1)); xv.append(a.x_v.data.reshape(-1)); xa.append(a.x_a.data.reshape(-1)); xg.append(a.x_g.data.reshape(-1))
        kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1)); rk_l.append(a.r_k.data.reshape(-1))
        gnw.append(a.g_norm.weight.data); gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden, dtype=torch.float16, device=DEV)); fxk.append(f.x_k.data.reshape(-1))
        anw.append(layer.attn_norm.weight.data); anb.append(layer.attn_norm.bias.data)
        fnw.append(layer.ffn_norm.weight.data); fnb.append(layer.ffn_norm.bias.data)
        if hasattr(layer, "pre_norm"): pnw.append(layer.pre_norm.weight.data); pnb.append(layer.pre_norm.bias.data)
        else: pnw.append(torch.ones(hidden, dtype=torch.float16, device=DEV)); pnb.append(torch.zeros(hidden, dtype=torch.float16, device=DEV))
    W=(rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)
    return base, W, hidden, model.lm_head.weight.data, base.norm.weight.data, base.norm.bias.data


def bench_fn(fn, n=50):
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.npu.synchronize(); return (time.time() - t0) / n * 1000


def setup_fwd(B, H, N, L, hidden, base, W, lm_w, fnw, fnb):
    sa = torch.zeros(L, B, H, N, N, dtype=torch.float32, device=DEV)
    xp = torch.zeros(L, B, hidden, dtype=torch.float16, device=DEV)
    xf = torch.zeros(L, B, hidden, dtype=torch.float16, device=DEV)
    vf = torch.zeros(B, hidden, dtype=torch.float16, device=DEV)
    emb = base.embeddings(torch.full((B,), 42, device=DEV)).clone()
    def fwd(): return mod.rwkv7_decode_full(emb, *W, sa, xp, xf, vf, H, N, lm_w, fnw, fnb)
    return fwd


def run(H, N, L, label, batches):
    print("\n=== %s (H%d N%d L%d) ===" % (label, H, N, L), flush=True)
    print("%4s | %9s | %9s | %11s | %11s | %7s" % ("B", "eager ms", "graph ms", "eager tok/s", "graph tok/s", "speedup"), flush=True)
    base, W, hidden, lm_w, fnw, fnb = build(H, N, L)
    for B in batches:
        try:
            fwd = setup_fwd(B, H, N, L, hidden, base, W, lm_w, fnw, fnb)
            with torch.no_grad():
                for _ in range(5): fwd()
            torch.npu.synchronize()
            ems = bench_fn(fwd)
            fwd2 = setup_fwd(B, H, N, L, hidden, base, W, lm_w, fnw, fnb)
            with torch.no_grad():
                for _ in range(3): fwd2()
            s = torch.npu.Stream(); s.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(s):
                for _ in range(3): gout = fwd2()
            torch.npu.current_stream().wait_stream(s)
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g): gout = fwd2()
            torch.npu.synchronize()
            gms = bench_fn(lambda: g.replay())
            print("%4d | %9.3f | %9.3f | %11.0f | %11.0f | %6.2fx" % (B, ems, gms, 1000/ems*B, 1000/gms*B, ems/gms), flush=True)
        except Exception as e:
            print("%4d | FAILED %s: %s" % (B, type(e).__name__, str(e)[:100]), flush=True)
    del base, W; torch.npu.empty_cache()


if __name__ == "__main__":
    print("NPUGraph vs eager (random-init weights, pure speed)", flush=True)
    run(12, 64, 12, "0.1B", [1, 8, 64])
    run(32, 64, 24, "1.5B", [1, 8, 64])
