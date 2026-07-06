"""Perf module: C++ op-coalesced RWKV-7 forward on Ascend NPU.

Compiles rwkv7_ascend_v3.cpp (the full 12-layer TMix+CMix in ONE C++ call,
eliminating Python per-op dispatch) via torch cpp_extension, extracts weights
from the HF-native model, runs the forward, and benches tok/s.

This is the proven fast path (323 tok/s B=1 on 910B2C, cos=1.0). It complements
the pure-PyTorch shim (correctness) — use this when you need speed.

Usage (on 910B3):
  PYTHONPATH=/root/rwkv7-ascend:. python perf/run_perf.py <hf-dir>
e.g. PYTHONPATH=/root/rwkv7-ascend:. python perf/run_perf.py /root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf
"""
import os
import sys
import time

os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
import torch_npu  # noqa
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config  # noqa: F401 (kept for parity)
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

HF_DIR = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "npu:0"

print(f"[perf] loading {HF_DIR} on {DEV}", flush=True)
model = NativeRWKV7ForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.float16).to(DEV).eval()
base = model.model
L = len(base.layers)
H = base.layers[0].attn.num_heads
N = base.layers[0].attn.head_dim
hidden = H * N

print("[perf] compiling C++ coalesced forward (rwkv7_ascend_v3.cpp)...", flush=True)
mod = load(name="rwkv7_ascend_v3_perf",
           sources=[os.path.join(HERE, "rwkv7_ascend_v3.cpp")],
           verbose=False, extra_cflags=["-O3", "-std=c++17"])

# --- extract weights into the layout rwkv7_ascend_v3.cpp expects ---
rw, kw, vw, ow, fkw, fvw = [], [], [], [], [], []
w0l, w2l, a0l, a2l, g0l, g2l, v0l, v2l = [], [], [], [], [], [], [], []
w2b, a2b, v2b = [], [], []
xr, xw, xk, xv, xa, xg = [], [], [], [], [], []
kk_l, ka_l, rk_l, gnw, gnb, fxk = [], [], [], [], [], []
anw, anb, fnw, fnb, pnw, pnb = [], [], [], [], [], []
for li, layer in enumerate(base.layers):
    a = layer.attn
    f = layer.ffn
    rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data)
    vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
    fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
    w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
    a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
    g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
    if hasattr(a, "v_lora") and a.v_lora is not None:
        v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data)
        v2b.append(a.v_lora.lora[2].bias.data)
    else:
        v0l.append(w0l[-1]); v2l.append(w2l[-1])
        v2b.append(torch.zeros(hidden, dtype=torch.float16, device=DEV))
    w2b.append(a.w_lora.lora[2].bias.data); a2b.append(a.a_lora.lora[2].bias.data)
    for lst, attr in ((xr, "x_r"), (xw, "x_w"), (xk, "x_k"), (xv, "x_v"), (xa, "x_a"), (xg, "x_g")):
        lst.append(getattr(a, attr).data.reshape(-1))
    kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1)); rk_l.append(a.r_k.data.reshape(-1))
    gnw.append(a.g_norm.weight.data)
    gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden, dtype=torch.float16, device=DEV))
    fxk.append(f.x_k.data.reshape(-1))
    anw.append(layer.attn_norm.weight.data); anb.append(layer.attn_norm.bias.data)
    fnw.append(layer.ffn_norm.weight.data); fnb.append(layer.ffn_norm.bias.data)
    if hasattr(layer, "pre_norm"):
        pnw.append(layer.pre_norm.weight.data); pnb.append(layer.pre_norm.bias.data)
    else:
        pnw.append(torch.ones(hidden, dtype=torch.float16, device=DEV))
        pnb.append(torch.zeros(hidden, dtype=torch.float16, device=DEV))
fnw_ = base.norm.weight.data
fnb_ = base.norm.bias.data
lmw = model.lm_head.weight.data
W = (rw, kw, vw, ow, fkw, fvw, w0l, w2l, a0l, a2l, g0l, g2l, v0l, v2l, w2b, a2b, v2b,
     xr, xw, xk, xv, xa, xg, kk_l, ka_l, rk_l, gnw, gnb, fxk,
     anw, anb, fnw, fnb, pnw, pnb)


def call(B):
    sa = torch.zeros(L, B, H, N, N, dtype=torch.float32, device=DEV)
    xp = torch.zeros(L, B, hidden, dtype=torch.float16, device=DEV)
    xf = torch.zeros(L, B, hidden, dtype=torch.float16, device=DEV)
    vf = torch.zeros(B, hidden, dtype=torch.float16, device=DEV)
    emb = base.embeddings(torch.full((B,), 42, device=DEV))
    return mod.rwkv7_decode_full(emb, *W, sa, xp, xf, vf, H, N, lmw, fnw_, fnb_)


# --- correctness vs HF-native python forward (real weights) ---
ids = torch.tensor([[42]], device=DEV)
with torch.no_grad():
    py = model(ids).logits[0, -1].float().cpu()
out = call(1)
c = out[0].float().cpu()
cos = F.cosine_similarity(py.unsqueeze(0), c.unsqueeze(0)).item()
print(f"[perf] correctness C++_vs_python cos={cos:.5f} argmax_match={int(py.argmax()==c.argmax())}", flush=True)

# --- bench: single-seq + batch aggregate tok/s ---
print("[perf] benchmarking...", flush=True)
with torch.no_grad():
    for B in [1, 8, 16, 32]:
        for _ in range(3):
            call(B)
        torch.npu.synchronize()
        t0 = time.time()
        for _ in range(30):
            call(B)
        torch.npu.synchronize()
        ms = (time.time() - t0) / 30 * 1000
        print(f"B={B:2d}  {ms:.2f} ms/step  agg={1000/ms*B:.0f} tok/s", flush=True)
print("PERF_DONE", flush=True)
