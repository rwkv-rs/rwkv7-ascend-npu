#!/usr/bin/env python3
"""Verify the triton-ascend WKV kernel matches the pure-torch wkv_recurrent on NPU."""
import torch, torch_npu, torch.nn.functional as F
from ascend_port.wkv import wkv_recurrent as wt
from ascend_port.wkv_triton import wkv_recurrent as wtri
dev = "npu" if torch.npu.is_available() else "cpu"
def mk(B, T, H, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    r = rn(B, T, H, K) * 0.5; k = rn(B, T, H, K) * 0.5; v = rn(B, T, H, K) * 0.5
    w = -0.6065306597126334 * torch.sigmoid(rn(B, T, H, K))
    kk = F.normalize(rn(B, T, H, K), dim=-1); a = torch.sigmoid(rn(B, T, H, K))
    return r, w, k, v, kk, a
worst = 0.0
for (B, T, H, K) in [(4, 1, 4, 64), (2, 16, 4, 64)]:
    r, w, k, v, kk, a = mk(B, T, H, K, seed=(B * 7 + T) % 100)
    r = r.to(dev); w = w.to(dev); k = k.to(dev); v = v.to(dev); kk = kk.to(dev); a = a.to(dev)
    S0 = torch.zeros(B, H, K, K, device=dev, dtype=torch.float32)
    o_t, _ = wt(r, w, k, v, kk, a, scale=1.0, initial_state=S0, output_final_state=True)
    o_k, _ = wtri(r, w, k, v, kk, a, scale=1.0, initial_state=S0.clone(), output_final_state=True)
    rel = (o_t.cpu() - o_k.cpu()).abs().max().item() / max(o_t.cpu().std().item(), 1e-9)
    worst = max(worst, rel)
    print(f"B={B} T={T}: rel_err={rel:.3e}")
print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.3e})")
