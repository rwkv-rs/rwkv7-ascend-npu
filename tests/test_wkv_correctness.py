#!/usr/bin/env python3
# coding=utf-8
"""M1a correctness gate: the pure-torch RWKV-7 WKV recurrence (ascend_port.wkv)
must match an INDEPENDENT numpy implementation of the oracle recurrence on the
same random inputs.

Two genuinely independent formulations are cross-checked:
  * torch (`ascend_port.wkv.wkv_recurrent`): state S [H, K, V], a_kernel = -kk
    (the kernel convention, run on NPU / CPU).
  * numpy (oracle form): state S [H, V, K], sa = S @ kk with +kk
    (``bench/oracle_numpy.py`` ``time_mixing`` line 57, run on CPU).

Agreement of these two (different state layout, different sign convention,
different runtime) is strong evidence the recurrence is correct.

Run on the 910B3 (torch_npu): the torch path uses ``device="npu"``; falls back
to ``cpu`` off-box.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from ascend_port.wkv import wkv_recurrent

DTYPE = torch.float32
TOL = 1e-3  # rel_err = max|Δ| / ref.std(); both fp32 -> expect << 1e-3.


# --------------------------------------------------------------------------- #
# Independent numpy reference (oracle [V, K] form, +kk).
# --------------------------------------------------------------------------- #
def wkv_numpy_oracle(r, w_log, k, v, kk, a_lr, scale, S0_vk):
    """One sequence. r,w_log,k,kk,a_lr: [T,H,K]; v:[T,H,V]; S0_vk:[H,V,K].

    Returns (o[T,H,V], S_final[H,V,K]).
    """
    T = r.shape[0]
    S = S0_vk.astype(np.float32).copy()                  # [H, V, K]
    o = np.zeros((T, r.shape[1], v.shape[-1]), dtype=np.float32)
    for t in range(T):
        decay = np.exp(w_log[t])                         # [H, K]
        sa = np.einsum("hvk,hk->hv", S, kk[t])           # [H, V]  (+kk)
        delta = np.einsum("hv,hk->hvk", sa, kk[t] * a_lr[t])
        kv = np.einsum("hv,hk->hvk", v[t], k[t])
        S = S * decay[:, None, :] - delta + kv           # [H, V, K]
        o[t] = np.einsum("hvk,hk->hv", S, r[t] * scale)
    return o, S


def rel_err(out, ref):
    out = np.asarray(out, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    denom = ref.std() or 1.0
    return float(np.abs(out - ref).max() / denom)


# --------------------------------------------------------------------------- #
# Input synthesis (kernel convention; generate on CPU for portability, then move).
# --------------------------------------------------------------------------- #
def make_inputs(B, T, H, K, V, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE) * (0.5 if shape[-1] else 1.0)

    r = torch.randn(B, T, H, K, generator=g, dtype=DTYPE) * 0.5
    k = torch.randn(B, T, H, K, generator=g, dtype=DTYPE) * 0.5
    v = torch.randn(B, T, H, V, generator=g, dtype=DTYPE) * 0.5
    w = -0.6065306597126334 * torch.sigmoid(torch.randn(B, T, H, K, generator=g, dtype=DTYPE))
    kk = F.normalize(torch.randn(B, T, H, K, generator=g, dtype=DTYPE), dim=-1)
    a_lr = torch.sigmoid(torch.randn(B, T, H, K, generator=g, dtype=DTYPE))
    return r, w, k, v, kk, a_lr


def pick_device():
    if torch.npu.is_available():
        return "npu"
    return "cpu"


def run_case(name, B, T, H, K, V, seed, varlen, device, scale):
    r, w, k, v, kk, a_lr = make_inputs(B, T, H, K, V, seed)

    cu = None
    initial_state = None
    N = B
    if varlen:
        assert B == 1
        # three packed segments of length 30, 47, 51 -> offsets 0,30,77,128
        cu_cpu = torch.tensor([0, 30, 77, 128], dtype=torch.long)
        T = int(cu_cpu[-1].item())
        N = int(cu_cpu.shape[0]) - 1
        # regenerate with the real T so shapes match the offsets
        r, w, k, v, kk, a_lr = make_inputs(B, T, H, K, V, seed)
        cu = cu_cpu.to(device)

    # initial state (small) so the carry-in path is exercised
    gi = torch.Generator(device="cpu").manual_seed(seed + 999)
    S0_torch = torch.randn(N, H, K, V, generator=gi, dtype=DTYPE) * 0.1  # [N,H,K,V]

    # ---- torch (NPU/CPU) ----
    r_d = r.to(device); w_d = w.to(device); k_d = k.to(device)
    v_d = v.to(device); kk_d = kk.to(device); a_lr_d = a_lr.to(device)
    S0_d = S0_torch.to(device)
    o_torch, ht_torch = wkv_recurrent(
        r_d, w_d, k_d, v_d, kk_d, a_lr_d,
        scale=scale, initial_state=S0_d, output_final_state=True, cu_seqlens=cu,
    )
    o_torch = o_torch.detach().cpu().numpy()             # [B,T,H,V] or [1,T,H,V]
    ht_torch = ht_torch.detach().cpu().numpy()           # [N,H,K,V]

    # ---- numpy oracle (CPU), independent form ----
    if varlen:
        # compare per-segment
        cu_np = cu_cpu.numpy()
        worst_o, worst_s = 0.0, 0.0
        for n in range(N):
            bos, eos = int(cu_np[n]), int(cu_np[n + 1])
            o_ref, S_ref = wkv_numpy_oracle(
                r[0, bos:eos].numpy(), w[0, bos:eos].numpy(), k[0, bos:eos].numpy(),
                v[0, bos:eos].numpy(), kk[0, bos:eos].numpy(), a_lr[0, bos:eos].numpy(),
                scale, S0_torch[n].permute(0, 2, 1).numpy(),   # [H,K,V]->[H,V,K]
            )
            worst_o = max(worst_o, rel_err(o_torch[0, bos:eos], o_ref))
            worst_s = max(worst_s, rel_err(ht_torch[n], S_ref.transpose(0, 2, 1)))  # [H,V,K]->[H,K,V]
        errs = dict(o=worst_o, ht=worst_s)
    else:
        # batched: check every sequence
        worst_o, worst_s = 0.0, 0.0
        for b in range(B):
            o_ref, S_ref = wkv_numpy_oracle(
                r[b].numpy(), w[b].numpy(), k[b].numpy(), v[b].numpy(),
                kk[b].numpy(), a_lr[b].numpy(), scale,
                S0_torch[b].permute(0, 2, 1).numpy(),
            )
            worst_o = max(worst_o, rel_err(o_torch[b], o_ref))
            worst_s = max(worst_s, rel_err(ht_torch[b], S_ref.transpose(0, 2, 1)))
        errs = dict(o=worst_o, ht=worst_s)

    tag = "VARLEN" if varlen else f"BATCH{B}"
    print(f"[{tag:7s}] {name:20s} T={T:>3} H={H} K={K} V={V} scale={scale:.4f} | "
          f"o_err={errs['o']:.3e} ht_err={errs['ht']:.3e}")
    return errs


def main():
    device = pick_device()
    if device == "npu":
        import torch_npu  # noqa: F401  (registers the npu backend)
        print(f"torch={torch.__version__}  device=npu:{torch.npu.get_device_name(0)}")
    else:
        print(f"torch={torch.__version__}  device=cpu (NPU not available)")
    print(f"tol={TOL}")
    print("-" * 96)

    cases = [
        # name, B, T, H, K, V, seed, varlen
        dict(name="multistep_small", B=2, T=64, H=4, K=64, V=64, seed=0, varlen=False),
        dict(name="multistep_init",  B=2, T=96, H=4, K=64, V=64, seed=2, varlen=False),
        dict(name="decode_t1",       B=4, T=1,  H=4, K=64, V=64, seed=5, varlen=False),
        dict(name="varlen_packed",   B=1, T=0,  H=4, K=64, V=64, seed=3, varlen=True),
    ]

    worst = {"o": 0.0, "ht": 0.0}
    for c in cases:
        # the production backend uses scale=1.0 (matches oracle); also probe K**-0.5
        for scale in (1.0, float(c["K"] ** -0.5)):
            e = run_case(device=device, scale=scale, **c)
            worst["o"] = max(worst["o"], e["o"])
            worst["ht"] = max(worst["ht"], e["ht"])

    print("-" * 96)
    print(f"WORST: o_err={worst['o']:.3e} ht_err={worst['ht']:.3e}")
    fail = []
    if worst["o"] >= TOL:
        fail.append(f"output err {worst['o']:.3e} >= {TOL}")
    if worst["ht"] >= TOL:
        fail.append(f"final-state err {worst['ht']:.3e} >= {TOL}")
    if fail:
        print("GATE: FAIL")
        for f in fail:
            print("  -", f)
        sys.exit(1)
    print("GATE: PASS  (ascend_port.wkv matches the numpy oracle recurrence)")
    sys.exit(0)


if __name__ == "__main__":
    main()
