"""CI-runnable unit test for the pure-torch RWKV-7 WKV recurrence.

Cross-checks ``ascend_port.wkv.wkv_recurrent`` (the DPLR delta-rule, kernel
convention) against an independent einsum formulation of the same per-step math.
Pure torch on CPU — no torch_npu / Triton / NPU needed, so it runs in GitHub
Actions.

Ground-truth per step (state S: [H, K, V], a_kernel = -kk, b_kernel = kk*a_lr):
    sa[v]    = sum_k (-kk[k]) * S[k, v]
    S[k, v] := exp(w[k]) * S[k, v] + (kk[k]*a_lr[k]) * sa[v] + k[k] * v[v]
    o[v]     = sum_k S[k, v] * (r[k] * scale)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # make ascend_port importable

import torch
from ascend_port.wkv import wkv_recurrent


def _reference(r, w, k, v, kk, a_lr, scale):
    """Independent einsum formulation of the recurrence (different expression of
    the math than the vectorized impl, so it cross-checks axis/sign bugs)."""
    B, T, H, K = r.shape
    V = v.shape[-1]
    out = torch.zeros(B, T, H, V)
    for b in range(B):
        S = torch.zeros(H, K, V)
        for t in range(T):
            sa = torch.einsum("hk,hkv->hv", -kk[b, t], S)              # [H, V]
            S = (
                torch.exp(w[b, t]).unsqueeze(-1) * S                   # exp(w) * S
                + (kk[b, t] * a_lr[b, t]).unsqueeze(-1) * sa.unsqueeze(1)  # (kk*a_lr) * sa
                + k[b, t].unsqueeze(-1) * v[b, t].unsqueeze(1)         # k (outer) v
            )                                                          # [H, K, V]
            out[b, t] = torch.einsum("hkv,hk->hv", S, r[b, t] * scale)  # [H, V]
    return out


def _inputs(B=2, T=3, H=2, K=4, seed=0):
    torch.manual_seed(seed)
    V = K
    r = torch.randn(B, T, H, K)
    w = torch.randn(B, T, H, K) * 0.1            # small log-decay
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    kk = torch.randn(B, T, H, K)
    kk = kk / kk.norm(dim=-1, keepdim=True)       # L2-normalized (caller duty)
    a_lr = torch.sigmoid(torch.randn(B, T, H, K))  # in (0, 1)
    return r, w, k, v, kk, a_lr


def test_wkv_matches_reference():
    r, w, k, v, kk, a_lr = _inputs()
    scale = r.shape[-1] ** -0.5
    o, _ = wkv_recurrent(r, w, k, v, kk, a_lr, scale=scale)
    ref = _reference(r.float(), w.float(), k.float(), v.float(), kk.float(), a_lr.float(), float(scale))
    diff = (o.float() - ref).abs().max().item()
    assert torch.allclose(o.float(), ref, atol=1e-5, rtol=1e-4), f"max abs diff {diff}"


def test_wkv_shape_dtype():
    r, w, k, v, kk, a_lr = _inputs(B=1, T=5, H=3, K=4)
    o, st = wkv_recurrent(r.half(), w.half(), k.half(), v.half(), kk.half(), a_lr.half(),
                          scale=1.0, output_final_state=True)
    B, T, H, K = r.shape
    assert o.shape == (B, T, H, K), f"output shape {tuple(o.shape)} != {(B, T, H, K)}"
    assert st.shape == (B, H, K, K), f"final-state shape {tuple(st.shape)} != {(B, H, K, K)}"
    assert o.dtype == torch.float16, f"output dtype {o.dtype} should follow input (fp16)"


def test_wkv_initial_state_continues():
    """Re-feeding the returned final_state must equal running the two halves back-to-back."""
    r, w, k, v, kk, a_lr = _inputs(B=1, T=6, H=2, K=3, seed=7)
    T = r.shape[1]
    h = T // 2
    # run full sequence, take final state after first half as the bridge
    _, s_mid = wkv_recurrent(r[:, :h], w[:, :h], k[:, :h], v[:, :h], kk[:, :h], a_lr[:, :h],
                             scale=1.0, output_final_state=True)
    # second half seeded with s_mid
    o_split, _ = wkv_recurrent(r[:, h:], w[:, h:], k[:, h:], v[:, h:], kk[:, h:], a_lr[:, h:],
                               scale=1.0, initial_state=s_mid)
    # full run as ground truth
    o_full, _ = wkv_recurrent(r, w, k, v, kk, a_lr, scale=1.0)
    diff = (o_split.float() - o_full[:, h:].float()).abs().max().item()
    assert diff < 1e-4, f"initial_state continuation mismatch, max abs diff {diff}"


def test_batched_fast_path_matches_independent_sequence_walks():
    """Vectorizing B must not change per-request recurrence or carried state."""
    r, w, k, v, kk, a_lr = _inputs(B=4, T=5, H=3, K=4, seed=19)
    initial = torch.randn(4, 3, 4, 4) * 0.1
    batch_out, batch_state = wkv_recurrent(
        r,
        w,
        k,
        v,
        kk,
        a_lr,
        scale=1.0,
        initial_state=initial,
        output_final_state=True,
    )
    independent = [
        wkv_recurrent(
            r[i : i + 1],
            w[i : i + 1],
            k[i : i + 1],
            v[i : i + 1],
            kk[i : i + 1],
            a_lr[i : i + 1],
            scale=1.0,
            initial_state=initial[i : i + 1],
            output_final_state=True,
        )
        for i in range(4)
    ]
    expected_out = torch.cat([item[0] for item in independent], dim=0)
    expected_state = torch.cat([item[1] for item in independent], dim=0)
    torch.testing.assert_close(batch_out, expected_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(batch_state, expected_state, rtol=1e-6, atol=1e-6)
