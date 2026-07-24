"""Pure-torch RWKV-7 (Goose) WKV recurrence — the DPLR delta rule.

Ascend / torch_npu friendly drop-in for Hakureirm's Triton ``wkv_recurrent``
(https://github.com/Hakureirm/rwkv-sglang, ``wkv_recurrent.py``): same
signature and semantics, no Triton / CUDA. Correctness-first: the time loop is
intentionally sequential so the K-axis reduction order is fixed (greedy-exact
vs the numpy oracle). A faster AscendC / torch_npu kernel is a later (P3) step.

Ground-truth math (per head, per step ``t``), state ``S: [K, V]``::

    sa[v]    = sum_k (-kk[k]) * S[k, v]
    S[k, v] := exp(w[k]) * S[k, v] + (kk[k] * a[k]) * sa[v] + k[k] * v[v]
    o[v]     = sum_k S[k, v] * (r[k] * scale)

This is exactly ``bench/oracle_numpy.py``'s ``time_mixing`` recurrence
(``S = S*w.mT - S@kk*(kk*a).mT + v*k.mT;  y = S@r``) with the state laid out
as the transpose ``[K, V]`` and ``w`` taken as the LOG decay (per-step factor
``exp(w)``). See ``tests/test_wkv_correctness.py`` for the cross-check vs an
independent numpy implementation of the oracle form.
"""
from __future__ import annotations

import torch

__all__ = ["wkv_recurrent"]


def _wkv_one_seq(r, w, k, v, a_kernel, b_kernel, scale, S):
    """Walk one sequence in time. All tensors fp32.

    Args:
        r, w, k, a_kernel, b_kernel: ``[T, H, K]``
        v: ``[T, H, V]``
        S: ``[H, K, V]``
    Returns:
        ``(o[T, H, V], S_final[H, K, V])``
    """
    outs = []
    for t in range(r.shape[0]):
        rt, wt, kt = r[t], w[t], k[t]
        vt, at, bt = v[t], a_kernel[t], b_kernel[t]
        # RHS fully evaluated before assign -> all reads use the pre-update S.
        sa = (at.unsqueeze(-1) * S).sum(dim=1)                 # [H, V]
        S = (
            torch.exp(wt).unsqueeze(-1) * S
            + bt.unsqueeze(-1) * sa.unsqueeze(1)
            + kt.unsqueeze(-1) * vt.unsqueeze(1)
        )                                                       # [H, K, V]
        ot = (S * (rt * scale).unsqueeze(-1)).sum(dim=1)       # [H, V]
        outs.append(ot)
    return torch.stack(outs, dim=0), S                          # [T, H, V], [H, K, V]


def wkv_recurrent(
    r,
    w,
    k,
    v,
    kk,
    a_lr,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """RWKV-7 WKV recurrence (pure torch).

    Args (kernel convention — matches Hakureirm ``wkv_recurrent``):
        r, w, k, kk, a_lr: ``[B, T, H, K]``
        v:                  ``[B, T, H, V]``   (``V == K == head_dim``)
        w:    log decay (per-step factor is ``exp(w)``)
        kk:   L2-normalized over K (caller responsibility)
        a_lr: in-context learning rate (sigmoid in ``(0, 1)``)
        scale: r pre-scale; default ``K ** -0.5`` (NOTE: the production RWKV-7
            backend forces ``scale=1.0`` to match the numpy oracle — pass it
            explicitly there).
        initial_state: ``[N, H, K, V]`` fp32 or ``None`` (zeros).
        cu_seqlens: ``None`` -> batched (B sequences, each length T);
            or a length-``N+1`` offset tensor -> packed varlen (``B == 1``).

    Returns:
        ``(o, final_state)``. ``o: [B, T, H, V]``; ``final_state: [N, H, K, V]``
        or ``None`` (when ``output_final_state`` is False).
    """
    if r.dim() != 4:
        raise ValueError(f"r must be [B,T,H,K], got shape {tuple(r.shape)}")
    B, T, H, K = r.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    scale = float(scale)

    # Form the kernel a/b internally (a_kernel = -kk, b_kernel = kk * a_lr).
    a_kernel = -kk
    b_kernel = kk * a_lr

    # fp32 throughout (greedy-exact vs the fp32 numpy oracle). Cast the OUTPUT
    # back to the input dtype so bf16 serving doesn't leak fp32 into downstream
    # ops (aclnn LayerNorm rejects fp32 input + bf16 weight, EZ1001). The state
    # (ht) stays fp32.
    out_dtype = r.dtype
    r = r.float()
    w = w.float()
    k = k.float()
    v = v.float()
    a_kernel = a_kernel.float()
    b_kernel = b_kernel.float()
    dev = r.device

    if cu_seqlens is None:
        # Batched: B sequences, each length T.
        N = B
        if initial_state is not None:
            S = initial_state.float()                          # [B, H, K, V]
        else:
            S = torch.zeros(N, H, K, V, device=dev, dtype=torch.float32)

        # HF's native path advances the whole active batch at once. Preserve
        # the unavoidable time dependency, but vectorize independent requests
        # instead of launching the recurrence once per request. Decode has
        # T == 1, so this removes the Python B-loop from every model layer.
        outs = []
        for t in range(T):
            rt, wt, kt = r[:, t], w[:, t], k[:, t]
            vt, at, bt = v[:, t], a_kernel[:, t], b_kernel[:, t]
            sa = (at.unsqueeze(-1) * S).sum(dim=2)             # [B, H, V]
            S = (
                torch.exp(wt).unsqueeze(-1) * S
                + bt.unsqueeze(-1) * sa.unsqueeze(2)
                + kt.unsqueeze(-1) * vt.unsqueeze(2)
            )
            ot = (S * (rt * scale).unsqueeze(-1)).sum(dim=2)   # [B, H, V]
            outs.append(ot)
        o = torch.stack(outs, dim=1)                            # [B, T, H, V]
        ht = S                                                  # [B, H, K, V]
        return (o.to(out_dtype), ht) if output_final_state else (o.to(out_dtype), None)

    # Packed varlen (B == 1).
    if B != 1:
        raise ValueError(f"cu_seqlens requires packed B==1, got B={B}")
    N = int(cu_seqlens.shape[0]) - 1
    o = torch.empty(1, T, H, V, device=dev, dtype=torch.float32)
    ht = torch.zeros(N, H, K, V, device=dev, dtype=torch.float32)
    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())
        if initial_state is not None:
            S = initial_state[n].float().clone()
        else:
            S = torch.zeros(H, K, V, device=dev, dtype=torch.float32)
        ob, sb = _wkv_one_seq(
            r[0, bos:eos], w[0, bos:eos], k[0, bos:eos], v[0, bos:eos],
            a_kernel[0, bos:eos], b_kernel[0, bos:eos], scale, S,
        )
        o[0, bos:eos] = ob
        ht[n] = sb
    return (o.to(out_dtype), ht) if output_final_state else (o.to(out_dtype), None)
