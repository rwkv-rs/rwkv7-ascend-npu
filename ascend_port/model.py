"""Plain-torch RWKV-7 (Goose) full model — Ascend / torch_npu friendly.

Faithful port of ``reference/hakureirm/oracle_numpy.py`` (the BlinkDL RWKV-v7
numpy reference, which is the M1 correctness oracle). Loads a BlinkDL ``.pth``
by key name, runs on NPU, and reuses :mod:`ascend_port.wkv` for the recurrence.

The goal is greedy-exact parity with the numpy oracle on a real 0.1B checkpoint
(see ``tests/test_model_vs_oracle.py``). Precision notes that bite if missed:
numpy ``var()`` is population variance -> torch must use ``unbiased=False``;
group_norm eps is ``head_size * 1e-5``; the production recurrence uses
``scale = 1.0`` (no r-scaling before the group norm).
"""
from __future__ import annotations

import math

import torch

from .wkv import wkv_recurrent

INV_SQRT_E = 0.6065306597126334  # 1 / sqrt(e)


# --------------------------------------------------------------------------- #
# Norms (match oracle_numpy exactly: population variance, eps as noted).
# --------------------------------------------------------------------------- #
def layer_norm(x, w, b, eps=1e-5):
    m = x.mean()
    v = x.var(unbiased=False)
    return (x - m) / (v + eps).sqrt() * w + b


def group_norm(y_nh, w, b, head_size):
    """y_nh: [n_head, head_size] -> flattened [n_embd], per-head norm."""
    eps = head_size * 1e-5
    m = y_nh.mean(dim=-1, keepdim=True)
    v = y_nh.var(dim=-1, keepdim=True, unbiased=False)
    y = (y_nh - m) / (v + eps).sqrt()
    return y.flatten() * w + b


# --------------------------------------------------------------------------- #
# Per-layer mixing (mirrors oracle_numpy.time_mixing / channel_mixing).
# --------------------------------------------------------------------------- #
def time_mixing(x, v0, last_x, S, p, n_head, head_size):
    """x, last_x: [n_embd]; S: [1, n_head, K, V]; v0: [n_embd] | None.

    Returns (out[n_embd], v0, new_last_x[n_embd], new_S[1,H,K,V]).
    """
    xr = x + p["mr"] * (last_x - x)
    xw = x + p["mw"] * (last_x - x)
    xk = x + p["mk"] * (last_x - x)
    xv = x + p["mv"] * (last_x - x)
    xa = x + p["ma"] * (last_x - x)
    xg = x + p["mg"] * (last_x - x)

    r = p["Wr"] @ xr
    w_log = -torch.sigmoid(torch.tanh(xw @ p["Ww1"]) @ p["Ww2"] + p["w_bias"]) * INV_SQRT_E
    k = p["Wk"] @ xk
    v = p["Wv"] @ xv
    if v0 is None:
        v0 = v
    else:
        v = v + (v0 - v) * torch.sigmoid(xv @ p["Wv1"] @ p["Wv2"] + p["v_bias"])
    a = torch.sigmoid(xa @ p["Wa1"] @ p["Wa2"] + p["a_bias"])
    g = torch.sigmoid(xg @ p["Wg1"]) @ p["Wg2"]
    kk = k * p["k_k"]
    k = k + k * (a - 1.0) * p["k_a"]

    H, D = n_head, head_size
    r = r.view(H, D)
    w_log = w_log.view(H, D)
    k = k.view(H, D)
    v = v.view(H, D)
    a = a.view(H, D)
    kk = kk.view(H, D)
    kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    o, S_new = wkv_recurrent(
        r[None, None], w_log[None, None], k[None, None], v[None, None],
        kk[None, None], a[None, None], scale=1.0, initial_state=S,
        output_final_state=True,
    )
    o = o[0, 0]  # [H, D]

    y = group_norm(o, p["ln_w"], p["ln_b"], head_size)           # [n_embd]
    r_k = p["r_k"]                                               # [H, D]
    gate_corr = ((r * k * r_k).sum(dim=-1, keepdim=True) * v).flatten()
    y = y + gate_corr
    out = p["Wo"] @ (y * g)
    return out, v0, x, S_new


def channel_mixing(x, last_x, mix, Wk, Wv):
    k = Wk @ (x + mix * (last_x - x))
    v = Wv @ torch.clamp(k, min=0) ** 2
    return v, x


# --------------------------------------------------------------------------- #
# Weight loading (BlinkDL .pth by key name) + forward.
# --------------------------------------------------------------------------- #
def _sq(t):
    return t.squeeze().float()


def load_params(path, device):
    """Load a BlinkDL RWKV-7 .pth into a flat dict of fp32 tensors on `device`,
    plus per-layer att/ffn param dicts. Returns (params, n_layer, n_embd)."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in raw if k.startswith("blocks."))
    n_embd = raw["emb.weight"].shape[1]

    def sq(k):
        return _sq(raw[k]).to(device)

    params = {
        "emb.weight": raw["emb.weight"].float().to(device),
        "head.weight": raw["head.weight"].float().to(device),
        "ln_out.weight": sq("ln_out.weight"),
        "ln_out.bias": sq("ln_out.bias"),
    }
    # ln0 only exists on layer 0 (applied once to embeddings).
    if "blocks.0.ln0.weight" in raw:
        params["ln0.weight"] = sq("blocks.0.ln0.weight")
        params["ln0.bias"] = sq("blocks.0.ln0.bias")

    att = []
    ffn = []
    for i in range(n_layer):
        p = raw  # noqa: F841 (readability)
        a = {
            "mr": sq(f"blocks.{i}.att.x_r"), "mw": sq(f"blocks.{i}.att.x_w"),
            "mk": sq(f"blocks.{i}.att.x_k"), "mv": sq(f"blocks.{i}.att.x_v"),
            "ma": sq(f"blocks.{i}.att.x_a"), "mg": sq(f"blocks.{i}.att.x_g"),
            "w_bias": sq(f"blocks.{i}.att.w0"),
            "r_k": sq(f"blocks.{i}.att.r_k"),
            "Ww1": sq(f"blocks.{i}.att.w1"), "Ww2": sq(f"blocks.{i}.att.w2"),
            "Wa1": sq(f"blocks.{i}.att.a1"), "Wa2": sq(f"blocks.{i}.att.a2"),
            "a_bias": sq(f"blocks.{i}.att.a0"),
            "Wg1": sq(f"blocks.{i}.att.g1"), "Wg2": sq(f"blocks.{i}.att.g2"),
            "Wv2": sq(f"blocks.{i}.att.v2"), "Wv1": sq(f"blocks.{i}.att.v1"),
            "v_bias": sq(f"blocks.{i}.att.v0"),
            "k_k": sq(f"blocks.{i}.att.k_k"), "k_a": sq(f"blocks.{i}.att.k_a"),
            "Wr": sq(f"blocks.{i}.att.receptance.weight"),
            "Wk": sq(f"blocks.{i}.att.key.weight"),
            "Wv": sq(f"blocks.{i}.att.value.weight"),
            "Wo": sq(f"blocks.{i}.att.output.weight"),
            "ln_w": sq(f"blocks.{i}.att.ln_x.weight"),
            "ln_b": sq(f"blocks.{i}.att.ln_x.bias"),
        }
        att.append(a)
        ffn.append({
            "mix": sq(f"blocks.{i}.ffn.x_k"),
            "Wk": sq(f"blocks.{i}.ffn.key.weight"),
            "Wv": sq(f"blocks.{i}.ffn.value.weight"),
        })
        params[f"blocks.{i}.ln1.weight"] = sq(f"blocks.{i}.ln1.weight")
        params[f"blocks.{i}.ln1.bias"] = sq(f"blocks.{i}.ln1.bias")
        params[f"blocks.{i}.ln2.weight"] = sq(f"blocks.{i}.ln2.weight")
        params[f"blocks.{i}.ln2.bias"] = sq(f"blocks.{i}.ln2.bias")
    params["att"] = att
    params["ffn"] = ffn
    return params, n_layer, n_embd


def new_state(n_layer, n_embd, n_head, head_size, device):
    return {
        "conv": [torch.zeros(n_embd, device=device) for _ in range(n_layer)],
        "conv_ffn": [torch.zeros(n_embd, device=device) for _ in range(n_layer)],
        "S": [torch.zeros(1, n_head, head_size, head_size, device=device)
              for _ in range(n_layer)],
    }


def forward_token(params, token, state, n_layer, n_head, head_size):
    """One RWKV-7 forward for a single token. Returns (logits[n_vocab], state)."""
    x = params["emb.weight"][token]
    x = layer_norm(x, params["ln0.weight"], params["ln0.bias"])
    v0 = None
    for i in range(n_layer):
        x_ = layer_norm(x, params[f"blocks.{i}.ln1.weight"], params[f"blocks.{i}.ln1.bias"])
        dx, v0, state["conv"][i], state["S"][i] = time_mixing(
            x_, v0, state["conv"][i], state["S"][i], params["att"][i], n_head, head_size
        )
        x = x + dx
        x_ = layer_norm(x, params[f"blocks.{i}.ln2.weight"], params[f"blocks.{i}.ln2.bias"])
        dx, state["conv_ffn"][i] = channel_mixing(
            x_, state["conv_ffn"][i], params["ffn"][i]["mix"],
            params["ffn"][i]["Wk"], params["ffn"][i]["Wv"],
        )
        x = x + dx
    x = layer_norm(x, params["ln_out.weight"], params["ln_out.bias"])
    logits = params["head.weight"] @ x
    return logits, state


@torch.no_grad()
def greedy_generate(params, prompt_tokens, n_new, n_layer, n_head, head_size, device):
    """Feed prompt, then greedy-decode n_new tokens. Returns list[int]."""
    state = new_state(n_layer, params["emb.weight"].shape[1], n_head, head_size, device)
    logits = None
    for tok in prompt_tokens:
        logits, state = forward_token(params, int(tok), state, n_layer, n_head, head_size)
    gen = []
    for _ in range(n_new):
        tok = int(torch.argmax(logits).item())
        gen.append(tok)
        logits, state = forward_token(params, tok, state, n_layer, n_head, head_size)
    return gen
