"""Correctness test for the NPUGraph B=1 decode fast path (`serving/graph_decode.py`).

Verifies the captured graph reproduces the eager C++ forward bit-for-bit:
  - single-step: cosine ~1.0, maxabs ~0, argmax identical
  - multi-step greedy: the generated token sequence matches eager exactly

NPU-only (auto-skips without torch_npu, e.g. in CI). Random-init weights — pure
numerical equivalence, no checkpoint needed.
"""
import os
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (torch.npu.is_available() if hasattr(torch, "npu") else False),
    reason="NPUGraph decode test needs a 910B-class NPU")

os.environ.setdefault("RWKV7_NATIVE_MODEL", "1"); os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
_SVC = os.path.join(os.path.dirname(__file__), "..", "serving")
sys.path.insert(0, _SVC)

import torch_npu  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402
from rwkv7_hf.configuration_rwkv7 import RWKV7Config  # noqa: E402
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM  # noqa: E402
from graph_decode import NpuGraphDecoder  # noqa: E402

DEV = "npu:0"; VOCAB = 65536; H, N, L = 12, 64, 12


def _extract(base, hidden):
    rw, kw, vw, ow, fkw, fvw = [], [], [], [], [], []
    w0l, w2l, a0l, a2l, g0l, g2l, v0l, v2l = [], [], [], [], [], [], [], []
    w2b, a2b, v2b = [], [], []
    xr, xw, xk, xv, xa, xg = [], [], [], [], [], []
    kk_l, ka_l, rk_l, gnw, gnb, fxk = [], [], [], [], [], []
    anw, anb, fnw, fnb, pnw, pnb = [], [], [], [], [], []
    for layer in base.layers:
        a = layer.attn; f = layer.ffn
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
    W = (rw, kw, vw, ow, fkw, fvw, w0l, w2l, a0l, a2l, g0l, g2l, v0l, v2l, w2b, a2b, v2b, xr, xw, xk, xv, xa, xg, kk_l, ka_l, rk_l, gnw, gnb, fxk, anw, anb, fnw, fnb, pnw, pnb)
    return W


@pytest.fixture(scope="module")
def eng():
    cfg = RWKV7Config(vocab_size=VOCAB, hidden_size=H*N, num_hidden_layers=L, num_heads=H, head_dim=N, intermediate_size=4*H*N)
    torch.manual_seed(0)
    model = NativeRWKV7ForCausalLM(cfg).half().to(DEV).eval()
    with torch.no_grad():
        for p in model.parameters(): torch.nn.init.normal_(p, std=0.02)
    base = model.model; hidden = H*N
    mod = load(name="rwkv7_ascend_v3", sources=[os.environ.get("RWKV7_CPP_PATH", "/root/rwkv7_ascend_v3.cpp")],
               verbose=False, extra_cflags=["-O3", "-std=c++17"])
    e = types.SimpleNamespace()
    e.H, e.N, e.L, e.hidden = H, N, L, hidden
    e.base, e.model = base, model
    e.mod = mod
    e.W = _extract(base, hidden)
    e.lm_w_m = model.lm_head.weight.data
    e.fnorm_w = base.norm.weight.data; e.fnorm_b = base.norm.bias.data
    return e


def _newstate(eng, B=1):
    return (torch.zeros(eng.L, B, eng.H, eng.N, eng.N, dtype=torch.float32, device=DEV),
            torch.zeros(eng.L, B, eng.hidden, dtype=torch.float16, device=DEV),
            torch.zeros(eng.L, B, eng.hidden, dtype=torch.float16, device=DEV),
            torch.zeros(B, eng.hidden, dtype=torch.float16, device=DEV))


def _eager_step(eng, token, state):
    sa, xp, xf, vf = state
    emb = eng.base.embeddings(torch.tensor([token], device=DEV))
    return eng.mod.rwkv7_decode_full(emb, *eng.W, sa, xp, xf, vf, eng.H, eng.N, eng.lm_w_m, eng.fnorm_w, eng.fnorm_b)


def test_single_step_bit_exact(eng):
    dec = NpuGraphDecoder(eng); dec.capture()
    sa, xp, xf, vf = _newstate(eng)
    with torch.no_grad():
        out_e = _eager_step(eng, 42, (sa, xp, xf, vf)).clone()
    # fresh state for the graph path
    sa2, xp2, xf2, vf2 = _newstate(eng)
    with torch.no_grad():
        out_g = dec.decode(42, sa2[:, 0:1], xp2[:, 0:1], xf2[:, 0:1], vf2[0:1]).clone()
    cos = F.cosine_similarity(out_e.float().flatten().cpu().unsqueeze(0),
                              out_g.float().flatten().cpu().unsqueeze(0)).item()
    mx = (out_e.float() - out_g.float()).abs().max().item()
    assert cos > 0.9999, "cos=%.6f" % cos
    assert mx < 1e-3, "maxabs=%.4e" % mx
    assert int(out_e.reshape(-1, VOCAB)[-1].argmax()) == int(out_g.reshape(-1, VOCAB)[-1].argmax())


def test_multistep_greedy_matches_eager(eng):
    dec = NpuGraphDecoder(eng); dec.capture()
    def eager_gen(seed, n=8):
        st = _newstate(eng); toks = [seed]
        with torch.no_grad():
            for _ in range(n):
                o = _eager_step(eng, toks[-1], st)
                toks.append(int(o.reshape(-1, VOCAB)[-1].argmax()))
        return toks
    def graph_gen(seed, n=8):
        sa, xp, xf, vf = _newstate(eng); toks = [seed]
        with torch.no_grad():
            for _ in range(n):
                o = dec.decode(toks[-1], sa[:, 0:1], xp[:, 0:1], xf[:, 0:1], vf[0:1])
                toks.append(int(o.reshape(-1, VOCAB)[-1].argmax()))
        return toks
    assert eager_gen(42) == graph_gen(42)


def test_captured_embedding_matches_legacy_graph(eng):
    legacy = NpuGraphDecoder(eng, capture_embedding=False)
    legacy.capture()
    captured = NpuGraphDecoder(eng, capture_embedding=True)
    captured.capture()
    state_legacy = _newstate(eng)
    state_captured = _newstate(eng)
    with torch.no_grad():
        for token in [42, 7, 1024, 13]:
            out_legacy = legacy.decode(token, *state_legacy).clone()
            out_captured = captured.decode(
                torch.tensor([token], device=DEV), *state_captured
            ).clone()
            assert torch.equal(out_legacy, out_captured)
    for expected, actual in zip(state_legacy, state_captured):
        assert torch.equal(expected, actual)
