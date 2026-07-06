"""Find where _step_token_batched body diverges from my manual mirror (1 layer)."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu
import torch.nn.functional as F
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.native import _init_state_batched, _step_token_batched

H, N, dev = 12, 64, "npu:0"
cfg = RWKV7Config(vocab_size=65536, hidden_size=H*N, num_hidden_layers=1,
                  num_heads=H, head_dim=N, intermediate_size=4*H*N)
torch.manual_seed(0)
model = NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
base=model.model; hidden=H*N; B=1
emb=base.embeddings(torch.tensor([42],device=dev))
layer0=base.layers[0]

with torch.no_grad():
    state,xpa,xpf,v_first=_init_state_batched(model,1,dev,torch.float16)
    # REAL
    py_step, *_ = _step_token_batched(model, emb, state, xpa, xpf, v_first)
    py_step=py_step[0].float().cpu()
    # REPLICA: exact body of _step_token_batched, module calls
    state2,xpa2,xpf2,vf2=_init_state_batched(model,1,dev,torch.float16)
    x = emb
    for i, layer in enumerate(base.layers):
        attn = layer.attn
        residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
        h = layer.attn_norm(residual)
        a, xpa2[i], state2[i], vf2 = attn(h, xpa2[i], vf2, state2[i])
        x = residual + a
        residual = x
        h2 = layer.ffn_norm(x)
        f, xpf2[i] = layer.ffn(h2, xpf2[i])
        x = residual + f
    py_replica = x[0].float().cpu()
    # MANUAL MIRROR (inline)
    L0=layer0.attn; FF=layer0.ffn
    residual_m = layer0.pre_norm(emb)
    h_m = layer0.attn_norm(residual_m)
    # call module for attention to guarantee parity
    a_m, _, _, _ = L0(h_m, xpa[0], v_first, state[0])
    py_xa_m = residual_m + a_m
    h2_m = layer0.ffn_norm(py_xa_m)
    f_m, _ = FF(h2_m, xpf[0])
    py_manual = (py_xa_m + f_m)[0].float().cpu()

def show(name,ref,got):
    cos=F.cosine_similarity(ref.flatten().unsqueeze(0),got.flatten().unsqueeze(0)).item()
    print("  %-26s cos=%+.5f maxabs=%.5f%s"%(name,cos,(ref-got).abs().max().item(),"  <<<"if cos<0.999 else ""),flush=True)

print("1-layer, three Python paths:",flush=True)
show("py_step vs py_replica", py_step, py_replica)
show("py_step vs py_manual", py_step, py_manual)
show("py_replica vs py_manual", py_replica, py_manual)
print("py_step[:4]   =",py_step[:4].tolist(),flush=True)
print("py_replica[:4]=",py_replica[:4].tolist(),flush=True)
print("py_manual[:4] =",py_manual[:4].tolist(),flush=True)
