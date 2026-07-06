"""Pinpoint the per-layer divergence: mirror the C++ op-sequence in Python and
diff each piece against the real attn_step_batched for layer 0."""
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, torch_npu
import torch.nn.functional as F
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.native import _init_state_batched

H, N, dev = 12, 64, "npu:0"
cfg = RWKV7Config(vocab_size=65536, hidden_size=H*N, num_hidden_layers=2,
                  num_heads=H, head_dim=N, intermediate_size=4*H*N)
torch.manual_seed(0)
model = NativeRWKV7ForCausalLM(cfg).half().to(dev).eval()
with torch.no_grad():
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

base = model.model
emb = base.embeddings(torch.tensor([42], device=dev))  # [1,768]
hidden = H*N; B=1
layer0 = base.layers[0]; a0 = layer0.attn

# real attn input h = attn_norm(pre_norm(emb))
with torch.no_grad():
    state, xpa, xpf, v_first = _init_state_batched(model, 1, dev, torch.float16)
    residual = layer0.pre_norm(emb)
    h = layer0.attn_norm(residual)
    a_real, _, _, _ = a0(h, xpa[0], v_first, state[0])   # real attention output

# --- mirror C++ math in Python, piece by piece ---
x_prev = xpa[0]   # zeros
xp = torch.zeros_like(emb)
with torch.no_grad():
    xx = x_prev - h
    xr = h + xx * a0.x_r.reshape(1, hidden)
    xw = h + xx * a0.x_w.reshape(1, hidden)
    xk = h + xx * a0.x_k.reshape(1, hidden)
    xv = h + xx * a0.x_v.reshape(1, hidden)
    xa = h + xx * a0.x_a.reshape(1, hidden)
    xg = h + xx * a0.x_g.reshape(1, hidden)

    r = a0.r_proj(xr); k = a0.k_proj(xk); v = a0.v_proj(xv)
    w_raw = a0.w_lora.lora[2](torch.tanh(a0.w_lora.lora[0](xw)))
    a_sig = torch.sigmoid(a0.a_lora.lora[2](a0.a_lora.lora[0](xa)))
    g_sig = torch.sigmoid(a0.g_lora.lora[2](torch.sigmoid(a0.g_lora.lora[0](xg))))

    kk = F.normalize((k * a0.k_k.reshape(1, hidden)).view(B, H, N), dim=-1, p=2).view(B, hidden)
    k2 = k * (1 + (a_sig - 1) * a0.k_a.reshape(1, hidden))   # modified k
    v_first_new = v.clone()

    w = torch.exp(-0.606531 * torch.sigmoid(w_raw.float()))   # PYTHON: fp32
    vk = v.view(B,H,N,1) @ k2.view(B,H,1,N)
    ab = (-kk).view(B,H,N,1) @ (kk*a_sig).view(B,H,1,N)
    st = state[0]*w.view(B,H,1,N) + state[0] @ ab.float() + vk.float()
    out = st.to(v.dtype) @ r.view(B,H,N,1)
    out = out.view(B, hidden)
    out = F.group_norm(out, num_groups=H, weight=a0.g_norm.weight, bias=a0.g_norm.bias, eps=N*1e-5)
    sk = (r.view(B,H,N) * k2.view(B,H,N) * a0.r_k.reshape(1,H,N)).sum(-1, keepdim=True)
    out = out + (sk * v.view(B,H,N)).view(B, hidden)
    a_mirror = a0.o_proj(out * g_sig)

def show(name, ref, got):
    ref=ref.float().cpu(); got=got.float().cpu()
    cos = F.cosine_similarity(ref.flatten().unsqueeze(0), got.flatten().unsqueeze(0)).item()
    print("  %-10s cos=%.6f maxabs=%.5f" % (name, cos, (ref-got).abs().max().item()), flush=True)

print("Layer-0 attn output: real (module) vs C++-mirror:", flush=True)
show("attn_out", a_real, a_mirror)

# Also: is the model itself deterministic? run twice
with torch.no_grad():
    r1 = model(torch.tensor([[42]],device=dev)).logits[0,-1].float().cpu()
    r2 = model(torch.tensor([[42]],device=dev)).logits[0,-1].float().cpu()
print("model determinism cos=%.6f" % F.cosine_similarity(r1.unsqueeze(0),r2.unsqueeze(0)).item(), flush=True)
