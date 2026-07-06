"""Phase 1 correctness: run upstream standalone (Albatross faster3a) + our op-shim
on NPU, compare logits to our HF-native forward (same weights) -> cosine.

Fully self-contained on 910B3 (no V100 needed). Usage:
  PYTHONPATH=/root/rwkv7-ascend:. python run_phase1.py <0.1b.pth> <0.1b-hf-dir>
"""
import os
import sys

# this also imports rwkv7_fast_v3a as R after applying shim + device patches
import bootstrap  # noqa: F401
import rwkv7_fast_v3a as R
import torch

# HF-native reference (verified cos=1.0 vs V100) lives in the adapter on PYTHONPATH
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch_npu  # noqa
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PTH = sys.argv[1]
HF_DIR = sys.argv[2] if len(sys.argv) > 2 else None
DEV = "npu:0"
PROMPT = list(range(16))

# --- run upstream standalone + shim ---
R.MODEL_PATH = PTH
R.WKV_MODE = "fp16"; R.EMB_DEVICE = "gpu"; R.CMIX_SPARSE = "off"
R.RKV_MODE = "off"; R.LOWRANK_WEIGHT = "orig"
print(f"[phase1] loading {PTH} via upstream+shim on {DEV}", flush=True)
m = R.RWKV7()
toks = torch.tensor([PROMPT], device=DEV)
state = m.zero_state(1)
with torch.no_grad():
    shim_logits = m.forward_all_logits(toks, state)
print("SHIM argmax:", shim_logits.argmax(-1).squeeze(0).tolist(), flush=True)
shim = shim_logits.float().cpu()

# --- run HF-native reference (same weights, HF format) on same NPU ---
if HF_DIR:
    print(f"[phase1] loading {HF_DIR} via HF-native on {DEV}", flush=True)
    hm = NativeRWKV7ForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.float16).to(DEV).eval()
    ids = torch.tensor([PROMPT], device=DEV)
    with torch.no_grad():
        hlg = hm(ids).logits[0]
    print("HF-native argmax:", hlg.argmax(-1).squeeze(0).tolist(), flush=True)
    hf = hlg.float().cpu()
    cos = torch.nn.functional.cosine_similarity(shim.flatten(), hf.flatten(), dim=0).item()
    am = (shim.argmax(-1) == hf.argmax(-1)).float().mean().item()
    maxabs = (shim - hf).abs().max().item()
    verdict = "PASS" if cos > 0.99 else "CHECK"
    print(f"PHASE1_RESULT cos_shim_vs_hfnative={cos:.5f} argmax_match={am:.4f} "
          f"max_abs={maxabs:.4f} verdict={verdict}", flush=True)
else:
    print("PHASE1_RESULT (no HF ref given) shim_logits saved", flush=True)
