"""Benchmark the op-shim (pure-PyTorch rwkv7_* ops) on CUDA — same code that runs
on NPU in Phase 1. Measures the device-agnostic forward rate, isolating hardware
(NPU vs CUDA) for the SAME pure-PyTorch implementation.

Run on 5070:  python op_shim_cuda_bench.py
Run on NPU:   set DEV=npu:0 + drop torch.cuda.synchronize -> npu.synchronize
"""
import sys, os, time
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "harness"))
import torch
import rwkv7_npu_ops
rwkv7_npu_ops.install()
import rwkv7_fast_v3a as R
R.load_extensions = lambda *a, **k: None  # bypass CUDA-kernel compile; use the shim's ops

PTH = os.environ.get("RWKV7_PTH", r"D:\rwkv7-models\rwkv7-g1d-0.1b-20260129-ctx8192.pth")
DEV = os.environ.get("RWKV7_DEV", "cuda:0")
R.MODEL_PATH = PTH
R.WKV_MODE = os.environ.get("RWKV7_WKV_MODE", "fp32io16"); R.EMB_DEVICE = "gpu"; R.CMIX_SPARSE = "off"; R.RKV_MODE = "off"; R.LOWRANK_WEIGHT = "orig"
print(f"[bench] op-shim forward on {DEV}, model={PTH}", flush=True)
m = R.RWKV7()
toks = torch.tensor([list(range(16))], device=DEV)

def sync():
    if "npu" in DEV:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

with torch.no_grad():
    for _ in range(3):
        st = m.zero_state(1); m.forward_all_logits(toks, st)
    sync(); t0 = time.time()
    for _ in range(20):
        st = m.zero_state(1); m.forward_all_logits(toks, st)
    sync(); dt = (time.time() - t0) / 20
    st = m.zero_state(1); out = m.forward_all_logits(toks, st)
    print("OPSHIM %s: 16 tok in %.2fms = %.0f tok/s  argmax[:5]=%s"
          % (DEV, dt * 1000, 16 / dt, out.argmax(-1).squeeze(0).tolist()[:5]), flush=True)
