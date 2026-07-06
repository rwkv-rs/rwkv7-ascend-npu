"""op-shim tok/s on NPU — same code as the CUDA bench (op_shim_cuda_bench.py) but
on npu:0 via bootstrap's device_patch. Gives the clean same-code NPU-vs-CUDA ratio."""
import os, sys, time
REPO = "/root/rwkv7-ascend"
sys.path.insert(0, REPO)
import bootstrap  # noqa: F401  (install shim + device_patch gpu->npu + load_extensions no-op)
import rwkv7_fast_v3a as R
import torch
import torch_npu  # noqa
R.MODEL_PATH = os.environ.get("RWKV7_PTH", REPO + "/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
R.WKV_MODE = "fp32io16"; R.EMB_DEVICE = "gpu"; R.CMIX_SPARSE = "off"; R.RKV_MODE = "off"; R.LOWRANK_WEIGHT = "orig"
print("[bench] op-shim forward on npu:0", flush=True)
m = R.RWKV7()
DEV = "npu:0"
toks = torch.tensor([list(range(16))], device=DEV)
with torch.no_grad():
    for _ in range(3):
        st = m.zero_state(1); m.forward_all_logits(toks, st)
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(20):
        st = m.zero_state(1); m.forward_all_logits(toks, st)
    torch.npu.synchronize(); dt = (time.time() - t0) / 20
    st = m.zero_state(1); out = m.forward_all_logits(toks, st)
print("OPSHIM_NPU: 16tok in %.2fms = %.0f tok/s  continuation=%d (expect 16)"
      % (dt * 1000, 16 / dt, out[0, -1].argmax().item()), flush=True)
