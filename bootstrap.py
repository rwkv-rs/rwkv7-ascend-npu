"""Import this before constructing/running the model on NPU.

Does three things (all additive — no upstream file is modified):
  1. registers every `torch.ops.rwkv7_*` op as a pure-PyTorch fallback (rwkv7_npu_ops)
  2. monkey-patches the device hardcodes in rwkv7_fast_v3a to npu (device_patch)
  3. no-ops load_extensions (the shim already defines the ops; CUDA compile unneeded)
"""
import os
import sys
import torch

# make harness/ importable as `import rwkv7_fast_v3a`
_HERE = os.path.dirname(os.path.abspath(__file__))
_HARNESS = os.path.join(_HERE, "harness")
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

try:
    import torch_npu  # noqa: F401  — registers the npu backend / dispatch key
except ImportError:
    pass

# 1. op shim
import rwkv7_npu_ops
rwkv7_npu_ops.install()

# 2. device patches (must import upstream AFTER shim, then patch)
import rwkv7_fast_v3a as R          # noqa: E402
import device_patch
device_patch.apply("npu:0")

# 3. CUDA extension load -> no-op (ops already provided by shim)
R.load_extensions = lambda *a, **k: None

print("[bootstrap] op-shim installed + device patched for npu:0", flush=True)
