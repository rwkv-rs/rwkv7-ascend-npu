"""Ascend NPU defaults for the RWKV-7 HF adapter.

Import this module (or set the env vars manually) to route the adapter to the
fla-free native backend on Ascend NPU:

    import ascend_defaults  # sets env + patches device defaults

This makes ``AutoModelForCausalLM.from_pretrained(..., device_map='npu:0')``
use NativeRWKV7ForCausalLM automatically, with all Triton/CUDA paths disabled.
"""
import os

# --- force fla-free native backend ---
os.environ["RWKV7_NATIVE_MODEL"] = "1"
# --- disable all fast-token backends that need CUDA/Triton ---
os.environ["RWKV7_FAST_FORWARD"] = "0"
os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "eager"
os.environ["RWKV7_FAST_CACHE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

# --- NPU device helper ---
def get_npu_device() -> str:
    """Return 'npu:0' if torch_npu is available, else 'cpu'."""
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    return "cpu"

def is_npu() -> bool:
    """Check if Ascend NPU is available."""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False
