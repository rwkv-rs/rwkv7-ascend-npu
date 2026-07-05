"""Runtime monkey-patches that make upstream's `rwkv7_fast_v3a.py` device-agnostic.

We edit ZERO upstream files — every device hardcode is overridden at import time.
If upstream renames `first_device` / `zero_state` / an op namespace, only THIS
file needs a one-line update (low frequency, obvious). This is what keeps
`git pull upstream` fast-forward / zero-conflict.
"""
import contextlib
import torch


def apply(device: str = "npu:0"):
    import rwkv7_fast_v3a as R

    dev = torch.device(device)
    R.PP_DEVICES = []                       # single-device, no pipeline parallel

    # --- all device helpers -> target device ---
    R.first_device = lambda: dev
    R.last_device = lambda: dev
    R.layer_device_index = lambda layer: 0
    R.layer_device = lambda layer: dev
    R.key_device = lambda key: dev

    # --- sync: cuda -> npu (no-op fallback if npu sync unavailable) ---
    def _sync_all():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    R.sync_all = _sync_all

    # --- zero_state: hardcoded device="cuda" -> target device ---
    def _zero_state(self, B):
        wkv_dtype = torch.float32 if R.WKV_MODE == "fp32io16" else R.DTYPE
        return [
            torch.zeros((R.L, 2, B, R.C), dtype=R.DTYPE, device=dev),
            torch.zeros((R.L, B, R.H, R.N, R.N), dtype=wkv_dtype, device=dev),
            torch.zeros((B,), dtype=torch.int32, device=dev),
        ]
    R.RWKV7.zero_state = _zero_state

    # gpu-side embedding avoids the cpu-emb path that calls is_current_stream_capturing
    R.EMB_DEVICE = "gpu"

    # --- torch.cuda.* context managers / queries used inside upstream code ---
    # `torch.cuda.device(dev)` rejects non-cuda devices; make it a no-op ctx there.
    _orig_cuda_device = torch.cuda.device

    class _cuda_device_ctx:
        def __init__(self, dev):
            self._dev = dev
            self._inner = None

        def __enter__(self):
            if isinstance(self._dev, torch.device) and self._dev.type != "cuda":
                return self
            try:
                self._inner = _orig_cuda_device(self._dev)
                return self._inner.__enter__()
            except (ValueError, RuntimeError):
                return self

        def __exit__(self, *exc):
            if self._inner is not None:
                return self._inner.__exit__(*exc)
            return False

    torch.cuda.device = _cuda_device_ctx
    # never claim stream capture on NPU (upstream uses this to branch emb copy)
    torch.cuda.is_current_stream_capturing = lambda: False
