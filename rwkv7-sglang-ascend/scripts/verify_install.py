#!/usr/bin/env python3
"""Fail-closed API and version inspection (does not reserve the NPU)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    pins = dict(
        line.split("=", 1)
        for line in (root / "versions.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    sgl_root = Path(os.environ.get("SGLANG_ROOT", "/data/work/sglang-upstream"))
    actual = subprocess.check_output(
        ["git", "-C", str(sgl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == pins["SGLANG_COMMIT"], (actual, pins["SGLANG_COMMIT"])
    cann_root = Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()
    assert cann_root.name == f"cann-{pins['CANN_VERSION']}", (
        cann_root,
        pins["CANN_VERSION"],
    )

    import torch
    import torch_npu  # noqa: F401
    from sglang_rwkv7_ascend import Rwkv7Config, register
    from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config

    register()
    cfg = Rwkv7Config(hidden_size=256, num_heads=4, head_dim=64, num_hidden_layers=2)
    spec, resolved = get_linear_attn_config(cfg)
    assert resolved is cfg
    assert spec.support_mamba_cache and not spec.uses_mamba_radix_cache
    params = cfg.mamba2_cache_params
    assert params.shape.conv == [(256, 1), (256, 1)]
    assert params.shape.temporal == (4, 64, 64)
    assert params.dtype.temporal == torch.float32
    device_name = torch_npu.npu.get_device_name(0)
    assert device_name == pins["NPU_DEVICE_NAME"], (device_name, pins["NPU_DEVICE_NAME"])
    print(json.dumps({
        "sglang_commit": actual,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cann": cann_root.name,
        "npu_available": bool(torch.npu.is_available()),
        "npu_device": device_name,
        "plugin": "ok",
    }, indent=2))


if __name__ == "__main__":
    main()
