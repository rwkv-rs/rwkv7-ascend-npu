#!/usr/bin/env python3
# coding=utf-8
"""Huawei Ascend / torch_npu smoke for the RWKV-7 native HF backend.

The smoke is intentionally compatibility-first. It always stays on the
FLA-free native PyTorch path and emits a skip row on non-Ascend hosts unless
``--require-ascend`` is set. Tiny mode requires no checkpoint; passing it proves
basic torch_npu availability plus RWKV-7 native recurrent cache decode on NPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import types
from enum import Enum
from importlib.machinery import ModuleSpec
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "0")
os.environ.setdefault("RWKV7_FAST_CACHE", "0")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "eager")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def install_torchvision_stub_if_broken() -> None:
    """Keep Transformers text-only imports working when torchvision is broken.

    Some Ascend images ship a torchvision package that is import-discoverable but
    lacks compiled custom ops such as ``torchvision::nms``. Transformers then
    imports image helpers while importing ``PreTrainedModel`` and aborts before
    our text-only RWKV smoke can run. The stub is only installed after a real
    torchvision import fails.
    """

    if find_spec("torchvision") is None:
        return
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    class InterpolationMode(Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest_exact"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    class ImageReadMode(Enum):
        UNCHANGED = "UNCHANGED"
        GRAY = "GRAY"
        GRAY_ALPHA = "GRAY_ALPHA"
        RGB = "RGB"
        RGB_ALPHA = "RGB_ALPHA"

    def decode_image(*args, **kwargs):
        raise RuntimeError("torchvision.io.decode_image is unavailable in this text-only smoke")

    tv = types.ModuleType("torchvision")
    tv.__spec__ = ModuleSpec("torchvision", loader=None)
    tv.__path__ = []
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    transforms.InterpolationMode = InterpolationMode
    io = types.ModuleType("torchvision.io")
    io.__spec__ = ModuleSpec("torchvision.io", loader=None)
    io.ImageReadMode = ImageReadMode
    io.decode_image = decode_image
    tv.transforms = transforms
    tv.io = io
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.io"] = io


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    append_result(path, row)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def npu_smi_text() -> str:
    try:
        return subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception:
        return ""


def parse_npu_name(raw: str) -> str:
    match = re.search(r"\|\s*\d+\s+([A-Za-z0-9_.-]+)\s+\|", raw)
    return match.group(1) if match else "unknown"


def import_torch_and_npu():
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    return torch


def npu_available(torch: Any) -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_available"):
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def npu_count(torch: Any) -> int:
    try:
        return int(torch.npu.device_count())
    except Exception:
        return 0


def npu_name(torch: Any, index: int = 0) -> str:
    try:
        return str(torch.npu.get_device_name(index))
    except Exception:
        return "unknown"


def npu_memory_stats(torch: Any) -> dict[str, int]:
    if not npu_available(torch):
        return {}
    stats: dict[str, int] = {}
    for key, fn_name in (
        ("npu_memory_allocated_bytes", "memory_allocated"),
        ("npu_memory_reserved_bytes", "memory_reserved"),
        ("npu_max_memory_allocated_bytes", "max_memory_allocated"),
        ("npu_max_memory_reserved_bytes", "max_memory_reserved"),
    ):
        fn = getattr(torch.npu, fn_name, None)
        if fn is None:
            continue
        try:
            stats[key] = int(fn())
        except Exception:
            pass
    return stats


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested.startswith("npu") and not npu_available(torch):
            raise RuntimeError(f"requested --device {requested} but torch_npu NPU is unavailable")
        return requested
    return "npu:0" if npu_available(torch) else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def infer_model_size_label(model_path: str, explicit: str = "") -> str:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else "unknown"


def run_tiny_native(torch: Any, device: str, dtype: Any, max_new_tokens: int) -> dict[str, Any]:
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(20260704)
    cfg = NativeRWKV7Config(
        vocab_size=37,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        use_cache=True,
    )
    model = NativeRWKV7ForCausalLM(cfg).eval().to(device=device, dtype=dtype)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    generated = input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        assert out.logits.shape == (1, input_ids.shape[1], cfg.vocab_size)
        assert torch.isfinite(out.logits).all()
        past = out.past_key_values
        logits = out.logits
        for _ in range(max_new_tokens):
            next_id = logits[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_id], dim=1)
            out = model(next_id, past_key_values=past, use_cache=True)
            assert out.logits.shape == (1, 1, cfg.vocab_size)
            assert torch.isfinite(out.logits).all()
            past = out.past_key_values
            logits = out.logits
    elapsed = time.perf_counter() - t0
    assert generated.shape == (1, input_ids.shape[1] + max_new_tokens), tuple(generated.shape)
    generate_t0 = time.perf_counter()
    with torch.no_grad():
        generated_via_hf = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    generate_elapsed = time.perf_counter() - generate_t0
    assert generated_via_hf.shape == generated.shape, tuple(generated_via_hf.shape)
    row = {
        "axis": "huawei_ascend_tiny_native",
        "status": "pass",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(max_new_tokens),
        "decode_backend": model.rwkv7_native_model_last_decode_backend(),
        "elapsed_s": round(elapsed, 4),
        "generate_elapsed_s": round(generate_elapsed, 4),
        "generate_api": "pass",
    }
    row.update(npu_memory_stats(torch))
    return row


def run_hf_model(torch: Any, args: argparse.Namespace, device: str, dtype: Any) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    model.to(device)
    batch = tensor_to_device(tok(args.prompt, return_tensors="pt"), device)
    generated = batch["input_ids"]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, use_cache=True, logits_to_keep=1)
        assert torch.isfinite(out.logits).all()
        past = out.past_key_values
        logits = out.logits
        for _ in range(args.max_new_tokens):
            next_id = logits[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_id], dim=1)
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            assert out.logits.shape[0] == 1
            assert torch.isfinite(out.logits).all()
            past = out.past_key_values
            logits = out.logits
    elapsed = time.perf_counter() - t0
    assert out.logits.shape[0] == 1
    assert generated.shape[1] >= batch["input_ids"].shape[1]
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    generate_t0 = time.perf_counter()
    with torch.no_grad():
        generated_via_hf = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=None,
        )
    generate_elapsed = time.perf_counter() - generate_t0
    assert generated_via_hf.shape[0] == batch["input_ids"].shape[0]
    assert generated_via_hf.shape[1] == batch["input_ids"].shape[1] + args.max_new_tokens
    row = {
        "axis": "huawei_ascend_hf_model",
        "status": "pass",
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "generated_tokens": int(generated.shape[1] - batch["input_ids"].shape[1]),
        "elapsed_s": round(elapsed, 4),
        "generate_elapsed_s": round(generate_elapsed, 4),
        "generate_api": "pass",
        "backend_class": model.__class__.__name__,
    }
    row.update(npu_memory_stats(torch))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Converted RWKV-7 HF model dir. Optional for tiny smoke only.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=2)
    ap.add_argument("--prompt", default="User: Hello from Huawei Ascend.\n\nAssistant:")
    ap.add_argument("--results", default="")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--require-ascend", action="store_true")
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()

    install_torchvision_stub_if_broken()

    try:
        torch = import_torch_and_npu()
    except Exception as exc:
        row = {
            "axis": "huawei_ascend_smoke",
            "status": "skip",
            "reason": f"torch/torch_npu import failed: {type(exc).__name__}: {exc}",
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        emit(args.results, row)
        if args.require_ascend:
            raise SystemExit(2)
        return 0

    smi = npu_smi_text()
    if not npu_available(torch):
        row = {
            "axis": "huawei_ascend_smoke",
            "status": "skip",
            "reason": "torch_npu NPU unavailable",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_npu": package_version("torch-npu"),
            "npu_smi_name": parse_npu_name(smi),
        }
        emit(args.results, row)
        if args.require_ascend:
            raise SystemExit(2)
        return 0

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    header = {
        "axis": "huawei_ascend_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_npu": package_version("torch-npu"),
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "npu_count": npu_count(torch),
        "npu_name": npu_name(torch, 0),
        "npu_smi_name": parse_npu_name(smi),
        "device": device,
        "dtype": args.dtype,
        "native_model": os.environ.get("RWKV7_NATIVE_MODEL"),
    }
    header.update(npu_memory_stats(torch))
    emit(args.results, header)

    if not args.skip_tiny:
        emit(args.results, run_tiny_native(torch, device, dtype, args.max_new_tokens))
    if args.model:
        emit(args.results, run_hf_model(torch, args, device, dtype))

    print("HUAWEI ASCEND SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
