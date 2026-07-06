#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from enum import Enum
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from pathlib import Path

import torch


def install_torchvision_stub_if_broken() -> None:
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


install_torchvision_stub_if_broken()

from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_for(name: str, device: str):
    if name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def move_batch(batch, device: str):
    if device == "cpu":
        return batch
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return
    npu = getattr(torch, "npu", None)
    if device.startswith("npu") and npu is not None and hasattr(npu, "synchronize"):
        npu.synchronize()


def append_result(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def printable_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
    dtype = dtype_for(args.dtype, args.device)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.device != "cpu" and not args.device.startswith("cuda"):
        model.to(args.device)
    enc = tok(args.prompt, return_tensors="pt")
    enc = move_batch(enc, args.device)

    with torch.inference_mode():
        t0 = time.time()
        out = model(**enc, use_cache=True)
        sync_device(args.device)
        forward_sec = round(time.time() - t0, 4)
        print("logits_shape", tuple(out.logits.shape))
        top5 = out.logits[0, -1].float().topk(5).indices.tolist()
        print("top5", top5)
        print("forward_sec", forward_sec)
        gen_t0 = time.time()
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        sync_device(args.device)
        generate_sec = round(time.time() - gen_t0, 4)
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    fast_backend = getter() if callable(getter) else None
    if callable(getter):
        print("generate_fast_token_backend", fast_backend)
    print("generated_ids_shape", tuple(gen.shape))
    print("decoded_BEGIN")
    print(printable_text(tok.decode(gen[0].tolist(), skip_special_tokens=True)))
    print("decoded_END")
    append_result(
        args.results,
        {
            "axis": "smoke_hf_generate",
            "status": "pass",
            "model": Path(args.model).name,
            "backend_class": model.__class__.__name__,
            "device": args.device,
            "dtype": str(dtype).replace("torch.", ""),
            "native_model": os.environ.get("RWKV7_NATIVE_MODEL"),
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "generated_tokens": int(gen.shape[1] - enc["input_ids"].shape[1]),
            "logits_shape": list(out.logits.shape),
            "top5": top5,
            "forward_sec": forward_sec,
            "generate_sec": generate_sec,
            "fast_token_backend": fast_backend,
        },
    )


if __name__ == "__main__":
    main()
