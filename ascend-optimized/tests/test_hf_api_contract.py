#!/usr/bin/env python3
# coding=utf-8
"""HF API contract smoke tests for the RWKV-7 adapter.

This covers integration points commonly touched by PEFT/Trainer/generation
stacks but not exercised by a plain forward pass: fixed-vocab resize handling,
generation input preparation, recurrent cache beam reorder, and gradient
checkpointing toggles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from enum import Enum
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from pathlib import Path

# Keep the V100 training smoke path out of Dynamo/Triton compile trouble.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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


def move_batch(batch, device: str):
    if device == "cpu":
        return batch
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def append_result(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--beam-new-tokens", type=int, default=2)
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.device != "cpu" and not args.device.startswith("cuda"):
        model.to(args.device)
    set_attn_mode(model, args.attn_mode)
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")

    # Fixed official RWKV trie vocab: no-op resize should be accepted, changing
    # the size should fail loudly instead of creating a broken model/tokenizer.
    emb = model.get_input_embeddings()
    same = model.resize_token_embeddings(model.config.vocab_size)
    assert same is emb, "same-size resize should be a no-op returning input embeddings"
    try:
        model.resize_token_embeddings(model.config.vocab_size + 1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("changing RWKV vocab size should raise NotImplementedError")

    prompts = ["User: Alpha.\n\nAssistant:", "User: Beta.\n\nAssistant:"]
    batch = tok(prompts, return_tensors="pt", padding=True)
    batch = move_batch(batch, args.device)

    with torch.no_grad():
        out = model(**batch, use_cache=True, logits_to_keep=1)
    assert out.past_key_values is not None, "use_cache=True should return recurrent state"
    prepared = model.prepare_inputs_for_generation(
        batch["input_ids"],
        past_key_values=out.past_key_values,
        attention_mask=batch.get("attention_mask"),
        use_cache=True,
        logits_to_keep=1,
    )
    assert prepared["input_ids"].shape[1] == 1, prepared["input_ids"].shape
    assert prepared["past_key_values"] is out.past_key_values

    beam_idx = torch.tensor([1, 0], dtype=torch.long, device=batch["input_ids"].device)
    reordered = model._reorder_cache(out.past_key_values, beam_idx)
    assert reordered is out.past_key_values
    assert reordered.get_seq_length() >= batch["input_ids"].shape[1]

    if args.beam_new_tokens > 0:
        with torch.no_grad():
            beam = model.generate(
                **{k: v[:1] for k, v in batch.items()},
                max_new_tokens=args.beam_new_tokens,
                num_beams=2,
                do_sample=False,
                use_cache=True,
            )
        assert beam.shape[0] == 1 and beam.shape[1] >= batch["input_ids"].shape[1]
        backend_getter = getattr(model, "rwkv7_last_fast_token_backend", None)
        effective_backend = None
        if callable(backend_getter):
            effective_backend = backend_getter()
            print("generate_fast_token_backend", effective_backend)
            assert effective_backend in {"native_graph", "native_jit", "fla"}, effective_backend
        print("beam_ids", beam[0, -args.beam_new_tokens :].tolist())
    else:
        effective_backend = None

    model.train()
    model.config.use_cache = True
    model.gradient_checkpointing_enable()
    assert getattr(model, "is_gradient_checkpointing", True), "gradient checkpointing flag was not enabled"
    append_result(
        args.results,
        {
            "axis": "hf_api_contract",
            "status": "pass",
            "model": Path(args.model).name,
            "backend_class": model.__class__.__name__,
            "device": args.device,
            "dtype": args.dtype,
            "native_model": os.environ.get("RWKV7_NATIVE_MODEL"),
            "batch_size": int(batch["input_ids"].shape[0]),
            "prompt_tokens": int(batch["input_ids"].shape[1]),
            "cache_type": type(out.past_key_values).__name__,
            "cache_seq_length": int(out.past_key_values.get_seq_length()),
            "beam_new_tokens": int(args.beam_new_tokens),
            "fast_token_backend": effective_backend,
        },
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
