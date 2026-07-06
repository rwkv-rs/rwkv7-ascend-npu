#!/usr/bin/env python3
# coding=utf-8
"""Create a tiny FLA-free RWKV-7 HF fixture for hardware API smokes.

The fixture is intentionally random and small. It is not a converted official
checkpoint and must not be used for quality or speed claims. Its purpose is to
exercise the standard HF remote-code path (`AutoTokenizer` and
`AutoModelForCausalLM`) on devices where the real checkpoint is not available.
Run it from this repository with `PYTHONPATH=.`; the generated remote-code
entry point imports the current checked-out native backend.
"""
from __future__ import annotations

import argparse
import json
import sys
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

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def write_byte_vocab(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for byte in range(256):
            token = bytes([byte])
            f.write(f"{byte + 1} {token!r} 1\n")


def patch_metadata(output: Path, vocab_size: int) -> None:
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["architectures"] = ["NativeRWKV7ForCausalLM"]
    cfg["model_type"] = "rwkv7_native"
    cfg["auto_map"] = {
        "AutoConfig": "native_model.NativeRWKV7Config",
        "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tok_cfg = {
        "tokenizer_class": "RWKV7Tokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_rwkv7.RWKV7Tokenizer", None]},
        "model_vocab_size": int(vocab_size),
        "pad_token": "<|padding|>",
        "eos_token": "<|endoftext|>",
        "errors": "replace",
    }
    (output / "tokenizer_config.json").write_text(
        json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "<|padding|>", "eos_token": "<|endoftext|>"}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_remote_code_files(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    (output / "tokenization_rwkv7.py").write_bytes((root / "rwkv7_hf" / "tokenization_rwkv7.py").read_bytes())
    (output / "native_model.py").write_text(
        "\"\"\"HF remote-code shim for the repository native RWKV-7 backend.\"\"\"\n"
        "from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM\n"
        "\n"
        "__all__ = [\"NativeRWKV7Config\", \"NativeRWKV7ForCausalLM\"]\n",
        encoding="utf-8",
    )


def create_fixture(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cfg = NativeRWKV7Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        head_dim=args.head_dim,
        intermediate_size=args.intermediate_size,
        decay_low_rank_dim=args.low_rank_dim,
        gate_low_rank_dim=args.low_rank_dim,
        a_low_rank_dim=args.low_rank_dim,
        v_low_rank_dim=args.low_rank_dim,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
    )
    model = NativeRWKV7ForCausalLM(cfg).eval()
    model.save_pretrained(output, safe_serialization=True)

    write_remote_code_files(output)
    write_byte_vocab(output / "rwkv_vocab_v20230424.txt")
    patch_metadata(output, args.vocab_size)
    print(f"Saved tiny native RWKV-7 HF fixture to: {output}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--vocab-size", type=int, default=320)
    ap.add_argument("--hidden-size", type=int, default=16)
    ap.add_argument("--num-hidden-layers", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=4)
    ap.add_argument("--intermediate-size", type=int, default=32)
    ap.add_argument("--low-rank-dim", type=int, default=4)
    args = ap.parse_args()
    create_fixture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
