#!/usr/bin/env python3
# coding=utf-8
"""CPU/no-CUDA generation smoke for the experimental native RWKV-7 CausalLM.

This uses a tiny random config and does not require converted weights, FLA, CUDA,
or external model files. It guards the CPU fallback GenerationMixin path that
upstream Transformers / AMD / no-GPU contributors depend on.
"""
from __future__ import annotations

import sys
import types
from enum import Enum
from importlib.machinery import ModuleSpec
from importlib.util import find_spec

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

from transformers.cache_utils import DynamicCache

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(2026)
    cfg = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(cfg).eval()


class WrappedHead(torch.nn.Module):
    """Module-only output head, matching native mm8/mm4 Linear replacements."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> int:
    model = build_tiny_model()
    embeddings = model.get_input_embeddings()
    assert model.resize_token_embeddings(model.config.vocab_size) is embeddings
    try:
        model.resize_token_embeddings(model.config.vocab_size + 1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("native model should reject RWKV vocab resize")

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    batch_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        batch_out = model(batch_ids, use_cache=True)
        batch_clone = batch_out.past_key_values.clone()
        assert batch_clone is not batch_out.past_key_values
        assert batch_clone._state is not batch_out.past_key_values._state
        original_state = batch_out.past_key_values._state[0].clone()
        model(
            torch.tensor([[7]], dtype=torch.long),
            past_key_values=batch_clone.select_batch(torch.tensor([0], dtype=torch.long), inplace=False),
            use_cache=True,
        )
        assert torch.equal(batch_out.past_key_values._state[0], original_state)
        flat_out = model(torch.tensor([7], dtype=torch.long), past_key_values=batch_out.past_key_values.select_batch(torch.tensor([0], dtype=torch.long), inplace=False), use_cache=True)
    assert flat_out.logits.shape[:2] == (1, 1)
    old_first = batch_out.past_key_values._v_first.clone()
    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    reordered = model._reorder_cache(batch_out.past_key_values, beam_idx)
    assert reordered is batch_out.past_key_values
    assert torch.equal(reordered._v_first, old_first.index_select(0, beam_idx))
    model.gradient_checkpointing_enable()
    assert getattr(model, "is_gradient_checkpointing", True)
    assert model._supports_default_dynamic_cache() is False

    # Some HF GenerationMixin versions pre-create a DynamicCache for unknown
    # model classes. Native RWKV recurrent state is not a KV cache, so an empty
    # DynamicCache must be treated as no cache and run a full prompt prefill.
    with torch.no_grad():
        empty_dynamic = DynamicCache(config=model.config)
        dyn_out = model(input_ids, past_key_values=empty_dynamic, use_cache=True)
        ref_out = model(input_ids, use_cache=True)
    assert dyn_out.logits.shape == ref_out.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert torch.allclose(dyn_out.logits, ref_out.logits)
    assert dyn_out.past_key_values.get_seq_length() == input_ids.shape[1]

    # Native mm8/mm4 output heads are module replacements without a dense
    # `.weight`; the native fallback must call the head module instead.
    wrapped_model = build_tiny_model()
    wrapped_model.lm_head = WrappedHead(wrapped_model.lm_head)
    with torch.no_grad():
        wrapped = wrapped_model(torch.tensor([[1, 2]], dtype=torch.long), use_cache=True)
        wrapped_decode = wrapped_model(
            torch.tensor([[3]], dtype=torch.long),
            past_key_values=wrapped.past_key_values,
            use_cache=True,
        )
    assert wrapped.logits.shape == (1, 2, wrapped_model.config.vocab_size)
    assert wrapped_decode.logits.shape == (1, 1, wrapped_model.config.vocab_size)

    calls: list[tuple[tuple[int, ...], bool, bool]] = []
    original_forward = model.forward

    def counted_forward(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        calls.append((tuple(input_ids.shape), past_key_values is not None, bool(use_cache)))
        return original_forward(input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

    model.forward = types.MethodType(counted_forward, model)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )

    assert out.shape == (1, 7), tuple(out.shape)
    assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
    assert calls, "generate should call forward"
    assert calls[0] == ((1, input_ids.shape[1]), False, True), calls
    assert all(shape == (1, 1) and has_cache and use_cache for shape, has_cache, use_cache in calls[1:]), calls
    print("NATIVE CPU GENERATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
