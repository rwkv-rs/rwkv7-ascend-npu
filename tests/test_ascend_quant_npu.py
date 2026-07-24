"""Opt-in real-NPU correctness checks for the packed quantization ABI.

The full synchronized performance sweep lives in
``benchmarks/profile_ascend_quant_dispatch.py``. These checks deliberately do
not turn raw-operator candidates into production admission.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch
from torch import nn

from rwkv7_ascend_quant import (
    AscendWeightOnlyLinear,
    load_quantized_linear,
    save_quantized_linear,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RWKV7_RUN_ASCEND_QUANT_NPU") != "1",
    reason="set RWKV7_RUN_ASCEND_QUANT_NPU=1 on the pinned Ascend stack",
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("bit,min_cosine", [(8, 0.9998), (4, 0.99)])
def test_checkpoint_roundtrip_matches_cpu_packed_oracle(bit: int, min_cosine: float):
    import torch_npu  # noqa: F401

    torch.manual_seed(11)
    layer = nn.Linear(256, 512, bias=False, dtype=torch.float16)
    packed = AscendWeightOnlyLinear.from_float(
        layer, bit=bit, group_size=128, enforce_verified_shape=False
    )
    x = torch.randn(5, 256, dtype=torch.float16)
    cpu = packed(x)
    with tempfile.TemporaryDirectory() as directory:
        save_quantized_linear(packed, directory)
        restored = load_quantized_linear(
            directory, enforce_verified_shape=False
        ).to("npu:0")
        out = restored(x.to("npu:0")).cpu()
        torch.npu.synchronize()
    assert cosine(layer(x), out) >= min_cosine
    assert torch.allclose(cpu, out, rtol=5e-3, atol=2e-3)


@pytest.mark.parametrize(
    "bit,rows,min_cosine",
    [(4, 1, 0.992), (8, 17, 0.9999)],
)
def test_raw_candidate_binding_is_explicit(bit: int, rows: int, min_cosine: float):
    import torch_npu  # noqa: F401

    torch.manual_seed(12 + bit)
    k, n = 4096, 16384
    weight = torch.randn(n, k, device="npu:0", dtype=torch.float16) / (k**0.5)
    packed = AscendWeightOnlyLinear(
        k, n, bit=bit, group_size=128, enforce_verified_shape=False
    ).load_fp_weight(weight)
    packed.admission_scope = "raw-candidate"
    x = torch.randn(rows, k, device="npu:0", dtype=torch.float16)
    expected = x @ weight.t()
    actual = packed.bind_npu_fastpath(
        rows, dtype=torch.float16, scope="raw-candidate"
    )(x)
    torch.npu.synchronize()
    assert cosine(expected, actual) >= min_cosine
