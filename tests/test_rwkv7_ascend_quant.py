import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from rwkv7_ascend_quant import (
    AscendWeightOnlyLinear,
    UnverifiedQuantShapeError,
    is_raw_kernel_candidate,
    load_quantized_linear,
    save_quantized_linear,
    should_quantize,
)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_import_and_policy_are_fail_closed():
    stack = dict(
        device_name="Ascend910B3",
        torch_version="2.9.0+cpu",
        torch_npu_version="2.9.0",
        cann_version="8.5.0",
    )
    # The selected rows passed the synchronized raw-op sweep.
    assert is_raw_kernel_candidate(4096, 16384, 1, 4, group_size=128, **stack)
    assert is_raw_kernel_candidate(16384, 4096, 8, 4, group_size=128, **stack)
    assert is_raw_kernel_candidate(4096, 16384, 17, 8, group_size=0, **stack)
    assert is_raw_kernel_candidate(16384, 4096, 28, 8, group_size=0, **stack)
    assert not is_raw_kernel_candidate(4096, 16384, 9, 4, group_size=128, **stack)
    assert not is_raw_kernel_candidate(4096, 16384, 16, 8, group_size=0, **stack)
    assert not is_raw_kernel_candidate(
        4096, 16384, 17, 8, group_size=0, dtype=torch.bfloat16, **stack
    )
    assert not is_raw_kernel_candidate(
        4096, 16384, 17, 8, group_size=0, **{**stack, "device_name": "Ascend910B2"}
    )
    assert not is_raw_kernel_candidate(2048, 8192, 17, 8, group_size=0, **stack)

    # No backend/model-level result has yet passed every speed and quality
    # gate, so production admission remains empty even for raw candidates.
    for bit, rows, group_size in ((4, (1, 8), 128), (8, (17, 28), 0)):
        for row in rows:
            assert not should_quantize(
                4096, 16384, row, bit, group_size=group_size, **stack
            )


@pytest.mark.parametrize("bit,min_cos,max_ratio", [(8, 0.9998, 0.51), (4, 0.99, 0.28)])
def test_cpu_oracle_accuracy_and_storage(bit, min_cos, max_ratio):
    torch.manual_seed(7)
    layer = nn.Linear(256, 512, bias=False, dtype=torch.float16)
    q = AscendWeightOnlyLinear.from_float(layer, bit=bit, group_size=128, enforce_verified_shape=False)
    x = torch.randn(5, 256, dtype=torch.float16)
    assert cosine(layer(x), q(x)) >= min_cos
    assert q.packed_weight_bytes() / (layer.weight.numel() * layer.weight.element_size()) <= max_ratio


def test_rank_three_input_and_bias():
    torch.manual_seed(8)
    layer = nn.Linear(256, 512, bias=True, dtype=torch.float16)
    q = AscendWeightOnlyLinear.from_float(layer, bit=8, enforce_verified_shape=False)
    x = torch.randn(2, 3, 256, dtype=torch.float16)
    assert q(x).shape == (2, 3, 512)
    assert cosine(layer(x), q(x)) >= 0.9998


@pytest.mark.parametrize("bit", [4, 8])
def test_manifest_state_dict_roundtrip(bit):
    torch.manual_seed(9)
    layer = nn.Linear(256, 512, bias=False, dtype=torch.float16)
    q = AscendWeightOnlyLinear.from_float(layer, bit=bit, group_size=128, enforce_verified_shape=False)
    x = torch.randn(3, 256, dtype=torch.float16)
    expected = q(x)
    with tempfile.TemporaryDirectory() as d:
        save_quantized_linear(q, d)
        restored = load_quantized_linear(d, enforce_verified_shape=False)
        assert torch.equal(q.qweight, restored.qweight)
        assert torch.equal(q.scales, restored.scales)
        assert torch.equal(expected, restored(x))
        assert (Path(d) / "quant_manifest.json").is_file()


def test_rejects_wrong_dtype_and_uninitialized():
    q = AscendWeightOnlyLinear(256, 512, bit=8)
    with pytest.raises(RuntimeError):
        q(torch.randn(1, 256, dtype=torch.float16))
    layer = nn.Linear(256, 512, bias=False, dtype=torch.float16)
    q = AscendWeightOnlyLinear.from_float(layer, bit=8, enforce_verified_shape=False)
    with pytest.raises(TypeError):
        q(torch.randn(1, 256, dtype=torch.float32))
