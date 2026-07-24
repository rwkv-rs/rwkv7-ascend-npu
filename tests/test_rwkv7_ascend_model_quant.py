import tempfile
import copy

import pytest
import torch
from torch import nn

from rwkv7_ascend_model_quant import (
    RWKV7FFNQuantSpec,
    apply_rwkv7_sqrelu_equalization,
    compute_rwkv7_sqrelu_equalization_scale,
    discover_rwkv7_ffn_pairs,
    load_quantized_model_state,
    quantize_rwkv7_ffn_model,
    save_quantized_model_checkpoint,
)
from rwkv7_ascend_quant import (
    AscendWeightOnlyLinear,
    UnverifiedQuantShapeError,
    is_raw_kernel_candidate,
)


class TinyFFN(nn.Module):
    def __init__(self, hidden=128, intermediate=256):
        super().__init__()
        self.key = nn.Linear(hidden, intermediate, bias=False, dtype=torch.float16)
        self.value = nn.Linear(intermediate, hidden, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.value(torch.relu(self.key(x)).square())


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = TinyFFN()


class TinyModel(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList(TinyBlock() for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer.ffn(x)
        return x


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_discovers_canonical_pairs():
    pairs = discover_rwkv7_ffn_pairs(TinyModel())
    assert [(pair.layer, pair.module_path) for pair in pairs] == [
        (0, "layers.0.ffn"),
        (1, "layers.1.ffn"),
    ]


def test_raw_policy_binds_w4_group_size_exactly():
    stack = dict(
        device_name="Ascend910B3",
        torch_version="2.9.0+cpu",
        torch_npu_version="2.9.0",
        cann_version="8.5.0",
    )
    assert is_raw_kernel_candidate(4096, 16384, 1, 4, group_size=128, **stack)
    assert not is_raw_kernel_candidate(4096, 16384, 1, 4, group_size=64, **stack)


def test_square_relu_equalization_is_function_preserving():
    torch.manual_seed(100)
    key = torch.randn(256, 128, dtype=torch.float64)
    value = torch.randn(128, 256, dtype=torch.float64)
    x = torch.randn(7, 128, dtype=torch.float64)
    scale = torch.rand(256, dtype=torch.float64) * 3.75 + 0.25
    key_eq, value_eq = apply_rwkv7_sqrelu_equalization(key, value, scale)
    expected = torch.relu(x @ key.t()).square() @ value.t()
    actual = torch.relu(x @ key_eq.t()).square() @ value_eq.t()
    assert torch.allclose(expected, actual, rtol=1e-12, atol=1e-10)


def test_weight_cle_reduces_group_column_range_without_runtime_scale():
    value = torch.randn(128, 256)
    value[:, 0] *= 40
    before = value.abs().amax(0)[:128]
    scale = compute_rwkv7_sqrelu_equalization_scale(
        value,
        group_size=128,
        mode="weight-cle",
        scale_min=1e-3,
        scale_max=1e3,
    )
    after = (value * scale[None, :]).abs().amax(0)[:128]
    assert float(before.max() / before.min()) > 50
    assert float(after.max() / after.min()) < 1.001


def test_awq_refuses_missing_or_wrong_calibration_stats():
    value = torch.randn(128, 256)
    with pytest.raises(ValueError, match="requires real"):
        compute_rwkv7_sqrelu_equalization_scale(value, group_size=128, mode="awq")
    with pytest.raises(ValueError, match="shape"):
        compute_rwkv7_sqrelu_equalization_scale(
            value, group_size=128, mode="awq", activation_max=torch.ones(255)
        )


def test_experiment_conversion_removes_fp_copies_and_supports_layer_selection():
    torch.manual_seed(101)
    model = TinyModel()
    x = torch.randn(3, 128, dtype=torch.float16)
    expected = model(x)
    spec = RWKV7FFNQuantSpec(
        bit=4,
        group_size=128,
        layers=(1,),
        admitted_rows=(1, 3),
        equalization="weight-cle",
    )
    report = quantize_rwkv7_ffn_model(
        model,
        spec,
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    assert len(report.projections) == 2
    assert not report.production_eligible
    assert report.floating_weight_copies_remaining == []
    assert report.packed_storage_ratio <= 0.27
    assert isinstance(model.layers[1].ffn.key, AscendWeightOnlyLinear)
    assert isinstance(model.layers[1].ffn.value, AscendWeightOnlyLinear)
    assert model.layers[1].ffn.key.admission_scope == "experiment"
    assert model.layers[1].ffn.key.enforce_verified_shape is False
    assert not hasattr(model.layers[1].ffn.key, "weight")
    assert isinstance(model.layers[0].ffn.key, nn.Linear)
    assert cosine(expected, model(x)) > 0.98


def test_pair_equalization_preserves_key_int_codes():
    torch.manual_seed(103)
    plain = TinyModel(layers=1)
    equalized = copy.deepcopy(plain)
    base_spec = dict(bit=4, group_size=128, admitted_rows=(1,))
    quantize_rwkv7_ffn_model(
        plain,
        RWKV7FFNQuantSpec(**base_spec),
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    report = quantize_rwkv7_ffn_model(
        equalized,
        RWKV7FFNQuantSpec(**base_spec, equalization="weight-cle"),
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    assert torch.equal(plain.layers[0].ffn.key.qweight, equalized.layers[0].ffn.key.qweight)
    assert report.equalization["layers"][0]["key_int_codes_preserved"] is True


def test_value_only_equalization_absorbs_inverse_scale_into_single_fp_key():
    torch.manual_seed(104)
    model = TinyModel(layers=1)
    original_key = model.layers[0].ffn.key.weight.detach().clone()
    report = quantize_rwkv7_ffn_model(
        model,
        RWKV7FFNQuantSpec(
            bit=4,
            group_size=128,
            projections=("value",),
            equalization="weight-cle",
        ),
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    assert isinstance(model.layers[0].ffn.key, nn.Linear)
    assert isinstance(model.layers[0].ffn.value, AscendWeightOnlyLinear)
    assert not torch.equal(original_key, model.layers[0].ffn.key.weight)
    assert len(report.projections) == 1
    assert report.equalization["layers"][0]["key_int_codes_preserved"] is False


def test_production_gate_rejects_before_model_mutation():
    model = TinyModel(layers=1)
    spec = RWKV7FFNQuantSpec(bit=8, admitted_rows=(1,))
    with pytest.raises(UnverifiedQuantShapeError):
        quantize_rwkv7_ffn_model(model, spec)
    assert isinstance(model.layers[0].ffn.key, nn.Linear)
    assert isinstance(model.layers[0].ffn.value, nn.Linear)


def test_experiment_scope_needs_double_opt_in():
    with pytest.raises(ValueError, match="allow_unverified"):
        quantize_rwkv7_ffn_model(
            TinyModel(layers=1),
            RWKV7FFNQuantSpec(bit=8),
            admission_scope="experiment",
        )


@pytest.mark.parametrize("bit", [4, 8])
def test_quant_only_checkpoint_roundtrip(bit):
    torch.manual_seed(102)
    model = TinyModel(layers=1)
    spec = RWKV7FFNQuantSpec(bit=bit, group_size=128, admitted_rows=(2,))
    report = quantize_rwkv7_ffn_model(
        model,
        spec,
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    x = torch.randn(2, 128, dtype=torch.float16)
    expected = model(x)
    with tempfile.TemporaryDirectory() as directory:
        save_quantized_model_checkpoint(model, report, directory)
        restored = TinyModel(layers=1)
        manifest = load_quantized_model_state(restored, directory)
        assert manifest["state_sha256"]
        assert torch.equal(expected, restored(x))
        selected_keys = [
            key for key in restored.state_dict() if key.startswith("layers.0.ffn")
        ]
        assert not any(key.endswith(".weight") for key in selected_keys)
