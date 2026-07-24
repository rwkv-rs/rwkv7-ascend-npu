"""Backend-neutral RWKV-7 FFN quantization plumbing for Ascend.

The module contains no transformers, vLLM or SGLang imports.  Each backend can
call the same discovery, replacement and quant-only checkpoint functions.  A
production conversion is fail-closed through ``should_quantize``; experiments
must opt into the visibly different ``experiment`` scope.

RWKV-7 uses ``square(relu(key(x)))`` before the FFN ``value`` projection.  This
lets us equalize a channel pair without changing the floating-point function::

    key'[i, :] = key[i, :] / sqrt(s[i])
    value'[:, i] = value[:, i] * s[i]

because ``square(relu(key'(x))) == square(relu(key(x))) / s`` for positive s.
The transform is useful for W4 weight-only calibration and adds no runtime op.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from rwkv7_ascend_quant import (
    AscendWeightOnlyLinear,
    UnverifiedQuantShapeError,
    is_raw_kernel_candidate,
    should_quantize,
)


MODEL_QUANT_FORMAT_VERSION = 1
AdmissionScope = Literal["production", "raw-candidate", "experiment"]
EqualizationMode = Literal["none", "weight-cle", "awq"]


@dataclass(frozen=True)
class RWKV7FFNQuantSpec:
    """Selection and calibration policy shared by all three serving backends."""

    bit: int
    group_size: int = 128
    projections: tuple[str, ...] = ("key", "value")
    layers: tuple[int, ...] | None = None
    admitted_rows: tuple[int, ...] = (1,)
    equalization: EqualizationMode = "none"
    equalization_alpha: float = 0.5
    equalization_scale_min: float = 0.25
    equalization_scale_max: float = 4.0
    activation_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.bit not in (4, 8):
            raise ValueError("bit must be 4 or 8")
        if self.bit == 4 and self.group_size not in (32, 64, 128):
            raise ValueError("W4 group_size must be 32, 64 or 128")
        if self.bit == 8 and self.group_size not in (0, 128):
            raise ValueError("W8 group_size is unused and must be 0 or 128")
        if not self.projections or not set(self.projections) <= {"key", "value"}:
            raise ValueError("projections must be a non-empty subset of ('key', 'value')")
        if not self.admitted_rows or any(int(row) <= 0 for row in self.admitted_rows):
            raise ValueError("admitted_rows must contain positive flattened row counts")
        if self.equalization != "none" and "value" not in self.projections:
            raise ValueError("pair equalization requires the value projection to be quantized")
        if not 0.0 <= self.equalization_alpha <= 1.0:
            raise ValueError("equalization_alpha must be in [0, 1]")
        if not 0.0 < self.equalization_scale_min <= self.equalization_scale_max:
            raise ValueError("invalid equalization scale clamp")
        if self.activation_dtype != "float16":
            raise ValueError("the measured Ascend weight-only ABI is FP16 activation only")

    @property
    def effective_group_size(self) -> int:
        return self.group_size if self.bit == 4 else 0


@dataclass(frozen=True)
class RWKV7FFNPair:
    layer: int
    module_path: str
    module: nn.Module = field(compare=False, repr=False)
    key: nn.Linear = field(compare=False, repr=False)
    value: nn.Linear = field(compare=False, repr=False)


@dataclass
class QuantizedProjectionRecord:
    layer: int
    projection: str
    module_path: str
    in_features: int
    out_features: int
    bit: int
    group_size: int
    fp_weight_bytes_removed: int
    packed_bytes: int
    packed_storage_ratio: float


@dataclass
class RWKV7ModelQuantReport:
    format_version: int
    admission_scope: AdmissionScope
    spec: dict[str, Any]
    projections: list[QuantizedProjectionRecord]
    equalization: dict[str, Any]
    fp_weight_bytes_removed: int
    packed_bytes: int
    packed_storage_ratio: float
    floating_weight_copies_remaining: list[str]
    production_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["projections"] = [asdict(item) for item in self.projections]
        return value


def _layer_number(path: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.ffn$", path)
    return int(match.group(1)) if match else None


def discover_rwkv7_ffn_pairs(model: nn.Module) -> list[RWKV7FFNPair]:
    """Find canonical ``layers.N.ffn.key/value`` pairs without backend imports."""
    found: list[RWKV7FFNPair] = []
    for path, module in model.named_modules():
        layer = _layer_number(path)
        if layer is None:
            continue
        key = getattr(module, "key", None)
        value = getattr(module, "value", None)
        if not isinstance(key, nn.Linear) or not isinstance(value, nn.Linear):
            continue
        if key.out_features != value.in_features:
            raise ValueError(
                f"{path} is not an RWKV FFN pair: key.out_features="
                f"{key.out_features} != value.in_features={value.in_features}"
            )
        found.append(RWKV7FFNPair(layer, path, module, key, value))
    found.sort(key=lambda item: item.layer)
    if not found:
        raise ValueError("no layers.N.ffn.key/value nn.Linear pairs found")
    if len({item.layer for item in found}) != len(found):
        raise ValueError("duplicate RWKV FFN layer numbers discovered")
    return found


def _normalise_per_group(values: Tensor, group_size: int, eps: float) -> Tensor:
    if values.ndim != 1 or values.numel() % group_size:
        raise ValueError("channel statistic length must be divisible by group_size")
    groups = values.float().clamp_min(eps).reshape(-1, group_size)
    geometric_mean = groups.log().mean(dim=1, keepdim=True).exp()
    return (groups / geometric_mean).reshape(-1)


@torch.no_grad()
def compute_rwkv7_sqrelu_equalization_scale(
    value_weight: Tensor,
    *,
    group_size: int,
    mode: Literal["weight-cle", "awq"],
    activation_max: Tensor | None = None,
    alpha: float = 0.5,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
    eps: float = 1e-8,
) -> Tensor:
    """Return positive channel scales for the exact square-ReLU transform.

    ``weight-cle`` equalizes column maxima inside each W4 input group. ``awq``
    additionally uses post-square-ReLU activation maxima from a real prompt
    calibration suite.  The activation-aware mode refuses to invent stats.
    """
    if value_weight.ndim != 2 or value_weight.shape[1] % group_size:
        raise ValueError("value weight must be [hidden, intermediate] with divisible groups")
    weight_max = value_weight.detach().float().abs().amax(dim=0).clamp_min(eps)
    # value' = value * scale.  The inverse normalized column maximum makes the
    # transformed weight magnitudes similar within each quantization group.
    weight_equalizer = _normalise_per_group(weight_max, group_size, eps).reciprocal()
    if mode == "weight-cle":
        scale = weight_equalizer
    elif mode == "awq":
        if activation_max is None:
            raise ValueError("awq requires real post-square-ReLU activation_max")
        if tuple(activation_max.shape) != (value_weight.shape[1],):
            raise ValueError(f"activation_max must have shape {(value_weight.shape[1],)}")
        activation = _normalise_per_group(activation_max.detach().float(), group_size, eps)
        # Alpha interpolates between pure weight equalization and protecting
        # channels that carry large calibrated activations.
        scale = weight_equalizer.pow(1.0 - alpha) * activation.pow(alpha)
    else:
        raise ValueError(f"unsupported equalization mode {mode!r}")
    return scale.clamp(min=float(scale_min), max=float(scale_max)).to(value_weight.device)


@torch.no_grad()
def apply_rwkv7_sqrelu_equalization(
    key_weight: Tensor,
    value_weight: Tensor,
    scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a function-preserving paired transform to FP key/value weights."""
    if key_weight.ndim != 2 or value_weight.ndim != 2:
        raise ValueError("key and value weights must be matrices")
    intermediate = key_weight.shape[0]
    if value_weight.shape[1] != intermediate or tuple(scale.shape) != (intermediate,):
        raise ValueError("key/value/scale intermediate dimensions do not agree")
    if not torch.isfinite(scale).all() or not bool((scale > 0).all()):
        raise ValueError("equalization scales must be finite and positive")
    work_dtype = torch.float64 if key_weight.dtype is torch.float64 else torch.float32
    s = scale.to(device=key_weight.device, dtype=work_dtype)
    key = key_weight.to(work_dtype) / s.sqrt()[:, None]
    value = value_weight.to(work_dtype) * s.to(value_weight.device)[None, :]
    return key.to(key_weight.dtype), value.to(value_weight.dtype)


def _tensor_sha256(tensor: Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(memoryview(cpu.numpy()).cast("B")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _admit_projection(
    linear: nn.Linear,
    spec: RWKV7FFNQuantSpec,
    scope: AdmissionScope,
) -> None:
    for rows in sorted(set(int(row) for row in spec.admitted_rows)):
        kwargs = dict(group_size=spec.effective_group_size, dtype=torch.float16)
        if scope == "production":
            accepted = should_quantize(
                linear.in_features, linear.out_features, rows, spec.bit, **kwargs
            )
        elif scope == "raw-candidate":
            accepted = is_raw_kernel_candidate(
                linear.in_features, linear.out_features, rows, spec.bit, **kwargs
            )
        elif scope == "experiment":
            accepted = True
        else:  # pragma: no cover
            raise ValueError(scope)
        if not accepted:
            raise UnverifiedQuantShapeError(
                f"{scope} gate rejected bit={spec.bit}, group={spec.effective_group_size}, "
                f"M={rows}, K={linear.in_features}, N={linear.out_features}"
            )


def _quantize_linear(
    linear: nn.Linear,
    weight: Tensor,
    spec: RWKV7FFNQuantSpec,
    admission_scope: AdmissionScope,
) -> AscendWeightOnlyLinear:
    quant = AscendWeightOnlyLinear(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        bit=spec.bit,
        group_size=spec.group_size,
        enforce_verified_shape=admission_scope != "experiment",
    )
    quant.admission_scope = admission_scope
    return quant.load_fp_weight(weight, None if linear.bias is None else linear.bias.detach())


@torch.no_grad()
def _rescale_quantized_key_output(
    quant: AscendWeightOnlyLinear,
    scale: Tensor,
) -> None:
    """Absorb ``1/sqrt(scale)`` into key dequant scales, preserving int codes."""
    root = scale.detach().to(device=quant.scales.device, dtype=torch.float32).sqrt()
    if quant.bit == 8:
        quant.scales.div_(root.to(torch.float16))
    else:
        quant.scales.div_(root.to(torch.float16)[None, :])
        # Symmetric W4 offsets are zero, but apply the same affine transform so
        # the invariant remains explicit if a future calibrated offset is used.
        quant.offsets.div_(root.to(torch.float16)[None, :])
    if quant.has_bias:
        quant.bias.div_(root.to(torch.float16))


@torch.no_grad()
def quantize_rwkv7_ffn_model(
    model: nn.Module,
    spec: RWKV7FFNQuantSpec,
    *,
    admission_scope: AdmissionScope = "production",
    activation_max_by_layer: Mapping[int, Tensor] | None = None,
    allow_unverified_experiment: bool = False,
) -> RWKV7ModelQuantReport:
    """Replace selected FFN projections and remove their floating weights.

    The function preflights every layer/row before mutating the model.  Passing
    ``experiment`` requires a second explicit boolean and is recorded as
    non-production in the returned manifest.
    """
    if admission_scope == "experiment" and not allow_unverified_experiment:
        raise ValueError("experiment scope requires allow_unverified_experiment=True")
    pairs = discover_rwkv7_ffn_pairs(model)
    selected_layers = set(spec.layers) if spec.layers is not None else {pair.layer for pair in pairs}
    selected = [pair for pair in pairs if pair.layer in selected_layers]
    missing = selected_layers - {pair.layer for pair in selected}
    if missing:
        raise ValueError(f"requested RWKV layers were not found: {sorted(missing)}")
    if not selected:
        raise ValueError("layer selection is empty")
    # Runtime kernels are FP16-only.  Refuse a mixed BF16/FP16 production model;
    # adapters must convert the model once, outside the timed serving loop.
    if admission_scope != "experiment":
        wrong_dtype = [
            f"{pair.module_path}.{projection}"
            for pair in selected
            for projection in spec.projections
            if getattr(pair, projection).weight.dtype is not torch.float16
        ]
        if wrong_dtype:
            raise TypeError(f"production Ascend quantization requires FP16 source modules: {wrong_dtype}")
    for pair in selected:
        for projection in spec.projections:
            _admit_projection(getattr(pair, projection), spec, admission_scope)

    records: list[QuantizedProjectionRecord] = []
    equalization_rows: list[dict[str, Any]] = []
    for pair in selected:
        key_source = pair.key
        value_source = pair.value
        key_weight = key_source.weight.detach()
        value_weight = value_source.weight.detach()
        scale: Tensor | None = None
        transformed_value_weight = value_weight
        transformed_fp_key_weight: Tensor | None = None
        transformed_fp_key_bias: Tensor | None = None
        if spec.equalization != "none":
            activation_max = None
            if activation_max_by_layer is not None:
                activation_max = activation_max_by_layer.get(pair.layer)
            scale = compute_rwkv7_sqrelu_equalization_scale(
                value_weight,
                group_size=spec.group_size,
                mode=spec.equalization,
                activation_max=activation_max,
                alpha=spec.equalization_alpha,
                scale_min=spec.equalization_scale_min,
                scale_max=spec.equalization_scale_max,
            )
            transformed_value_weight = (
                value_weight.float() * scale.to(value_weight.device)[None, :]
            ).to(value_weight.dtype)
            if "key" not in spec.projections:
                # Value-only is attractive on 910B3 because the contraction has
                # the largest speed margin. Its sole FP key is transformed in
                # place only after the quant value has been built successfully.
                root = scale.to(key_weight.device, dtype=torch.float32).sqrt()
                transformed_fp_key_weight = (
                    key_weight.float() / root[:, None]
                ).to(key_weight.dtype)
                if key_source.bias is not None:
                    transformed_fp_key_bias = (
                        key_source.bias.detach().float() / root
                    ).to(key_source.bias.dtype)
            equalization_rows.append(
                {
                    "layer": pair.layer,
                    "mode": spec.equalization,
                    "min": float(scale.min().cpu()),
                    "max": float(scale.max().cpu()),
                    "sha256": _tensor_sha256(scale),
                    "key_int_codes_preserved": "key" in spec.projections,
                }
            )

        # Build both packed projections before touching the pair. If allocation,
        # packing or validation fails, this RWKV pair remains entirely FP.
        replacements: dict[str, AscendWeightOnlyLinear] = {}
        for projection, weight in (
            ("key", key_weight),
            ("value", transformed_value_weight),
        ):
            if projection not in spec.projections:
                continue
            source = key_source if projection == "key" else value_source
            quant = _quantize_linear(source, weight, spec, admission_scope)
            if projection == "key" and scale is not None:
                _rescale_quantized_key_output(quant, scale)
            replacements[projection] = quant

        # Commit the already-built pair without any remaining fallible packing.
        if transformed_fp_key_weight is not None:
            key_source.weight.copy_(transformed_fp_key_weight)
            if transformed_fp_key_bias is not None:
                assert key_source.bias is not None
                key_source.bias.copy_(transformed_fp_key_bias)
        for projection, quant in replacements.items():
            source = key_source if projection == "key" else value_source
            fp_bytes = source.weight.numel() * source.weight.element_size()
            setattr(pair.module, projection, quant)
            packed_bytes = quant.packed_weight_bytes()
            records.append(
                QuantizedProjectionRecord(
                    layer=pair.layer,
                    projection=projection,
                    module_path=f"{pair.module_path}.{projection}",
                    in_features=source.in_features,
                    out_features=source.out_features,
                    bit=spec.bit,
                    group_size=spec.effective_group_size,
                    fp_weight_bytes_removed=fp_bytes,
                    packed_bytes=packed_bytes,
                    packed_storage_ratio=packed_bytes / fp_bytes,
                )
            )

    floating_copies = []
    for record in records:
        module = model.get_submodule(record.module_path)
        if isinstance(getattr(module, "weight", None), Tensor):
            floating_copies.append(f"{record.module_path}.weight")
    fp_total = sum(item.fp_weight_bytes_removed for item in records)
    packed_total = sum(item.packed_bytes for item in records)
    return RWKV7ModelQuantReport(
        format_version=MODEL_QUANT_FORMAT_VERSION,
        admission_scope=admission_scope,
        spec=asdict(spec),
        projections=records,
        equalization={"layers": equalization_rows},
        fp_weight_bytes_removed=fp_total,
        packed_bytes=packed_total,
        packed_storage_ratio=packed_total / fp_total,
        floating_weight_copies_remaining=floating_copies,
        production_eligible=admission_scope == "production" and not floating_copies,
    )


def _manifest_dict(report_or_manifest: RWKV7ModelQuantReport | Mapping[str, Any]) -> dict[str, Any]:
    return report_or_manifest.to_dict() if isinstance(report_or_manifest, RWKV7ModelQuantReport) else dict(report_or_manifest)


def prepare_rwkv7_for_quantized_state_dict(
    model: nn.Module,
    report_or_manifest: RWKV7ModelQuantReport | Mapping[str, Any],
) -> None:
    """Swap empty quant modules before HF/vLLM/SGLang load a quant state_dict."""
    manifest = _manifest_dict(report_or_manifest)
    if manifest.get("format_version") != MODEL_QUANT_FORMAT_VERSION:
        raise ValueError(f"unsupported model quant format {manifest.get('format_version')}")
    for record in manifest["projections"]:
        path = str(record["module_path"])
        parent_path, attribute = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        current = getattr(parent, attribute)
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{path} must be nn.Linear before quant checkpoint preparation")
        if current.in_features != int(record["in_features"]) or current.out_features != int(record["out_features"]):
            raise ValueError(f"{path} shape does not match quant manifest")
        setattr(
            parent,
            attribute,
            AscendWeightOnlyLinear(
                current.in_features,
                current.out_features,
                current.bias is not None,
                bit=int(record["bit"]),
                group_size=int(record["group_size"]) or 128,
                enforce_verified_shape=manifest.get("admission_scope") != "experiment",
            ),
        )
        prepared = getattr(parent, attribute)
        assert isinstance(prepared, AscendWeightOnlyLinear)
        prepared.admission_scope = manifest.get("admission_scope", "production")


def save_quantized_model_checkpoint(
    model: nn.Module,
    report: RWKV7ModelQuantReport,
    directory: str | Path,
) -> Path:
    """Save a quant-only state_dict plus backend-neutral loader manifest."""
    if report.floating_weight_copies_remaining:
        raise RuntimeError("refusing to save while selected FP weight copies remain")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    state_path = path / "rwkv7_quantized_state.pt"
    torch.save(model.state_dict(), state_path)
    state_sha = _file_sha256(state_path)
    manifest = report.to_dict()
    manifest["state_file"] = state_path.name
    manifest["state_sha256"] = state_sha
    (path / "rwkv7_quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_quantized_model_state(
    model: nn.Module,
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Reference loader seam used by HF and mirrored by engine weight loaders."""
    path = Path(directory)
    manifest_path = path / "rwkv7_quantization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_path = path / manifest["state_file"]
    actual_sha = _file_sha256(state_path)
    if actual_sha != manifest.get("state_sha256"):
        raise ValueError("quantized state_dict SHA256 does not match manifest")
    prepare_rwkv7_for_quantized_state_dict(model, manifest)
    state = torch.load(state_path, map_location=map_location, weights_only=True)
    # Constructor buffers are intentionally zero-sized. Resize them from the
    # authenticated state before asking PyTorch's strict loader to validate all
    # keys, so this path never materializes a floating selected weight.
    for record in manifest["projections"]:
        module_path = str(record["module_path"])
        module = model.get_submodule(module_path)
        assert isinstance(module, AscendWeightOnlyLinear)
        prefix = module_path + "."
        module.load_quantized_buffers(
            state[prefix + "qweight"],
            state[prefix + "scales"],
            state.get(prefix + "offsets"),
            state.get(prefix + "bias"),
        )
    model.load_state_dict(state, strict=strict)
    return manifest


__all__ = [
    "MODEL_QUANT_FORMAT_VERSION",
    "RWKV7FFNPair",
    "RWKV7FFNQuantSpec",
    "RWKV7ModelQuantReport",
    "apply_rwkv7_sqrelu_equalization",
    "compute_rwkv7_sqrelu_equalization_scale",
    "discover_rwkv7_ffn_pairs",
    "load_quantized_model_state",
    "prepare_rwkv7_for_quantized_state_dict",
    "quantize_rwkv7_ffn_model",
    "save_quantized_model_checkpoint",
]
