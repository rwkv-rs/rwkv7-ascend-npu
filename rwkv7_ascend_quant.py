"""Portable, fail-closed weight-only quantization for RWKV-7 on Ascend.

This module is intentionally import-safe on hosts without ``torch_npu``.  The
packed checkpoint format is ordinary PyTorch state_dict data.  NPU execution is
only allowed for device/dtype/shape tuples measured by the acceptance harness;
an unmeasured tuple raises instead of silently violating the latency contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import json
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

FORMAT_VERSION = 1
VERIFIED_STACK = "Ascend910B3 / CANN 8.5.0 / torch_npu 2.9.0 / FP16"
# Raw-operator candidates where both RWKV-7 7.2B FFN projections exceeded the
# same-shape FP16 matmul by >=2%. Only the exact rows independently rechecked by
# the committed 7x200-call clean rebuild are kept here. They are not production
# accepted: the module/model evidence misses the no-regression and quality
# gates. Keep the production table empty until a backend E2E gate passes.
RAW_KERNEL_CANDIDATE_BATCHES = {
    4: (1, 8),
    8: (17, 28),
}
RAW_KERNEL_CANDIDATE_GROUP_SIZES = {4: (128,), 8: (0,)}
PRODUCTION_VERIFIED_BATCHES = {4: (), 8: ()}
# Backwards-compatible name, deliberately describing the production policy.
VERIFIED_BATCHES = PRODUCTION_VERIFIED_BATCHES
VERIFIED_FFN_SHAPES = ((4096, 16384), (16384, 4096))


class UnverifiedQuantShapeError(RuntimeError):
    """Raised when a quantized kernel has no accepted no-regression result."""


@functools.lru_cache(maxsize=1)
def _torch_npu():
    try:
        import torch_npu  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on non-Ascend CI
        raise RuntimeError("torch_npu is required for Ascend execution") from exc
    return torch_npu


@functools.lru_cache(maxsize=1)
def _runtime_device_name() -> str | None:
    try:
        mod = _torch_npu()
        if not torch.npu.is_available():
            return None
        return str(mod.npu.get_device_name(0))
    except (AttributeError, RuntimeError):
        return None


@functools.lru_cache(maxsize=1)
def _runtime_torch_npu_version() -> str | None:
    try:
        return str(_torch_npu().__version__)
    except RuntimeError:
        return None


def _runtime_torch_version() -> str:
    return str(torch.__version__)


@functools.lru_cache(maxsize=1)
def _runtime_cann_version() -> str | None:
    """Best-effort CANN discovery; an unknown value deliberately fails closed."""
    try:
        mod = _torch_npu()
        for owner in (getattr(mod, "npu", None), mod):
            getter = getattr(owner, "get_cann_version", None)
            if callable(getter):
                value = getter()
                if value:
                    return str(value)
    except (AttributeError, RuntimeError, TypeError):
        pass
    for path in (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info"),
    ):
        try:
            contents = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in contents.splitlines():
            if "version" in line.lower() and "=" in line:
                return line.split("=", 1)[1].strip().strip('"')
    return None


def is_raw_kernel_candidate(
    in_features: int,
    out_features: int,
    batch: int,
    bit: int,
    *,
    group_size: int | None = None,
    dtype: torch.dtype = torch.float16,
    device_name: str | None = None,
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    cann_version: str | None = None,
) -> bool:
    """Return whether a tuple passed only the synchronized raw-op microbench.

    This experimental candidate query is not permission to replace a production
    layer. Exact hardware, PyTorch, torch_npu and CANN identities are required.
    """
    if bit not in RAW_KERNEL_CANDIDATE_BATCHES or dtype is not torch.float16:
        return False
    effective_group_size = 0 if bit == 8 else int(128 if group_size is None else group_size)
    if effective_group_size not in RAW_KERNEL_CANDIDATE_GROUP_SIZES[bit]:
        return False
    if (int(in_features), int(out_features)) not in VERIFIED_FFN_SHAPES:
        return False
    if int(batch) not in RAW_KERNEL_CANDIDATE_BATCHES[bit]:
        return False
    name = device_name if device_name is not None else _runtime_device_name()
    pt = torch_version if torch_version is not None else _runtime_torch_version()
    npu = torch_npu_version if torch_npu_version is not None else _runtime_torch_npu_version()
    cann = cann_version if cann_version is not None else _runtime_cann_version()
    return (
        name == "Ascend910B3"
        and pt.split("+", 1)[0] == "2.9.0"
        and npu == "2.9.0"
        and cann == "8.5.0"
    )


def should_quantize(
    in_features: int,
    out_features: int,
    batch: int,
    bit: int,
    *,
    group_size: int | None = None,
    dtype: torch.dtype = torch.float16,
    device_name: str | None = None,
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    cann_version: str | None = None,
) -> bool:
    """Return whether the exact tuple passed the production end-to-end gate.

    The table is currently empty: raw kernels pass selected tuples, but module
    and HF evidence do not satisfy no-regression and quality contracts. Thus
    callers retain their FP16/BF16 layer. Use :func:`is_raw_kernel_candidate`
    only in explicit benchmark experiments.
    """
    del in_features, out_features, batch, bit, group_size, dtype, device_name
    del torch_version, torch_npu_version, cann_version
    return False


def _pack_int4_cpu(q_kn: Tensor) -> Tensor:
    """Pack signed int4 ``[K,N]`` into eight nibbles per int32."""
    if q_kn.ndim != 2 or q_kn.shape[1] % 8:
        raise ValueError("int4 output dimension must be divisible by 8")
    q = q_kn.to(device="cpu", dtype=torch.int64).reshape(q_kn.shape[0], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    return torch.sum((q & 0xF) << shifts, dim=-1).to(torch.int32).contiguous()


def _unpack_int4_cpu(packed: Tensor, out_features: int) -> Tensor:
    """Unpack the checkpoint's int32 format into signed int8 ``[K,N]``."""
    p = packed.detach().to(device="cpu", dtype=torch.int64)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    q = ((p.unsqueeze(-1) >> shifts) & 0xF)
    q = torch.where(q >= 8, q - 16, q).to(torch.int8)
    return q.reshape(p.shape[0], -1)[:, :out_features].contiguous()


@dataclass(frozen=True)
class QuantManifest:
    format_version: int
    implementation: str
    bit: int
    group_size: int
    in_features: int
    out_features: int
    bias: bool
    weight_layout: str
    scale_layout: str
    verified_stack: str
    verified_batches: tuple[int, ...]
    raw_candidate_batches: tuple[int, ...]
    acceptance_scope: str


class AscendWeightOnlyLinear(nn.Module):
    """Inference-only Ascend W8A16 or groupwise W4A16 linear layer.

    ``load_fp_weight`` accepts a normal ``[out_features, in_features]`` FP16
    weight and drops it after packing.  Consequently the module delivers a real
    weight-memory reduction; it does not hide a second FP16 fallback copy.
    Call :func:`should_quantize` before replacing a layer or run with
    ``enforce_verified_shape=False`` only for explicit experiments.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        bit: int = 8,
        group_size: int = 128,
        enforce_verified_shape: bool = True,
    ) -> None:
        super().__init__()
        if bit not in (4, 8):
            raise ValueError("bit must be 4 or 8")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear dimensions must be positive")
        if bit == 4 and (in_features % group_size or out_features % 8):
            raise ValueError("W4 requires in_features % group_size == 0 and out_features % 8 == 0")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bit = int(bit)
        self.group_size = int(group_size if bit == 4 else 0)
        self.enforce_verified_shape = bool(enforce_verified_shape)
        self.admission_scope: Literal["production", "raw-candidate", "experiment"] = (
            "production" if self.enforce_verified_shape else "experiment"
        )
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32 if bit == 4 else torch.int8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("offsets", torch.empty(0, dtype=torch.float16), persistent=bit == 4)
        self.register_buffer("bias", torch.empty(0, dtype=torch.float16), persistent=bias)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.has_bias = bool(bias)
        # Per-M closures capture packed buffers and the raw op after one policy
        # check. They are deliberately non-persistent and cleared by .to().
        self._npu_fastpaths: dict[int, Any] = {}
        # Kept outside the state_dict. ``prepare_npu_kernel`` resolves the
        # Python wrapper once, rather than on every token/layer invocation.
        self._npu_op: Any | None = None

    def _apply(self, fn, recurse: bool = True):
        self._npu_fastpaths.clear()
        self._npu_op = None
        return super()._apply(fn, recurse=recurse)

    @property
    def manifest(self) -> QuantManifest:
        return QuantManifest(
            format_version=FORMAT_VERSION,
            implementation="rwkv7_ascend_quant.AscendWeightOnlyLinear",
            bit=self.bit,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.has_bias,
            weight_layout="[K,N] int8" if self.bit == 8 else "[K,N/8] int32, eight signed nibbles",
            scale_layout="[N]" if self.bit == 8 else "[K/group_size,N]",
            verified_stack=VERIFIED_STACK,
            verified_batches=VERIFIED_BATCHES[self.bit],
            raw_candidate_batches=RAW_KERNEL_CANDIDATE_BATCHES[self.bit],
            acceptance_scope="production-disabled; raw-kernel-candidate-only",
        )

    @torch.no_grad()
    def load_fp_weight(self, weight: Tensor, bias: Tensor | None = None) -> "AscendWeightOnlyLinear":
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} != {(self.out_features, self.in_features)}"
            )
        if not weight.dtype.is_floating_point:
            raise TypeError("source weight must be floating point")
        original_device = weight.device
        if self.bit == 8:
            wf = weight.float()
            scale = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
            q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
            self.qweight = q.t().contiguous().to(original_device)
            self.scales = scale.to(device=original_device, dtype=torch.float16).contiguous()
            self.offsets = torch.empty(0, device=original_device, dtype=torch.float16)
        else:
            wf = weight.float()
            groups = self.in_features // self.group_size
            grouped = wf.reshape(self.out_features, groups, self.group_size)
            scale = (grouped.abs().amax(dim=2) / 7.0).clamp_min(1e-8)
            q_nk = torch.round(grouped / scale[:, :, None]).clamp(-8, 7).to(torch.int32)
            q_kn = q_nk.reshape(self.out_features, self.in_features).t().contiguous()
            if original_device.type == "npu":
                packed = _torch_npu().npu_convert_weight_to_int4pack(q_kn)
            else:
                packed = _pack_int4_cpu(q_kn)
            self.qweight = packed.to(original_device)
            self.scales = scale.t().to(device=original_device, dtype=torch.float16).contiguous()
            self.offsets = torch.zeros_like(self.scales)
        if self.has_bias:
            if bias is None or tuple(bias.shape) != (self.out_features,):
                raise ValueError(f"bias must have shape {(self.out_features,)}")
            self.bias = bias.detach().to(device=original_device, dtype=torch.float16).contiguous()
        elif bias is not None:
            raise ValueError("module was constructed without bias")
        self.initialized = torch.tensor(True, device=original_device, dtype=torch.bool)
        self._npu_fastpaths.clear()
        self._npu_op = None
        return self

    @torch.no_grad()
    def load_quantized_buffers(
        self,
        qweight: Tensor,
        scales: Tensor,
        offsets: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> "AscendWeightOnlyLinear":
        """Initialize directly from a quant-only checkpoint.

        This path never creates an FP16 weight and is the loading seam shared by
        HF, vLLM and SGLang adapters.  Shape and dtype checks are deliberately
        strict so a malformed checkpoint cannot silently take a different op
        ABI.
        """
        expected_qshape = (
            (self.in_features, self.out_features)
            if self.bit == 8
            else (self.in_features, self.out_features // 8)
        )
        expected_scale_shape = (
            (self.out_features,)
            if self.bit == 8
            else (self.in_features // self.group_size, self.out_features)
        )
        expected_qdtype = torch.int8 if self.bit == 8 else torch.int32
        if tuple(qweight.shape) != expected_qshape or qweight.dtype is not expected_qdtype:
            raise ValueError(
                f"qweight must be {expected_qshape} {expected_qdtype}, got "
                f"{tuple(qweight.shape)} {qweight.dtype}"
            )
        if tuple(scales.shape) != expected_scale_shape or scales.dtype is not torch.float16:
            raise ValueError(
                f"scales must be {expected_scale_shape} torch.float16, got "
                f"{tuple(scales.shape)} {scales.dtype}"
            )
        device = qweight.device
        if scales.device != device:
            raise ValueError("qweight and scales must be on the same device")
        if self.bit == 4:
            if offsets is None:
                offsets = torch.zeros_like(scales)
            if tuple(offsets.shape) != expected_scale_shape or offsets.dtype is not torch.float16:
                raise ValueError("W4 offsets must match scales and use torch.float16")
            if offsets.device != device:
                raise ValueError("W4 offsets must be on the packed-weight device")
        elif offsets is not None and offsets.numel():
            raise ValueError("W8 does not accept offsets")
        if self.has_bias:
            if bias is None or tuple(bias.shape) != (self.out_features,):
                raise ValueError(f"bias must have shape {(self.out_features,)}")
            if bias.dtype is not torch.float16 or bias.device != device:
                raise ValueError("bias must be torch.float16 on the packed-weight device")
        elif bias is not None and bias.numel():
            raise ValueError("module was constructed without bias")
        self.qweight = qweight.contiguous()
        self.scales = scales.contiguous()
        self.offsets = (
            offsets.contiguous()
            if self.bit == 4 and offsets is not None
            else torch.empty(0, device=device, dtype=torch.float16)
        )
        self.bias = (
            bias.contiguous()
            if self.has_bias and bias is not None
            else torch.empty(0, device=device, dtype=torch.float16)
        )
        self.initialized = torch.tensor(True, device=device, dtype=torch.bool)
        self._npu_fastpaths.clear()
        self._npu_op = None
        return self

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        *,
        bit: int = 8,
        group_size: int = 128,
        enforce_verified_shape: bool = True,
    ) -> "AscendWeightOnlyLinear":
        result = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bit=bit,
            group_size=group_size,
            enforce_verified_shape=enforce_verified_shape,
        )
        return result.load_fp_weight(linear.weight.detach(), None if linear.bias is None else linear.bias.detach())

    def packed_weight_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.qweight, self.scales, self.offsets, self.bias)
        )

    def bind_npu_fastpath(
        self,
        batch: int,
        *,
        dtype: torch.dtype = torch.float16,
        scope: Literal["production", "raw-candidate", "experiment"] = "production",
    ) -> Callable[[Tensor], Tensor]:
        """Return a cached raw-op closure after a single acceptance check.

        Serving schedulers should create this once per accepted batch row count,
        then reuse it across layers/tokens or graph capture. This avoids repeating
        Python policy and buffer lookup on every projection.
        """
        if self.qweight.device.type != "npu":
            raise RuntimeError("packed buffers must be on an NPU")
        if scope == "production":
            accepted = should_quantize(
                self.in_features,
                self.out_features,
                batch,
                self.bit,
                group_size=self.group_size,
                dtype=dtype,
            )
        elif scope == "raw-candidate":
            accepted = is_raw_kernel_candidate(
                self.in_features,
                self.out_features,
                batch,
                self.bit,
                group_size=self.group_size,
                dtype=dtype,
            )
        elif scope == "experiment":
            accepted = not self.enforce_verified_shape
        else:  # pragma: no cover - Literal catches this for typed callers
            raise ValueError(f"unknown acceptance scope {scope!r}")
        if not accepted:
            raise UnverifiedQuantShapeError(
                f"no {scope} acceptance for {VERIFIED_STACK}: bit={self.bit}, "
                f"group_size={self.group_size}, M={batch}, K={self.in_features}, "
                f"N={self.out_features}"
            )
        op = _torch_npu().npu_weight_quant_batchmatmul
        self._npu_op = op
        qweight, scales = self.qweight, self.scales
        bias = self.bias if self.has_bias else None
        if self.bit == 8:
            if bias is None:
                def kernel(x):
                    return op(x, qweight, scales)
            else:
                def kernel(x):
                    return op(x, qweight, scales) + bias
        else:
            offsets, group_size = self.offsets, self.group_size
            if bias is None:
                def kernel(x):
                    return op(x, qweight, scales, offsets, None, None, None, group_size, 1)
            else:
                def kernel(x):
                    return op(x, qweight, scales, offsets, None, None, None, group_size, 1) + bias
        self._npu_fastpaths[int(batch)] = kernel
        return kernel

    def make_npu_fastpath(self, batch: int, *, dtype: torch.dtype = torch.float16):
        """Backward-compatible production binding (or explicit experiment).

        New backend integrations should call :meth:`bind_npu_fastpath` with an
        explicit scope.  ``enforce_verified_shape=False`` remains an opt-in
        benchmark escape hatch and can never populate the production table.
        """
        return self.bind_npu_fastpath(batch, dtype=dtype, scope=self.admission_scope)

    def forward_unchecked(self, x2: Tensor) -> Tensor:
        """Call the NPU op with no Python policy/shape dispatch in the hot path.

        Callers must first obtain admission with :meth:`bind_npu_fastpath`.
        This method exists for model code that cannot retain the returned bound
        closure.  Serving adapters should prefer the closure, which is exactly
        the raw operator call and is graph-capture friendly.
        """
        op = self._npu_op
        if op is None:
            raise RuntimeError("bind_npu_fastpath must be called before forward_unchecked")
        if self.bit == 8:
            out = op(x2, self.qweight, self.scales)
        else:
            out = op(
                x2,
                self.qweight,
                self.scales,
                self.offsets,
                None,
                None,
                None,
                self.group_size,
                1,
            )
        return out + self.bias if self.has_bias else out

    def _cpu_forward(self, x2: Tensor) -> Tensor:
        if self.bit == 8:
            w = self.qweight.t().float() * self.scales.float()[:, None]
        else:
            q_kn = _unpack_int4_cpu(self.qweight, self.out_features).float()
            scale_kn = self.scales.float().repeat_interleave(self.group_size, dim=0)
            w = (q_kn * scale_kn).t().contiguous()
        out = F.linear(x2.float(), w, None)
        if self.has_bias:
            out = out + self.bias.float()
        return out.to(x2.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.qweight.numel() == 0 or self.scales.numel() == 0:
            raise RuntimeError("load_fp_weight or load_state_dict must initialize the module")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"last input dimension must be {self.in_features}")
        if x.dtype is not torch.float16:
            raise TypeError("accepted Ascend kernels are FP16-activation only")
        is_matrix = x.ndim == 2
        shape = x.shape[:-1] + (self.out_features,)
        x2 = x if is_matrix else x.reshape(-1, self.in_features)
        if x.device.type != "npu":
            result = self._cpu_forward(x2)
            return result if is_matrix else result.reshape(shape)
        batch = int(x2.shape[0])
        kernel = self._npu_fastpaths.get(batch)
        if kernel is None:
            kernel = self.make_npu_fastpath(batch, dtype=x.dtype)
        out = kernel(x2)
        return out if is_matrix else out.reshape(shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit={self.bit}, group_size={self.group_size}, bias={self.has_bias}, "
            f"enforce_verified_shape={self.enforce_verified_shape}"
        )


def save_quantized_linear(module: AscendWeightOnlyLinear, directory: str | Path) -> Path:
    """Save a standard state_dict plus a human-readable format manifest."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path / "quantized_linear.pt")
    (path / "quant_manifest.json").write_text(
        json.dumps(asdict(module.manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_quantized_linear(
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    enforce_verified_shape: bool = True,
) -> AscendWeightOnlyLinear:
    path = Path(directory)
    manifest: dict[str, Any] = json.loads((path / "quant_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported quant format version {manifest.get('format_version')}")
    module = AscendWeightOnlyLinear(
        manifest["in_features"],
        manifest["out_features"],
        manifest["bias"],
        bit=manifest["bit"],
        group_size=manifest["group_size"] or 128,
        enforce_verified_shape=enforce_verified_shape,
    )
    state = torch.load(path / "quantized_linear.pt", map_location=map_location, weights_only=True)
    # Empty constructor buffers intentionally resize to the packed checkpoint.
    for key in ("qweight", "scales", "offsets", "bias", "initialized"):
        if key in state:
            setattr(module, key, state[key])
    module.load_state_dict(state, strict=True)
    return module


__all__ = [
    "AscendWeightOnlyLinear",
    "QuantManifest",
    "UnverifiedQuantShapeError",
    "is_raw_kernel_candidate",
    "load_quantized_linear",
    "save_quantized_linear",
    "should_quantize",
]
