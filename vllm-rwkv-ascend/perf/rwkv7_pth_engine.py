"""Load an official BlinkDL RWKV-7 ``.pth`` into the fused decode contract.

The direct backend batches the four w/a/g/v low-rank chains.  Official
checkpoints use different ranks for those chains, so this module pads each
chain to the largest per-layer rank before packing.  Both projection matrices
are padded with zeros, preserving the original result exactly.
"""
from __future__ import annotations

import os
import types
from collections.abc import Sequence

import torch
import torch.nn.functional as F


LOADER_PROFILES = ("full", "prefill_only")


def normalize_loader_profile(profile: str) -> str:
    """Validate the model materialization contract used by a benchmark."""
    if profile not in LOADER_PROFILES:
        raise ValueError(
            f"unsupported loader profile {profile!r}; expected one of "
            + ", ".join(LOADER_PROFILES)
        )
    return profile


def unique_tensor_bytes(value) -> int:
    """Count unique tensor storage bytes in nested list/tuple structures."""
    tensors = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_iter_tensors(item))
    seen = set()
    total = 0
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        storage_key = (tensor.device.type, tensor.untyped_storage().data_ptr())
        if storage_key not in seen:
            seen.add(storage_key)
            total += tensor.untyped_storage().nbytes()
    return total


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _pad_output_features(weight: torch.Tensor, target: int) -> torch.Tensor:
    """Pad an ``[out, in]`` linear weight to ``target`` output features."""
    if weight.dim() != 2 or weight.shape[0] > target:
        raise ValueError(f"cannot output-pad weight {tuple(weight.shape)} to {target}")
    if weight.shape[0] == target:
        return weight
    return F.pad(weight, (0, 0, 0, target - weight.shape[0]))


def _pad_input_features(weight: torch.Tensor, target: int) -> torch.Tensor:
    """Pad an ``[out, in]`` linear weight to ``target`` input features."""
    if weight.dim() != 2 or weight.shape[1] > target:
        raise ValueError(f"cannot input-pad weight {tuple(weight.shape)} to {target}")
    if weight.shape[1] == target:
        return weight
    return F.pad(weight, (0, target - weight.shape[1], 0, 0))


def pack_lowrank_layer(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack four unequal-rank linear chains for two batched matmuls.

    ``first`` contains ``[rank, hidden]`` weights and ``second`` contains
    ``[hidden, rank]`` weights.  Returned BMM layouts are ``[4, hidden, rank]``
    and ``[4, rank, hidden]``.
    """
    if not (len(first) == len(second) == len(biases) == 4):
        raise ValueError("low-rank packing requires exactly four chains")
    hidden = first[0].shape[1]
    target_rank = max(weight.shape[0] for weight in first)
    for index, (down, up, bias) in enumerate(zip(first, second, biases)):
        if down.dim() != 2 or up.dim() != 2:
            raise ValueError(f"low-rank chain {index} weights must be matrices")
        if down.shape[1] != hidden or up.shape != (hidden, down.shape[0]):
            raise ValueError(
                f"low-rank chain {index} has incompatible shapes "
                f"{tuple(down.shape)} and {tuple(up.shape)}"
            )
        if bias.shape != (hidden,):
            raise ValueError(
                f"low-rank chain {index} bias has shape {tuple(bias.shape)}, "
                f"expected {(hidden,)}"
            )
    first_bmm = torch.stack(
        [_pad_output_features(weight, target_rank).t() for weight in first]
    ).contiguous()
    second_bmm = torch.stack(
        [_pad_input_features(weight, target_rank).t() for weight in second]
    ).contiguous()
    bias_bmm = torch.stack(list(biases))[:, None, :].contiguous()
    return first_bmm, second_bmm, bias_bmm, target_rank


def make_folded_mix_project_weight(
    weights: Sequence[torch.Tensor], mixes: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Fold current/previous token mixing into one projection weight.

    The first three weights are full R/K/V projections.  The remaining four
    are low-rank down projections and are padded to a common output rank.
    """
    if len(weights) != 7 or len(mixes) != 7:
        raise ValueError("folded mix projection requires seven weight/mix pairs")
    hidden = weights[0].shape[1]
    target_rank = max(weight.shape[0] for weight in weights[3:])
    folded = []
    for index, (weight, mix) in enumerate(zip(weights, mixes)):
        if weight.dim() != 2 or weight.shape[1] != hidden:
            raise ValueError(f"invalid projection weight {index}: {tuple(weight.shape)}")
        if mix.numel() != hidden:
            raise ValueError(f"invalid mix vector {index}: {tuple(mix.shape)}")
        if index >= 3:
            weight = _pad_output_features(weight, target_rank)
        weight_fp32 = weight.float()
        mix_fp32 = mix.reshape(1, hidden).float()
        folded.append(
            torch.cat(
                (weight_fp32 * (1.0 - mix_fp32), weight_fp32 * mix_fp32),
                dim=1,
            ).to(weight.dtype)
        )
    return torch.cat(folded, dim=0).contiguous()


class _FrozenEmbedding(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_blinkdl_engine(
    cpp_source: str,
    *,
    model_path: str,
    device: str,
    head_size: int = 64,
    include_mix_project: bool = False,
    loader_profile: str = "full",
    extension_name: str = "rwkv7_ascend_real_checkpoint",
    extra_cflags: Sequence[str] | None = None,
):
    """Build the existing fused-engine namespace from a BlinkDL checkpoint."""
    loader_profile = normalize_loader_profile(loader_profile)
    if loader_profile == "prefill_only" and include_mix_project:
        raise ValueError("prefill_only does not materialize mix-project weights")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    raw = _load_checkpoint(model_path)
    layer_ids = sorted(
        {
            int(key.split(".")[1])
            for key in raw
            if key.startswith("blocks.") and key.split(".")[1].isdigit()
        }
    )
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError(f"checkpoint has non-contiguous layer ids: {layer_ids}")
    layers = len(layer_ids)
    hidden = int(raw["emb.weight"].shape[1])
    if hidden % head_size:
        raise ValueError(f"hidden size {hidden} is not divisible by head size {head_size}")
    heads = hidden // head_size
    vocab_size = int(raw["emb.weight"].shape[0])
    dtype = torch.float16

    if loader_profile == "full":
        from torch.utils.cpp_extension import load

        mod = load(
            name=extension_name,
            sources=[cpp_source],
            verbose=False,
            extra_cflags=list(extra_cflags or ("-O3", "-std=c++17")),
        )
    else:
        mod = None

    def move(key: str, *, transpose: bool = False) -> torch.Tensor:
        if key not in raw:
            raise KeyError(f"checkpoint is missing {key}")
        tensor = raw[key].squeeze()
        if transpose:
            tensor = tensor.t()
        return tensor.to(device=device, dtype=dtype).contiguous()

    def per_layer(suffix: str, *, transpose: bool = False) -> list[torch.Tensor]:
        return [
            move(f"blocks.{layer}.{suffix}", transpose=transpose)
            for layer in range(layers)
        ]

    rw = per_layer("att.receptance.weight")
    kw = per_layer("att.key.weight")
    vw = per_layer("att.value.weight")
    ow = per_layer("att.output.weight")
    fkw = per_layer("ffn.key.weight")
    fvw = per_layer("ffn.value.weight")

    # BlinkDL stores x @ w1 and x @ w2; at::linear expects [out, in].
    w0 = per_layer("att.w1", transpose=True)
    w2 = per_layer("att.w2", transpose=True)
    a0 = per_layer("att.a1", transpose=True)
    a2 = per_layer("att.a2", transpose=True)
    g0 = per_layer("att.g1", transpose=True)
    g2 = per_layer("att.g2", transpose=True)

    v0: list[torch.Tensor] = []
    v2: list[torch.Tensor] = []
    v2b: list[torch.Tensor] = []
    for layer in range(layers):
        prefix = f"blocks.{layer}.att"
        if f"{prefix}.v1" in raw:
            v0.append(move(f"{prefix}.v1", transpose=True))
            v2.append(move(f"{prefix}.v2", transpose=True))
            v2b.append(move(f"{prefix}.v0").reshape(-1))
        else:
            # Layer zero does not consume value mixing.  Keep a valid zero chain
            # so the four-way BMM contract remains uniform.
            v0.append(torch.zeros_like(w0[layer]))
            v2.append(torch.zeros_like(w2[layer]))
            v2b.append(torch.zeros(hidden, device=device, dtype=dtype))

    w2b = [tensor.reshape(-1) for tensor in per_layer("att.w0")]
    a2b = [tensor.reshape(-1) for tensor in per_layer("att.a0")]
    xr = [tensor.reshape(-1) for tensor in per_layer("att.x_r")]
    xw = [tensor.reshape(-1) for tensor in per_layer("att.x_w")]
    xk = [tensor.reshape(-1) for tensor in per_layer("att.x_k")]
    xv = [tensor.reshape(-1) for tensor in per_layer("att.x_v")]
    xa = [tensor.reshape(-1) for tensor in per_layer("att.x_a")]
    xg = [tensor.reshape(-1) for tensor in per_layer("att.x_g")]
    kk = [tensor.reshape(-1) for tensor in per_layer("att.k_k")]
    ka = [tensor.reshape(-1) for tensor in per_layer("att.k_a")]
    rk = [tensor.reshape(-1) for tensor in per_layer("att.r_k")]
    gnw = [tensor.reshape(-1) for tensor in per_layer("att.ln_x.weight")]
    gnb = [tensor.reshape(-1) for tensor in per_layer("att.ln_x.bias")]
    fxk = [tensor.reshape(-1) for tensor in per_layer("ffn.x_k")]
    anw = [tensor.reshape(-1) for tensor in per_layer("ln1.weight")]
    anb = [tensor.reshape(-1) for tensor in per_layer("ln1.bias")]
    fnw = [tensor.reshape(-1) for tensor in per_layer("ln2.weight")]
    fnb = [tensor.reshape(-1) for tensor in per_layer("ln2.bias")]

    pre_weight = move("blocks.0.ln0.weight").reshape(-1)
    pre_bias = move("blocks.0.ln0.bias").reshape(-1)
    ones = torch.ones(hidden, device=device, dtype=dtype)
    zeros = torch.zeros(hidden, device=device, dtype=dtype)
    pnw = [pre_weight] + [ones] * (layers - 1)
    pnb = [pre_bias] + [zeros] * (layers - 1)

    if loader_profile == "full":
        rkv_bmm = [
            torch.stack((rw[layer].t(), kw[layer].t(), vw[layer].t())).contiguous()
            for layer in range(layers)
        ]
    else:
        rkv_bmm = [
            torch.empty(0, device=device, dtype=dtype) for _ in range(layers)
        ]
    lowrank_first = []
    lowrank_second = []
    lowrank_bias = []
    mix_project = []
    for layer in range(layers):
        first, second, bias, _ = pack_lowrank_layer(
            (w0[layer], a0[layer], g0[layer], v0[layer]),
            (w2[layer], a2[layer], g2[layer], v2[layer]),
            (w2b[layer], a2b[layer], zeros, v2b[layer]),
        )
        lowrank_first.append(first)
        lowrank_second.append(second)
        lowrank_bias.append(bias)
        if include_mix_project:
            mix_project.append(
                make_folded_mix_project_weight(
                    (
                        rw[layer], kw[layer], vw[layer], w0[layer],
                        a0[layer], g0[layer], v0[layer],
                    ),
                    (
                        xr[layer], xk[layer], xv[layer], xw[layer],
                        xa[layer], xg[layer], xv[layer],
                    ),
                )
            )
        else:
            # Keep the 40-group C++ call contract without paying the hundreds
            # of MiB folded-weight cost when that experiment is disabled.
            mix_project.append(torch.empty(0, device=device, dtype=dtype))

    if loader_profile == "prefill_only":
        # The layer-major prefill path consumes only the packed low-rank chains.
        # Drop the eleven raw chain groups before returning so large checkpoints
        # do not retain both representations.  R/K/V stay resident because the
        # prefill projection path consumes them directly.
        del w0, w2, a0, a2, g0, g2, v0, v2, w2b, a2b, v2b
        unused = [
            torch.empty(0, device=device, dtype=dtype) for _ in range(layers)
        ]
        raw_lowrank_groups = (unused,) * 11
    else:
        raw_lowrank_groups = (
            w0, w2, a0, a2, g0, g2, v0, v2, w2b, a2b, v2b,
        )

    eng = types.SimpleNamespace()
    eng.L, eng.H, eng.N, eng.hidden = layers, heads, head_size, hidden
    eng.vocab_size = vocab_size
    eng.model_path = model_path
    eng.loader_profile = loader_profile
    eng.mod = mod
    eng.W = (
        rw, kw, vw, rkv_bmm, ow, fkw, fvw,
        *raw_lowrank_groups,
        xr, xw, xk, xv, xa, xg,
        kk, ka, rk, gnw, gnb, fxk,
        anw, anb, fnw, fnb, pnw, pnb,
        lowrank_first, lowrank_second, lowrank_bias, mix_project,
    )
    embedding_weight = move("emb.weight")
    eng.base = types.SimpleNamespace(embeddings=_FrozenEmbedding(embedding_weight))
    eng.lm_w_m = move("head.weight")
    eng.fnorm_w = move("ln_out.weight").reshape(-1)
    eng.fnorm_b = move("ln_out.bias").reshape(-1)
    eng.packed_tensor_bytes = unique_tensor_bytes(
        (eng.W[3], eng.W[36], eng.W[37], eng.W[38], eng.W[39])
    )
    eng.resident_tensor_bytes = unique_tensor_bytes(
        (
            eng.W,
            eng.base.embeddings.weight,
            eng.lm_w_m,
            eng.fnorm_w,
            eng.fnorm_b,
        )
    )
    del raw
    return eng
