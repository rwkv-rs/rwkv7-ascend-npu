#!/usr/bin/env python3
"""Real-checkpoint HF W8/W4 diagnostic on one Ascend device.

This is intentionally an experimental gate.  It quantizes every RWKV-7 FFN
projection, removes the corresponding floating weights, binds the exact
single-token raw operator once, and compares forced-path logits, greedy choices,
model tensor footprint, allocator HBM, and synchronized decode wall time against
an independently loaded FP16 model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from rwkv7_hf import enable_ascend
from rwkv7_ascend_model_quant import (
    RWKV7FFNQuantSpec,
    quantize_rwkv7_ffn_model,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(model: torch.nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bit", type=int, choices=(4, 8), required=True)
    parser.add_argument(
        "--projections",
        choices=("all", "key", "value"),
        default="all",
        help="FFN projections to quantize; value-only is the 910B3 decode candidate",
    )
    parser.add_argument(
        "--equalization",
        choices=("none", "weight-cle"),
        default="none",
    )
    parser.add_argument("--quality-steps", type=int, default=8)
    parser.add_argument("--timed-steps", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=7)
    return parser.parse_args()


def load(model_path: Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    if type(model).__name__ != "NativeRWKV7ForCausalLM":
        raise RuntimeError(f"unexpected AutoModel class {type(model)}")
    return model.to("npu:0")


@torch.inference_mode()
def trace(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steps: int,
    *,
    forced: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    out = model(input_ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    logits: list[torch.Tensor] = []
    predicted: list[int] = []
    for index in range(steps):
        current = out.logits[:, -1].float().cpu()
        logits.append(current)
        predicted.append(int(torch.argmax(out.logits[:, -1], dim=-1).item()))
        token_id = predicted[-1] if forced is None else int(forced[index])
        token = torch.tensor([[token_id]], device="npu:0", dtype=torch.long)
        out = model(token, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
    return logits, predicted


@torch.inference_mode()
def timed_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    forced: list[torch.Tensor],
) -> float:
    out = model(input_ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    torch.npu.synchronize()
    started = time.perf_counter()
    for token in forced:
        out = model(token, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
    torch.npu.synchronize()
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.equalization != "none" and args.bit != 4:
        raise ValueError("equalization is only used by the W4 experiment")
    projections = (
        ("key", "value") if args.projections == "all" else (args.projections,)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enable_ascend("npu:0", backend="eager")
    torch.manual_seed(20260724)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="npu:0", dtype=torch.long)
    forced = [
        torch.tensor([[100 + index]], device="npu:0", dtype=torch.long)
        for index in range(args.timed_steps)
    ]

    dense = load(args.model)
    torch.npu.synchronize()
    dense_allocated = int(torch.npu.memory_allocated())
    dense_reserved = int(torch.npu.memory_reserved())
    dense_bytes = tensor_bytes(dense)
    dense_logits, dense_tokens = trace(dense, input_ids, args.quality_steps)

    candidate = load(args.model)
    torch.npu.synchronize()
    spec = RWKV7FFNQuantSpec(
        bit=args.bit,
        group_size=128,
        projections=projections,
        admitted_rows=(1,),
        equalization=args.equalization,
    )
    quant_started = time.perf_counter()
    report = quantize_rwkv7_ffn_model(
        candidate,
        spec,
        admission_scope="experiment",
        allow_unverified_experiment=True,
    )
    torch.npu.synchronize()
    quantize_seconds = time.perf_counter() - quant_started

    # Bind once per layer. This deliberately measures the intended serving hot
    # path rather than repeating runtime/policy discovery in every projection.
    for record in report.projections:
        module = candidate.get_submodule(record.module_path)
        raw = module.bind_npu_fastpath(1, scope="experiment")
        in_features = module.in_features
        out_features = module.out_features

        def bound_forward(
            value: torch.Tensor,
            *,
            operation=raw,
            k=in_features,
            n=out_features,
        ) -> torch.Tensor:
            shape = value.shape[:-1]
            return operation(value.reshape(-1, k)).reshape(*shape, n)

        module.forward = bound_forward

    candidate_logits, candidate_tokens = trace(
        candidate,
        input_ids,
        args.quality_steps,
        forced=dense_tokens,
    )
    timed_decode(dense, input_ids, forced)
    timed_decode(candidate, input_ids, forced)
    pairs = []
    for index in range(args.rounds):
        if index % 2 == 0:
            dense_s = timed_decode(dense, input_ids, forced)
            quant_s = timed_decode(candidate, input_ids, forced)
            order = "dense-quant"
        else:
            quant_s = timed_decode(candidate, input_ids, forced)
            dense_s = timed_decode(dense, input_ids, forced)
            order = "quant-dense"
        pairs.append(
            {
                "order": order,
                "dense_seconds": dense_s,
                "quant_seconds": quant_s,
                "speedup": dense_s / quant_s,
            }
        )

    cosine = []
    nrmse = []
    kl = []
    top20 = []
    for reference, actual in zip(dense_logits, candidate_logits):
        cosine.append(
            F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0).item()
        )
        nrmse.append(
            float(
                (reference - actual).square().mean().sqrt()
                / reference.square().mean().sqrt()
            )
        )
        probabilities = F.softmax(reference, dim=-1)
        kl.append(
            float(
                (
                    probabilities
                    * (F.log_softmax(reference, -1) - F.log_softmax(actual, -1))
                ).sum()
            )
        )
        reference_top = set(torch.topk(reference, 20).indices.flatten().tolist())
        actual_top = set(torch.topk(actual, 20).indices.flatten().tolist())
        top20.append(len(reference_top & actual_top) / 20)

    quant_bytes = tensor_bytes(candidate)
    del dense
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    quant_allocated = int(torch.npu.memory_allocated())
    quant_reserved = int(torch.npu.memory_reserved())
    median_dense = statistics.median(row["dense_seconds"] for row in pairs)
    median_quant = statistics.median(row["quant_seconds"] for row in pairs)
    ratio_of_medians = median_dense / median_quant
    # Pairing preserves the alternating execution order and avoids declaring a
    # win merely because the independently selected medians came from different
    # thermal/host-load conditions.
    speedup = statistics.median(row["speedup"] for row in pairs)
    quality_floor = 0.999 if args.bit == 8 else 0.99
    nrmse_ceiling = 0.05 if args.bit == 8 else 0.20
    gates = {
        "tensor_footprint_reduced": quant_bytes < dense_bytes,
        "allocator_hbm_reduced": quant_allocated < dense_allocated,
        "decode_not_slower": speedup >= 1.0,
        "minimum_logit_cosine": min(cosine) >= quality_floor,
        "maximum_logit_nrmse": max(nrmse) <= nrmse_ceiling,
        "greedy_choices_equal_on_dense_path": dense_tokens == candidate_tokens,
    }
    result = {
        "schema": "rwkv7-ascend-real-model-quant-diagnostic-v1",
        "scope": "experiment-not-production",
        "environment": {
            "device": torch.npu.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "model": str(args.model.resolve()),
            "config_sha256": sha256(args.model / "config.json"),
            "index_sha256": sha256(args.model / "model.safetensors.index.json"),
        },
        "bit": args.bit,
        "group_size": 0 if args.bit == 8 else 128,
        "equalization": args.equalization,
        "layers": "all",
        "projections": list(projections),
        "replaced_projections": len(report.projections),
        "quantize_seconds": quantize_seconds,
        "dense_tensor_bytes": dense_bytes,
        "quant_tensor_bytes": quant_bytes,
        "tensor_footprint_ratio": quant_bytes / dense_bytes,
        "dense_allocated_bytes": dense_allocated,
        "quant_allocated_bytes": quant_allocated,
        "allocator_hbm_ratio": quant_allocated / dense_allocated,
        "dense_reserved_bytes": dense_reserved,
        "quant_reserved_bytes": quant_reserved,
        "timed_steps_per_round": args.timed_steps,
        "rounds": args.rounds,
        "pairs": pairs,
        "median_dense_seconds": median_dense,
        "median_quant_seconds": median_quant,
        "decode_ratio_of_medians": ratio_of_medians,
        "decode_speedup": speedup,
        "minimum_logit_cosine": min(cosine),
        "maximum_logit_nrmse": max(nrmse),
        "maximum_kl": max(kl),
        "minimum_top20_overlap": min(top20),
        "dense_greedy_tokens": dense_tokens,
        "quant_argmax_on_dense_path": candidate_tokens,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
