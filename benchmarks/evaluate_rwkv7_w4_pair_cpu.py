#!/usr/bin/env python3
"""Offline quality study on real RWKV-7 key/value tensors (not production E2E).

The script deliberately labels synthetic hidden rows as such.  It is useful for
choosing group size/CLE candidates before scarce NPU time, but its output cannot
satisfy the real-prompt greedy/logit production gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from rwkv7_ascend_model_quant import (
    apply_rwkv7_sqrelu_equalization,
    compute_rwkv7_sqrelu_equalization_scale,
)
from rwkv7_ascend_quant import AscendWeightOnlyLinear


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((candidate.float() - reference.float()).norm() / reference.float().norm())


def tensor_from_index(model: Path, name: str) -> tuple[torch.Tensor, Path]:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard = model / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name), shard


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    key_name = f"model.layers.{args.layer}.ffn.key.weight"
    value_name = f"model.layers.{args.layer}.ffn.value.weight"
    key_weight, key_shard = tensor_from_index(args.model, key_name)
    value_weight, value_shard = tensor_from_index(args.model, value_name)
    if key_weight.shape[0] != value_weight.shape[1]:
        raise ValueError("checkpoint key/value shapes are not an RWKV FFN pair")
    hidden = key_weight.shape[1]
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows, hidden, generator=generator, dtype=torch.float16)
    key_fp = torch.nn.functional.linear(x.float(), key_weight.float())
    activation_fp = torch.relu(key_fp).square()
    output_fp = torch.nn.functional.linear(activation_fp, value_weight.float())
    provenance = {
        "kind": "environment",
        "scope": "offline-synthetic-pair-study-not-production",
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "rows": args.rows,
        "seed": args.seed,
        "torch": torch.__version__,
        "key_name": key_name,
        "value_name": value_name,
        "key_shard": key_shard.name,
        "value_shard": value_shard.name,
        "key_shard_sha256": file_sha256(key_shard),
        "value_shard_sha256": file_sha256(value_shard),
        "index_sha256": file_sha256(args.model / "model.safetensors.index.json"),
        "config_sha256": file_sha256(args.model / "config.json"),
    }
    print(json.dumps(provenance, sort_keys=True), flush=True)
    methods = [
        (8, 128, "none"),
        (4, 128, "none"),
        (4, 128, "weight-cle"),
        (4, 64, "none"),
        (4, 32, "none"),
    ]
    fp_bytes = (key_weight.numel() + value_weight.numel()) * 2
    for bit, group_size, equalization in methods:
        started = time.perf_counter()
        key_for_quant = key_weight
        value_for_quant = value_weight
        scale = None
        if equalization != "none":
            scale = compute_rwkv7_sqrelu_equalization_scale(
                value_weight,
                group_size=group_size,
                mode="weight-cle",
                scale_min=0.25,
                scale_max=4.0,
            )
            key_for_quant, value_for_quant = apply_rwkv7_sqrelu_equalization(
                key_weight, value_weight, scale
            )
        qkey = AscendWeightOnlyLinear(
            hidden,
            key_weight.shape[0],
            bit=bit,
            group_size=group_size,
            enforce_verified_shape=False,
        ).load_fp_weight(key_for_quant)
        qvalue = AscendWeightOnlyLinear(
            value_weight.shape[1],
            value_weight.shape[0],
            bit=bit,
            group_size=group_size,
            enforce_verified_shape=False,
        ).load_fp_weight(value_for_quant)
        key_quant = qkey(x).float()
        activation_quant = torch.relu(key_quant).square().to(torch.float16)
        output_quant = qvalue(activation_quant).float()
        packed_bytes = qkey.packed_weight_bytes() + qvalue.packed_weight_bytes()
        print(
            json.dumps(
                {
                    "kind": "pair_result",
                    "bit": bit,
                    "group_size": 0 if bit == 8 else group_size,
                    "equalization": equalization,
                    "key_cosine": cosine(key_fp, key_quant),
                    "ffn_cosine": cosine(output_fp, output_quant),
                    "ffn_relative_l2": relative_l2(output_fp, output_quant),
                    "fp16_weight_bytes": fp_bytes,
                    "packed_bytes": packed_bytes,
                    "packed_ratio": packed_bytes / fp_bytes,
                    "equalization_min": None if scale is None else float(scale.min()),
                    "equalization_max": None if scale is None else float(scale.max()),
                    "elapsed_s": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del qkey, qvalue, key_quant, activation_quant, output_quant


if __name__ == "__main__":
    main()
