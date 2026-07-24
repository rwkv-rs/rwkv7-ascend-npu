#!/usr/bin/env python3
"""Real 7.2B HF NPUGraph acceptance for the public Ascend W8A16 API.

The benchmark keeps independently loaded dense and quantized models alive for
alternating, synchronized timing.  It exercises the public Transformers
``generate`` path at B1/B4/B8, checks fixed-token logits on the same recurrent
state trajectory, and measures isolated active HBM after each model owns all
three captured graphs.

The public speed policy is fail-closed to the exact device, stack, FP16 dtype,
7.2B FFN shapes and B1/B4/B8 logical rows exercised by this gate. A raw-op win
alone can never populate that policy.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F

from rwkv7_hf import enable_ascend, quantize_ascend_w8a16
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


HELLO_TOKEN = 33155
HELLO_GREEDY_PREFIX = [45, 308, 459]
BATCH_SIZES = (1, 4, 8)
QUALITY_CASES = (
    ("synthetic_stress", (1, 2, 3, 4)),
    ("hello", (33155,)),
    (
        "english",
        (6699, 39418, 37917, 21704, 38828, 31601, 22590, 31261, 21551, 47),
    ),
    (
        "chinese",
        (10370, 12137, 13133, 15752, 13580, 11454, 12981, 11003, 10267, 14610, 10080),
    ),
    ("python", (7334, 21676, 41943, 41, 111, 501)),
    (
        "instruction",
        (24281, 59, 28851, 6957, 60342, 46658, 56705, 47, 58683, 59),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-steps", type=int, default=8)
    parser.add_argument("--corpus-new-tokens", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-nrmse", type=float, default=0.05)
    parser.add_argument("--maximum-loss-delta", type=float, default=0.02)
    parser.add_argument(
        "--maximum-near-tie-margin",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(model: torch.nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def load_model(path: Path, num_heads: int) -> NativeRWKV7ForCausalLM:
    config = NativeRWKV7Config.from_pretrained(path)
    config.num_heads = int(num_heads)
    config.num_attention_heads = int(num_heads)
    config.head_dim = int(config.hidden_size) // int(num_heads)
    config.attention_hidden_size = int(config.hidden_size)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        path,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    return model.to("npu:0")


@torch.inference_mode()
def generate_once(
    model: NativeRWKV7ForCausalLM,
    batch_size: int,
    new_tokens: int,
) -> tuple[float, list[list[int]]]:
    input_ids = torch.full(
        (batch_size, 1),
        HELLO_TOKEN,
        dtype=torch.long,
        device="npu:0",
    )
    torch.npu.synchronize()
    started = time.perf_counter()
    generated = model.generate(
        input_ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, generated[:, 1:].detach().cpu().tolist()


@torch.inference_mode()
def generate_prompt(
    model: NativeRWKV7ForCausalLM,
    token_ids: tuple[int, ...],
    new_tokens: int,
) -> list[int]:
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device="npu:0",
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    torch.npu.synchronize()
    return generated[0, len(token_ids):].detach().cpu().tolist()


@torch.inference_mode()
def sequence_loss(
    model: NativeRWKV7ForCausalLM,
    token_ids: tuple[int, ...],
) -> float | None:
    if len(token_ids) < 2:
        return None
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device="npu:0",
    )
    result = model(
        input_ids,
        labels=input_ids,
        use_cache=False,
    )
    torch.npu.synchronize()
    return float(result.loss.detach().cpu())


@torch.inference_mode()
def forced_trace(
    model: NativeRWKV7ForCausalLM,
    input_ids: torch.Tensor,
    steps: int,
    *,
    forced: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    output = model(input_ids, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    logits: list[torch.Tensor] = []
    choices: list[int] = []
    for index in range(steps):
        current = output.logits[:, -1].float().cpu()
        logits.append(current)
        choices.append(int(torch.argmax(current, dim=-1).item()))
        token_id = choices[-1] if forced is None else int(forced[index])
        token = torch.tensor([[token_id]], dtype=torch.long, device="npu:0")
        output = model(
            token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
    torch.npu.synchronize()
    return logits, choices


def compare_logits(
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> dict[str, Any]:
    cosine: list[float] = []
    nrmse: list[float] = []
    kl: list[float] = []
    top20: list[float] = []
    mismatches: list[dict[str, Any]] = []
    for step, (expected, observed) in enumerate(zip(reference, actual)):
        cosine.append(
            float(
                F.cosine_similarity(
                    expected.flatten(),
                    observed.flatten(),
                    dim=0,
                )
            )
        )
        nrmse.append(
            float(
                (expected - observed).square().mean().sqrt()
                / expected.square().mean().sqrt().clamp_min(1e-12)
            )
        )
        probability = F.softmax(expected, dim=-1)
        kl.append(
            float(
                (
                    probability
                    * (
                        F.log_softmax(expected, dim=-1)
                        - F.log_softmax(observed, dim=-1)
                    )
                ).sum()
            )
        )
        expected_top = set(torch.topk(expected, 20).indices.flatten().tolist())
        observed_top = set(torch.topk(observed, 20).indices.flatten().tolist())
        top20.append(len(expected_top & observed_top) / 20.0)
        expected_values, expected_indices = torch.topk(
            expected.flatten(),
            2,
        )
        expected_choice = int(expected_indices[0])
        observed_choice = int(torch.argmax(observed))
        if expected_choice != observed_choice:
            reference_rank = int(
                (expected.flatten() > expected.flatten()[observed_choice])
                .sum()
                .item()
            ) + 1
            mismatches.append(
                {
                    "step": step,
                    "reference_choice": expected_choice,
                    "quant_choice": observed_choice,
                    "quant_choice_reference_rank": reference_rank,
                    "reference_top1_margin": float(
                        expected_values[0] - expected_values[1]
                    ),
                }
            )
    return {
        "minimum_logit_cosine": min(cosine),
        "maximum_logit_nrmse": max(nrmse),
        "maximum_kl_divergence": max(kl),
        "minimum_top20_overlap": min(top20),
        "per_step_cosine": cosine,
        "per_step_nrmse": nrmse,
        "argmax_mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    if (
        args.quality_steps <= 0
        or args.corpus_new_tokens <= 0
        or args.new_tokens < len(HELLO_GREEDY_PREFIX)
    ):
        raise ValueError("quality steps must be positive and new tokens >= 3")
    if args.rounds < 3:
        raise ValueError("--rounds must be at least 3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = enable_ascend("npu:0", backend="native_graph")
    torch.manual_seed(20260724)
    torch.npu.reset_peak_memory_stats()

    load_started = time.perf_counter()
    dense = load_model(args.model, args.num_heads)
    torch.npu.synchronize()
    dense_load_s = time.perf_counter() - load_started
    dense_tensor_bytes = tensor_bytes(dense)

    dense_warm_outputs: dict[int, list[list[int]]] = {}
    for batch_size in BATCH_SIZES:
        _, dense_warm_outputs[batch_size] = generate_once(
            dense,
            batch_size,
            args.new_tokens,
        )
    dense_isolated_allocated = int(torch.npu.memory_allocated())
    dense_isolated_reserved = int(torch.npu.memory_reserved())

    load_started = time.perf_counter()
    quant = load_model(args.model, args.num_heads)
    torch.npu.synchronize()
    quant_load_s = time.perf_counter() - load_started
    quantize_started = time.perf_counter()
    replaced = quantize_ascend_w8a16(
        quant,
        policy="speed",
        strict=True,
    )
    quant_report = {"policy": "speed", "replaced": replaced}
    quant.rwkv7_clear_native_graph_cache()
    torch.npu.synchronize()
    quantize_s = time.perf_counter() - quantize_started
    quant_tensor_bytes = tensor_bytes(quant)

    quant_warm_outputs: dict[int, list[list[int]]] = {}
    for batch_size in BATCH_SIZES:
        _, quant_warm_outputs[batch_size] = generate_once(
            quant,
            batch_size,
            args.new_tokens,
        )

    dense_logits: list[torch.Tensor] = []
    quant_logits: list[torch.Tensor] = []
    quality_cases: list[dict[str, Any]] = []
    for name, prompt_ids in QUALITY_CASES:
        production_corpus = name != "synthetic_stress"
        quality_input = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device="npu:0",
        )
        dense_case_logits, dense_choices = forced_trace(
            dense,
            quality_input,
            args.quality_steps,
        )
        quant_case_logits, quant_choices = forced_trace(
            quant,
            quality_input,
            args.quality_steps,
            forced=dense_choices,
        )
        dense_logits.extend(dense_case_logits)
        quant_logits.extend(quant_case_logits)
        case_quality = compare_logits(
            dense_case_logits,
            quant_case_logits,
        )
        dense_loss = sequence_loss(dense, prompt_ids)
        quant_loss = sequence_loss(quant, prompt_ids)
        dense_generated = generate_prompt(
            dense,
            prompt_ids,
            args.corpus_new_tokens,
        )
        quant_generated = generate_prompt(
            quant,
            prompt_ids,
            args.corpus_new_tokens,
        )
        quality_cases.append(
            {
                "name": name,
                "production_corpus": production_corpus,
                "prompt_token_ids": list(prompt_ids),
                "dense_loss": dense_loss,
                "quant_loss": quant_loss,
                "absolute_loss_delta": (
                    None
                    if dense_loss is None or quant_loss is None
                    else abs(quant_loss - dense_loss)
                ),
                "dense_generated_ids": dense_generated,
                "quant_generated_ids": quant_generated,
                "generated_equal": dense_generated == quant_generated,
                "dense_greedy_choices": dense_choices,
                "quant_argmax_on_dense_path": quant_choices,
                **case_quality,
            }
        )
    quality = compare_logits(dense_logits, quant_logits)
    quality["cases"] = quality_cases
    loss_deltas = [
        case["absolute_loss_delta"]
        for case in quality_cases
        if case["production_corpus"]
        and case["absolute_loss_delta"] is not None
    ]
    quality["maximum_absolute_loss_delta"] = max(loss_deltas)
    quality["all_corpus_generations_equal"] = all(
        case["generated_equal"]
        for case in quality_cases
        if case["production_corpus"]
    )
    quality["all_argmax_mismatches_are_reference_top2_near_ties"] = all(
        mismatch["quant_choice_reference_rank"] <= 2
        and mismatch["reference_top1_margin"]
        <= args.maximum_near_tie_margin
        for mismatch in quality["argmax_mismatches"]
    )

    rows: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        pairs: list[dict[str, Any]] = []
        for round_index in range(args.rounds):
            if round_index % 2 == 0:
                dense_s, dense_ids = generate_once(
                    dense,
                    batch_size,
                    args.new_tokens,
                )
                quant_s, quant_ids = generate_once(
                    quant,
                    batch_size,
                    args.new_tokens,
                )
                order = "dense-quant"
            else:
                quant_s, quant_ids = generate_once(
                    quant,
                    batch_size,
                    args.new_tokens,
                )
                dense_s, dense_ids = generate_once(
                    dense,
                    batch_size,
                    args.new_tokens,
                )
                order = "quant-dense"
            pairs.append(
                {
                    "order": order,
                    "dense_seconds": dense_s,
                    "quant_seconds": quant_s,
                    "paired_speedup": dense_s / quant_s,
                    "dense_output_tokens_per_second": (
                        batch_size * args.new_tokens / dense_s
                    ),
                    "quant_output_tokens_per_second": (
                        batch_size * args.new_tokens / quant_s
                    ),
                    "dense_ids": dense_ids[0],
                    "quant_ids": quant_ids[0],
                }
            )
        dense_median = statistics.median(
            pair["dense_seconds"] for pair in pairs
        )
        quant_median = statistics.median(
            pair["quant_seconds"] for pair in pairs
        )
        paired_speedup = statistics.median(
            pair["paired_speedup"] for pair in pairs
        )
        dense_ids = pairs[-1]["dense_ids"]
        quant_ids = pairs[-1]["quant_ids"]
        rows.append(
            {
                "batch_size": batch_size,
                "pairs": pairs,
                "dense_median_seconds": dense_median,
                "quant_median_seconds": quant_median,
                "dense_output_tokens_per_second": (
                    batch_size * args.new_tokens / dense_median
                ),
                "quant_output_tokens_per_second": (
                    batch_size * args.new_tokens / quant_median
                ),
                "median_paired_speedup": paired_speedup,
                "ratio_of_medians": dense_median / quant_median,
                "dense_exact_hello_prefix": (
                    dense_ids[: len(HELLO_GREEDY_PREFIX)]
                    == HELLO_GREEDY_PREFIX
                ),
                "quant_exact_hello_prefix": (
                    quant_ids[: len(HELLO_GREEDY_PREFIX)]
                    == HELLO_GREEDY_PREFIX
                ),
                "dense_quant_generated_equal": dense_ids == quant_ids,
                "dense_batch_outputs_identical": all(
                    ids == pairs[-1]["dense_ids"]
                    for ids in dense_warm_outputs[batch_size]
                ),
                "quant_batch_outputs_identical": all(
                    ids == pairs[-1]["quant_ids"]
                    for ids in quant_warm_outputs[batch_size]
                ),
            }
        )

    dense_graph_stats = dense.rwkv7_native_graph_cache_stats()
    dense_copy_stats = dense.rwkv7_native_graph_runner_copy_stats()
    quant_graph_stats = quant.rwkv7_native_graph_cache_stats()
    quant_copy_stats = quant.rwkv7_native_graph_runner_copy_stats()
    del dense, dense_logits
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    quant_isolated_allocated = int(torch.npu.memory_allocated())
    quant_isolated_reserved = int(torch.npu.memory_reserved())
    peak_allocated = int(torch.npu.max_memory_allocated())

    cosine_floor = args.minimum_cosine
    nrmse_ceiling = args.maximum_nrmse
    gates = {
        "exact_validated_stack": runtime.validated_stack,
        "all_graph_batches_captured": (
            dense_graph_stats["batch_sizes"] == list(BATCH_SIZES)
            and quant_graph_stats["batch_sizes"] == list(BATCH_SIZES)
        ),
        "all_64_ffn_projections_replaced": len(replaced) == 64,
        "all_dense_hello_prefixes_exact": all(
            row["dense_exact_hello_prefix"] for row in rows
        ),
        "all_quant_hello_prefixes_exact": all(
            row["quant_exact_hello_prefix"] for row in rows
        ),
        "all_dense_quant_outputs_equal": all(
            row["dense_quant_generated_equal"] for row in rows
        ),
        "all_batch_outputs_identical": all(
            row["dense_batch_outputs_identical"]
            and row["quant_batch_outputs_identical"]
            for row in rows
        ),
        "all_batches_not_slower_than_fp16": all(
            row["median_paired_speedup"] >= 1.0 for row in rows
        ),
        "tensor_footprint_reduced": quant_tensor_bytes < dense_tensor_bytes,
        "isolated_active_hbm_reduced": (
            quant_isolated_allocated < dense_isolated_allocated
        ),
        "minimum_logit_cosine": (
            quality["minimum_logit_cosine"] >= cosine_floor
        ),
        "maximum_logit_nrmse": (
            quality["maximum_logit_nrmse"] <= nrmse_ceiling
        ),
        "maximum_corpus_loss_delta": (
            quality["maximum_absolute_loss_delta"]
            <= args.maximum_loss_delta
        ),
        "all_corpus_generations_equal": (
            quality["all_corpus_generations_equal"]
        ),
        "argmax_mismatches_are_top2_near_ties": (
            quality[
                "all_argmax_mismatches_are_reference_top2_near_ties"
            ]
        ),
        "finite_positive_throughput": all(
            math.isfinite(row["quant_output_tokens_per_second"])
            and row["quant_output_tokens_per_second"] > 0
            for row in rows
        ),
    }
    result = {
        "schema": "rwkv7-ascend-hf-quant-native-graph-e2e-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "real-checkpoint-production-admission",
        "environment": {
            "device": runtime.device_name,
            "torch": torch.__version__,
            "torch_npu": runtime.torch_npu_version,
            "cann": runtime.cann_version,
            "python": platform.python_version(),
            "model": str(args.model.resolve()),
            "config_sha256": sha256(args.model / "config.json"),
            "index_sha256": sha256(
                args.model / "model.safetensors.index.json"
            ),
        },
        "backend": "transformers-generate-native_graph",
        "dtype": "float16",
        "bit": 8,
        "group_size": 0,
        "projections": ["key", "value"],
        "layers": "all",
        "replaced_projections": len(replaced),
        "new_tokens_per_request": args.new_tokens,
        "quality_steps": args.quality_steps,
        "quality_cases": len(QUALITY_CASES),
        "production_quality_cases": sum(
            name != "synthetic_stress" for name, _ in QUALITY_CASES
        ),
        "corpus_new_tokens": args.corpus_new_tokens,
        "maximum_loss_delta": args.maximum_loss_delta,
        "maximum_near_tie_margin": args.maximum_near_tie_margin,
        "rounds": args.rounds,
        "dense_load_seconds": dense_load_s,
        "quant_load_seconds": quant_load_s,
        "quantize_seconds": quantize_s,
        "dense_tensor_bytes": dense_tensor_bytes,
        "quant_tensor_bytes": quant_tensor_bytes,
        "tensor_footprint_ratio": quant_tensor_bytes / dense_tensor_bytes,
        "dense_isolated_allocated_bytes": dense_isolated_allocated,
        "quant_isolated_allocated_bytes": quant_isolated_allocated,
        "isolated_active_hbm_ratio": (
            quant_isolated_allocated / dense_isolated_allocated
        ),
        "dense_isolated_reserved_bytes": dense_isolated_reserved,
        "quant_isolated_reserved_bytes": quant_isolated_reserved,
        "peak_allocated_bytes_with_both_models": peak_allocated,
        "quality": quality,
        "rows": rows,
        "dense_graph_cache": dense_graph_stats,
        "dense_graph_state_copy": dense_copy_stats,
        "quant_graph_cache": quant_graph_stats,
        "quant_graph_state_copy": quant_copy_stats,
        "quant_report": quant_report,
        "gates": gates,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("ASCEND_QUANT_GRAPH_JSON", json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)
    print("ASCEND_QUANT_GRAPH_PASS")


if __name__ == "__main__":
    main()
