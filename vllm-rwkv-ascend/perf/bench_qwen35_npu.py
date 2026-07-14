"""Benchmark a local Qwen3.5 checkpoint on Ascend through Transformers.

This is the reproducible framework baseline for the RWKV-7 comparison.  It
uses random, shape-controlled token IDs so tokenizer and prompt formatting do
not influence prefill/decode throughput.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers torch.npu
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _time_prefill(
    model,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, list[float]]:
    samples = []
    with torch.no_grad():
        for index in range(warmup + iterations):
            torch.npu.synchronize()
            started = time.perf_counter()
            output = model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
            )
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= warmup:
                samples.append(elapsed_ms)
            del output
    return statistics.median(samples), samples


def _time_decode(
    model,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    steps: int,
) -> tuple[float, list[float]]:
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        del output

        for _ in range(warmup):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)

        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
        torch.npu.synchronize()
        mean_ms = (time.perf_counter() - started) * 1000.0 / steps

        # Collect device-side latency percentiles separately so host-side
        # synchronization is not charged to the primary throughput row.
        event_pairs = []
        for _ in range(min(32, steps)):
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start_event.record()
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            end_event.record()
            event_pairs.append((start_event, end_event))
        torch.npu.synchronize()
        samples = [start.elapsed_time(end) for start, end in event_pairs]
    return mean_ms, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-warmup", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--prefill-warmup", type=int, default=1)
    parser.add_argument("--prefill-iterations", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(value < 1 for value in batch_sizes):
        parser.error("batch sizes must be positive")

    torch.manual_seed(20260714)
    torch.npu.set_device(args.device)
    torch.npu.reset_peak_memory_stats(args.device)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
    ).eval().to(args.device)
    torch.npu.synchronize()

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    vocab_size = model.config.text_config.vocab_size
    loaded_memory = torch.npu.memory_allocated(args.device)
    rows = []

    for batch_size in batch_sizes:
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, args.prompt_length),
            device=args.device,
        )
        prefill_ms, prefill_samples = _time_prefill(
            model,
            input_ids,
            warmup=args.prefill_warmup,
            iterations=args.prefill_iterations,
        )
        decode_ms, decode_samples = _time_decode(
            model,
            input_ids,
            warmup=args.decode_warmup,
            steps=args.decode_steps,
        )
        peak_memory = torch.npu.max_memory_allocated(args.device)
        row = {
            "batch_size": batch_size,
            "prompt_length": args.prompt_length,
            "prefill_latency_ms_median": prefill_ms,
            "prefill_latency_ms_p90": _percentile(prefill_samples, 0.90),
            "prefill_tokens_per_second": (
                batch_size * args.prompt_length * 1000.0 / prefill_ms
            ),
            "decode_latency_ms_mean": decode_ms,
            "decode_latency_ms_p50": _percentile(decode_samples, 0.50),
            "decode_latency_ms_p90": _percentile(decode_samples, 0.90),
            "decode_tokens_per_second": batch_size * 1000.0 / decode_ms,
            "peak_memory_mib": peak_memory / 2**20,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del input_ids
        gc.collect()
        torch.npu.empty_cache()

    result = {
        "benchmark": "qwen35_transformers_npu",
        "model": os.path.abspath(args.model),
        "architecture": type(model).__name__,
        "dtype": args.dtype,
        "device": args.device,
        "model_parameters": model_parameters,
        "model_bytes": model_bytes,
        "loaded_memory_mib": loaded_memory / 2**20,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
