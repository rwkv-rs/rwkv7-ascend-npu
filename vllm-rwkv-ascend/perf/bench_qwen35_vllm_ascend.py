"""Measure Qwen3.5 through the official vLLM-Ascend runtime."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401 - registers torch.npu
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from benchmark_metadata import (
    checkpoint_metadata,
    collect_cann_metadata,
    collect_npu_metadata,
)
from npu_memory import PeakNPUMemorySampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("tensor-parallel-size must be positive")
    if args.warmup_iterations < 0 or args.iterations < 1:
        parser.error("warmup-iterations must be non-negative and iterations positive")
    visible_devices = torch.npu.device_count()
    if args.tensor_parallel_size > visible_devices:
        parser.error(
            f"tensor-parallel-size {args.tensor_parallel_size} exceeds "
            f"{visible_devices} visible NPUs"
        )

    device_ids = range(args.tensor_parallel_size)
    load_memory_sampler = PeakNPUMemorySampler(device_ids).start()
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=False,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    load_peak_memory_mib = load_memory_sampler.stop()
    workload_memory_sampler = PeakNPUMemorySampler(device_ids).start()
    warmup = TokensPrompt(prompt_token_ids=list(range(8)))
    llm.generate(
        [warmup],
        SamplingParams(temperature=0.0, max_tokens=2),
        use_tqdm=False,
    )

    prompts = [
        TokensPrompt(
            prompt_token_ids=[
                (token + request * 997) % 65536
                for token in range(args.prompt_length)
            ]
        )
        for request in range(args.batch_size)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_length,
        ignore_eos=True,
    )
    warmup_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
    )
    for _ in range(args.warmup_iterations):
        llm.generate(prompts, warmup_sampling, use_tqdm=False)

    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        metrics = []
        for output in outputs:
            item = output.metrics
            if item is None:
                raise RuntimeError(
                    "vLLM did not return request metrics; "
                    "disable_log_stats must be false"
                )
            metrics.append(
                {
                    "arrival_time": item.arrival_time,
                    "scheduled_time": item.scheduled_ts,
                    "first_token_time": item.first_token_ts,
                    "finished_time": item.last_token_ts,
                    "time_to_first_token_ms": item.first_token_latency * 1000.0,
                    "decode_time_ms": (
                        item.last_token_ts - item.first_token_ts
                    ) * 1000.0,
                    "output_tokens": len(output.outputs[0].token_ids),
                }
            )
        first_started = min(item["scheduled_time"] for item in metrics)
        last_first = max(item["first_token_time"] for item in metrics)
        last_finished = max(item["finished_time"] for item in metrics)
        generated = sum(item["output_tokens"] for item in metrics)
        post_first_tokens = sum(
            max(0, item["output_tokens"] - 1) for item in metrics
        )
        samples.append(
            {
                "wall_latency_ms": elapsed * 1000.0,
                "prefill_latency_ms": max(
                    (last_first - first_started) * 1000.0, 1.0e-9
                ),
                "decode_latency_ms": max(
                    (last_finished - last_first) * 1000.0, 1.0e-9
                ),
                "generated_tokens": generated,
                "post_first_tokens": post_first_tokens,
                "metrics": metrics,
            }
        )
    peak_memory_mib = workload_memory_sampler.stop()
    wall_latency_ms = statistics.median(
        sample["wall_latency_ms"] for sample in samples
    )
    prefill_latency_ms = statistics.median(
        sample["prefill_latency_ms"] for sample in samples
    )
    decode_latency_ms = statistics.median(
        sample["decode_latency_ms"] for sample in samples
    )
    generated = samples[0]["generated_tokens"]
    post_first_tokens = samples[0]["post_first_tokens"]
    result = {
        "benchmark": "qwen35_vllm_ascend",
        "dtype": "fp16",
        "engine_version": vllm.__version__,
        "load_peak_memory_mib": load_peak_memory_mib,
        "peak_memory_mib": peak_memory_mib,
        "peak_memory_scope": workload_memory_sampler.scope,
        "peak_memory_phase": "prefill_decode",
        "memory_sampler_errors": workload_memory_sampler.errors,
        **checkpoint_metadata(args.model, revision=args.checkpoint_revision),
        **collect_cann_metadata(),
        **collect_npu_metadata(
            torch,
            torch_npu,
            "npu:0",
            device_count=args.tensor_parallel_size,
        ),
        "enforce_eager": args.enforce_eager,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "decode_length": args.decode_length,
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "wall_latency_ms": wall_latency_ms,
        "wall_latency_ms_samples": [
            sample["wall_latency_ms"] for sample in samples
        ],
        "aggregate_output_tokens_per_second": generated / (wall_latency_ms / 1000.0),
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_latency_ms_samples": [
            sample["prefill_latency_ms"] for sample in samples
        ],
        "prefill_tokens_per_second": (
            args.batch_size * args.prompt_length / (prefill_latency_ms / 1000.0)
        ),
        "decode_latency_ms": decode_latency_ms,
        "decode_latency_ms_samples": [
            sample["decode_latency_ms"] for sample in samples
        ],
        "decode_tokens_per_second": (
            post_first_tokens / (decode_latency_ms / 1000.0)
        ),
        "metrics": samples[len(samples) // 2]["metrics"],
    }
    print(json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
