"""Measure Qwen3.5 through the official vLLM-Ascend runtime."""
from __future__ import annotations

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=False,
    )
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
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started

    metrics = []
    for output in outputs:
        item = output.metrics
        if item is None:
            raise RuntimeError(
                "vLLM did not return request metrics; disable_log_stats must be false"
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
    post_first_tokens = sum(max(0, item["output_tokens"] - 1) for item in metrics)
    prefill_seconds = max(last_first - first_started, 1.0e-12)
    decode_seconds = max(last_finished - last_first, 1.0e-12)
    result = {
        "benchmark": "qwen35_vllm_ascend",
        "model": os.path.abspath(args.model),
        "enforce_eager": args.enforce_eager,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "decode_length": args.decode_length,
        "wall_latency_ms": elapsed * 1000.0,
        "aggregate_output_tokens_per_second": generated / elapsed,
        "prefill_tokens_per_second": (
            args.batch_size * args.prompt_length / prefill_seconds
        ),
        "decode_tokens_per_second": post_first_tokens / decode_seconds,
        "metrics": metrics,
    }
    print(json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
