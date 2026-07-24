#!/usr/bin/env python3
"""Real vLLM V1 decode-throughput acceptance for RWKV-7 on Ascend.

Unlike the historical C++ forward microbenchmark, every measured row goes
through ``vllm.LLM.generate`` and therefore includes the real V1 scheduler,
model plugin, recurrent cache, sampler, and output collection.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

from vllm import LLM, SamplingParams


HELLO_TOKEN = 33155
HELLO_GREEDY_PREFIX = [45, 308, 459]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/models/fla-hub-rwkv7-7.2B-g0a",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument(
        "--minimum-b4-scaling",
        type=float,
        default=1.25,
        help="minimum B=4 aggregate output tok/s divided by B=1 output tok/s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.new_tokens < len(HELLO_GREEDY_PREFIX):
        raise ValueError("--new-tokens must be at least 3")

    config = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": 128,
        "block_size": 128,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.45,
        "num_gpu_blocks_override": 8,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
    }
    load_started = time.perf_counter()
    llm = LLM(**config)
    engine_load_s = time.perf_counter() - load_started
    params = SamplingParams(
        temperature=0,
        max_tokens=args.new_tokens,
        ignore_eos=True,
    )

    def execute(batch_size: int, *, measured: bool) -> dict:
        prompts = [{"prompt_token_ids": [HELLO_TOKEN]} for _ in range(batch_size)]
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        output_tokens = sum(len(ids) for ids in token_ids)
        exact_prefix = all(
            ids[: len(HELLO_GREEDY_PREFIX)] == HELLO_GREEDY_PREFIX for ids in token_ids
        )
        deterministic = all(ids == token_ids[0] for ids in token_ids)
        row = {
            "batch_size": batch_size,
            "measured": measured,
            "elapsed_s": elapsed,
            "input_tokens": batch_size,
            "output_tokens": output_tokens,
            "output_tokens_per_second": output_tokens / elapsed,
            "per_request_tokens_per_second": output_tokens / elapsed / batch_size,
            "exact_hello_prefix": exact_prefix,
            "batch_outputs_identical": deterministic,
            "first_output_token_ids": token_ids[0],
        }
        print("E2E_ROW", json.dumps(row, sort_keys=True), flush=True)
        return row

    # Compile/import and allocator effects are deliberately excluded.
    execute(1, measured=False)
    rows = [execute(batch, measured=True) for batch in (1, 4, 8)]
    by_batch = {row["batch_size"]: row for row in rows}
    scaling_b4 = (
        by_batch[4]["output_tokens_per_second"]
        / by_batch[1]["output_tokens_per_second"]
    )
    scaling_b8 = (
        by_batch[8]["output_tokens_per_second"]
        / by_batch[1]["output_tokens_per_second"]
    )
    gates = {
        "all_rows_finite_positive": all(
            math.isfinite(row["output_tokens_per_second"])
            and row["output_tokens_per_second"] > 0
            for row in rows
        ),
        "hello_greedy_exact": all(row["exact_hello_prefix"] for row in rows),
        "batch_outputs_identical": all(row["batch_outputs_identical"] for row in rows),
        "b4_dynamic_batch_scaling": scaling_b4 >= args.minimum_b4_scaling,
        "b8_not_slower_than_b4": (
            by_batch[8]["output_tokens_per_second"]
            >= by_batch[4]["output_tokens_per_second"]
        ),
    }
    record = {
        "axis": "vllm_ascend_real_7p2b_e2e_performance",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model": args.model,
        "hardware": "Ascend910B3 64GB",
        "runtime": {
            "vllm": importlib.metadata.version("vllm"),
            "vllm_ascend": importlib.metadata.version("vllm-ascend"),
            "plugin": importlib.metadata.version("rwkv7-vllm-ascend"),
        },
        "engine_load_s": engine_load_s,
        "new_tokens_per_request": args.new_tokens,
        "minimum_b4_scaling": args.minimum_b4_scaling,
        "scaling_b4_over_b1": scaling_b4,
        "scaling_b8_over_b1": scaling_b8,
        "rows": rows,
        "gates": gates,
        "config": {key: value for key, value in config.items() if key != "model"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("E2E_JSON", json.dumps(record, sort_keys=True), flush=True)
    if record["status"] != "PASS":
        raise SystemExit(1)
    print("E2E_PASS", flush=True)


if __name__ == "__main__":
    main()
