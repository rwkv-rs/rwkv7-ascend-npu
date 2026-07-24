#!/usr/bin/env python3
"""Real Transformers ``generate`` throughput acceptance on Ascend 910B3."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from rwkv7_hf import enable_ascend
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


HELLO_TOKEN = 33155
HELLO_GREEDY_PREFIX = [45, 308, 459]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--minimum-b4-scaling", type=float, default=1.25)
    parser.add_argument(
        "--backend",
        choices=("eager", "native_graph"),
        default="eager",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=64,
        help="validated 7.2B repair; source config incorrectly declares 32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.new_tokens < len(HELLO_GREEDY_PREFIX):
        raise ValueError("--new-tokens must be at least 3")

    runtime = enable_ascend("npu:0", backend=args.backend)
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    torch.npu.reset_peak_memory_stats()
    load_started = time.perf_counter()
    # Load the local FLA-free adapter class directly.  The source checkpoint's
    # AutoMap points at FLA, which is intentionally not a runtime dependency of
    # the Huawei package.
    model_config = NativeRWKV7Config.from_pretrained(args.model)
    model_config.num_heads = args.num_heads
    model_config.num_attention_heads = args.num_heads
    model_config.head_dim = int(model_config.hidden_size) // args.num_heads
    model_config.attention_hidden_size = int(model_config.hidden_size)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).eval()
    load_cpu_s = time.perf_counter() - load_started
    transfer_started = time.perf_counter()
    model.to("npu:0")
    torch.npu.synchronize()
    transfer_s = time.perf_counter() - transfer_started

    def execute(batch_size: int, *, measured: bool) -> dict:
        input_ids = torch.full(
            (batch_size, 1),
            HELLO_TOKEN,
            dtype=torch.long,
            device="npu:0",
        )
        torch.npu.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=None,
            )
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started
        generated_ids = generated[:, 1:].detach().cpu().tolist()
        output_tokens = sum(len(ids) for ids in generated_ids)
        row = {
            "batch_size": batch_size,
            "measured": measured,
            "elapsed_s": elapsed,
            "input_tokens": batch_size,
            "output_tokens": output_tokens,
            "output_tokens_per_second": output_tokens / elapsed,
            "per_request_tokens_per_second": output_tokens / elapsed / batch_size,
            "exact_hello_prefix": all(
                ids[: len(HELLO_GREEDY_PREFIX)] == HELLO_GREEDY_PREFIX
                for ids in generated_ids
            ),
            "batch_outputs_identical": all(
                ids == generated_ids[0] for ids in generated_ids
            ),
            "first_output_token_ids": generated_ids[0],
            "npu_allocated_bytes": torch.npu.memory_allocated(),
            "npu_peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "last_decode_backend": getattr(
                model,
                "_rwkv7_native_model_last_decode_backend",
                None,
            ),
        }
        print("E2E_ROW", json.dumps(row, sort_keys=True), flush=True)
        return row

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
    graph_cache = (
        model.rwkv7_native_graph_cache_stats()
        if args.backend == "native_graph"
        else None
    )
    graph_state_copy = (
        model.rwkv7_native_graph_runner_copy_stats()
        if args.backend == "native_graph"
        else None
    )
    if args.backend == "native_graph":
        gates["native_graph_used"] = all(
            row["last_decode_backend"] == "native_graph" for row in rows
        )
        gates["native_graph_batch_cache"] = (
            graph_cache is not None
            and graph_cache["batch_sizes"] == [1, 4, 8]
        )
    record = {
        "axis": "hf_ascend_real_7p2b_e2e_performance",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model": args.model,
        "hardware": runtime.device_name,
        "cann": runtime.cann_version,
        "dtype": args.dtype,
        "backend": f"transformers-generate-{args.backend}",
        "config_repair": {
            "num_heads": args.num_heads,
            "head_dim": model_config.head_dim,
            "attention_hidden_size": model_config.attention_hidden_size,
        },
        "load_cpu_s": load_cpu_s,
        "transfer_to_npu_s": transfer_s,
        "new_tokens_per_request": args.new_tokens,
        "minimum_b4_scaling": args.minimum_b4_scaling,
        "scaling_b4_over_b1": scaling_b4,
        "scaling_b8_over_b1": scaling_b8,
        "rows": rows,
        "graph_cache": graph_cache,
        "graph_state_copy": graph_state_copy,
        "gates": gates,
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
