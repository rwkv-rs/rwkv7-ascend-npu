#!/usr/bin/env python3
"""Real SGLang Engine decode-throughput acceptance on Ascend 910B3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time


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
    parser.add_argument("--minimum-b4-scaling", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.new_tokens < len(HELLO_GREEDY_PREFIX):
        raise ValueError("--new-tokens must be at least 3")

    from sglang_rwkv7_ascend import register

    register()
    from sglang.srt.entrypoints.engine import Engine

    source_model = Path(args.model).resolve()
    model_view = Path(__file__).resolve().parents[1] / ".e2e-model-view"
    shutil.rmtree(model_view, ignore_errors=True)
    model_view.mkdir(parents=True)
    source_config = json.loads((source_model / "config.json").read_text())
    source_config.pop("auto_map", None)
    (model_view / "config.json").write_text(
        json.dumps(source_config, indent=2) + "\n",
        encoding="utf-8",
    )
    for item in source_model.iterdir():
        if item.name != "config.json":
            (model_view / item.name).symlink_to(item)

    config = {
        "model_path": str(model_view),
        "device": "npu",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "attention_backend": "ascend",
        "max_running_requests": 8,
        "max_mamba_cache_size": 16,
        "disable_radix_cache": True,
        "chunked_prefill_size": 64,
        "enable_mixed_chunk": True,
        "page_size": 1,
        "disable_cuda_graph": True,
        "log_level": "warning",
    }
    load_started = time.perf_counter()
    engine = Engine(**config)
    engine_load_s = time.perf_counter() - load_started
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.new_tokens,
        "ignore_eos": True,
    }

    def execute(batch_size: int, *, measured: bool) -> dict:
        started = time.perf_counter()
        responses = engine.generate(
            input_ids=[[HELLO_TOKEN] for _ in range(batch_size)],
            sampling_params=sampling,
        )
        elapsed = time.perf_counter() - started
        if isinstance(responses, dict):
            responses = [responses]
        output_ids = [list(response.get("output_ids") or []) for response in responses]
        output_tokens = sum(len(ids) for ids in output_ids)
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
                for ids in output_ids
            ),
            "batch_outputs_identical": all(ids == output_ids[0] for ids in output_ids),
            "first_output_token_ids": output_ids[0],
        }
        print("E2E_ROW", json.dumps(row, sort_keys=True), flush=True)
        return row

    try:
        execute(1, measured=False)
        rows = [execute(batch, measured=True) for batch in (1, 4, 8)]
    finally:
        engine.shutdown()

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
        "axis": "sglang_ascend_real_7p2b_e2e_performance",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model": args.model,
        "hardware": "Ascend910B3 64GB",
        "dtype": "bfloat16",
        "backend": "sglang-engine-eager",
        "engine_load_s": engine_load_s,
        "new_tokens_per_request": args.new_tokens,
        "minimum_b4_scaling": args.minimum_b4_scaling,
        "scaling_b4_over_b1": scaling_b4,
        "scaling_b8_over_b1": scaling_b8,
        "rows": rows,
        "gates": gates,
        "config": config,
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
