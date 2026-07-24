#!/usr/bin/env python3
"""Real-engine RWKV-7 serving churn and HBM-stability gate on Ascend.

The default is a 30-minute measured soak after allocator warm-up.  Requests
cycle through B1/B4/B8, must reproduce one complete greedy token sequence, and
must reuse recurrent-state slots without stale-state output.  HBM is sampled
from ``npu-smi`` so vLLM's engine subprocess is included.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


HELLO_TOKEN = 33155
HELLO_GREEDY_PREFIX = [45, 308, 459]
BATCH_PATTERN = (1, 4, 8, 4)
HBM_PAIR_RE = re.compile(r"(?P<used>\d+)\s*/\s*(?P<capacity>\d+)")


class SoakError(RuntimeError):
    """The runtime cannot execute or prove one required soak invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SoakError(message)


def parse_npu_smi_hbm(text: str) -> tuple[int, int]:
    """Return the used/capacity MB pair with the largest nonzero capacity."""

    pairs = [
        (int(match.group("used")), int(match.group("capacity")))
        for match in HBM_PAIR_RE.finditer(text)
        if int(match.group("capacity")) > 0
    ]
    if not pairs:
        raise SoakError("npu-smi output has no HBM used/capacity pair")
    used, capacity = max(pairs, key=lambda pair: pair[1])
    _require(0 <= used <= capacity, "npu-smi reported invalid HBM usage")
    return used, capacity


def sample_hbm_mb() -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SoakError("cannot sample HBM through npu-smi") from exc
    return parse_npu_smi_hbm(result.stdout)


def _linear_slope_per_hour(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        return math.inf
    xs = [float(sample["elapsed_s"]) for sample in samples]
    ys = [float(sample["hbm_used_mb"]) for sample in samples]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return math.inf
    slope_per_second = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    )
    return slope_per_second * 3600


def analyze_hbm(
    samples: list[dict[str, Any]],
    *,
    max_growth_mb: float,
    max_slope_mb_per_hour: float,
) -> dict[str, Any]:
    _require(len(samples) >= 2, "at least two measured HBM samples are required")
    window = min(8, max(1, len(samples) // 4))
    head = statistics.median(
        float(sample["hbm_used_mb"]) for sample in samples[:window]
    )
    tail = statistics.median(
        float(sample["hbm_used_mb"]) for sample in samples[-window:]
    )
    growth = tail - head
    slope = _linear_slope_per_hour(samples)
    return {
        "sample_count": len(samples),
        "head_window": window,
        "head_median_mb": head,
        "tail_median_mb": tail,
        "tail_growth_mb": growth,
        "minimum_mb": min(sample["hbm_used_mb"] for sample in samples),
        "maximum_mb": max(sample["hbm_used_mb"] for sample in samples),
        "linear_slope_mb_per_hour": slope,
        "max_growth_mb": max_growth_mb,
        "max_slope_mb_per_hour": max_slope_mb_per_hour,
        "growth_gate": growth <= max_growth_mb,
        "slope_gate": slope <= max_slope_mb_per_hour,
    }


def analyze_throughput(
    rows: list[dict[str, Any]], *, minimum_tail_ratio: float
) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["batch_size"])].append(row)
    result: dict[str, Any] = {}
    for batch in sorted(groups):
        batch_rows = groups[batch]
        _require(len(batch_rows) >= 2, f"B{batch} needs at least two soak samples")
        window = min(8, max(1, len(batch_rows) // 4))
        head = statistics.median(
            float(row["output_tokens_per_second"]) for row in batch_rows[:window]
        )
        tail = statistics.median(
            float(row["output_tokens_per_second"]) for row in batch_rows[-window:]
        )
        ratio = tail / head
        result[str(batch)] = {
            "sample_count": len(batch_rows),
            "head_median_output_tokens_per_second": head,
            "tail_median_output_tokens_per_second": tail,
            "tail_over_head": ratio,
            "minimum_tail_ratio": minimum_tail_ratio,
            "gate": ratio >= minimum_tail_ratio,
        }
    return result


def summarize_trace(path: Path, backend: str) -> dict[str, Any]:
    try:
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SoakError(f"cannot read valid scheduler trace: {path}") from exc

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") == "soak_marker":
            segments.append(current)
            current = []
        else:
            current.append(event)
    if current:
        segments.append(current)
    _require(bool(segments), "scheduler trace contains no completed soak segment")

    if backend == "vllm":
        zero_events = [
            event
            for segment in segments
            for event in segment
            if event.get("event") == "fresh_state_zero"
        ]
        reused = [
            event for event in zero_events if event.get("pre_zero_had_nonzero") is True
        ]
        failures = [
            event
            for event in zero_events
            if event.get("post_zero_nonzero") is not False
        ]
        return {
            "segment_count": len(segments),
            "fresh_zero_events": len(zero_events),
            "reused_nonzero_slots_before_clear": len(reused),
            "reused_slot_ids": sorted({int(event["slot"]) for event in reused}),
            "post_zero_failures": len(failures),
            "state_reuse_gate": bool(reused) and not failures,
        }

    _require(backend == "sglang", f"unsupported trace backend: {backend}")
    fresh_by_segment: list[set[int]] = []
    for segment in segments:
        fresh: set[int] = set()
        for event in segment:
            if event.get("kind") != "forward":
                continue
            slots = event.get("state_slot_ids") or []
            prefixes = event.get("extend_prefix_lens")
            if prefixes is None:
                continue
            for slot, prefix in zip(slots, prefixes):
                if int(slot) >= 0 and int(prefix) == 0:
                    fresh.add(int(slot))
        fresh_by_segment.append(fresh)
    appearances: dict[int, int] = defaultdict(int)
    for slots in fresh_by_segment:
        for slot in slots:
            appearances[slot] += 1
    reused_slots = sorted(slot for slot, count in appearances.items() if count >= 2)
    return {
        "segment_count": len(segments),
        "fresh_slot_assignments": sum(len(slots) for slots in fresh_by_segment),
        "unique_fresh_slots": sorted(appearances),
        "reused_slot_ids": reused_slots,
        "state_reuse_gate": bool(reused_slots),
    }


def _append_marker(path: Path, *, phase: str, cycle: int, batch_size: int) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "soak_marker",
                    "phase": phase,
                    "cycle": cycle,
                    "batch_size": batch_size,
                },
                separators=(",", ":"),
            )
            + "\n"
        )


def _make_model_view(source: Path, destination: Path) -> Path:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.pop("auto_map", None)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for item in source.iterdir():
        if item.name != "config.json":
            (destination / item.name).symlink_to(item)
    return destination


def create_vllm_engine(
    model: str, new_tokens: int
) -> tuple[Callable[[int], list[list[int]]], Callable[[], None], dict[str, Any]]:
    from vllm import LLM, SamplingParams

    config = {
        "model": model,
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
    llm = LLM(**config)
    sampling = SamplingParams(temperature=0, max_tokens=new_tokens, ignore_eos=True)

    def execute(batch_size: int) -> list[list[int]]:
        prompts = [{"prompt_token_ids": [HELLO_TOKEN]} for _ in range(batch_size)]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def shutdown() -> None:
        llm.llm_engine.engine_core.shutdown(timeout=120)

    runtime = {
        "vllm": importlib.metadata.version("vllm"),
        "vllm_ascend": importlib.metadata.version("vllm-ascend"),
        "plugin": importlib.metadata.version("rwkv7-vllm-ascend"),
        "config": {key: value for key, value in config.items() if key != "model"},
    }
    return execute, shutdown, runtime


def create_sglang_engine(
    model: str, new_tokens: int
) -> tuple[Callable[[int], list[list[int]]], Callable[[], None], dict[str, Any]]:
    from sglang_rwkv7_ascend import register

    register()
    from sglang.srt.entrypoints.engine import Engine

    source = Path(model).resolve()
    model_view = _make_model_view(
        source, Path(__file__).resolve().parents[1] / ".soak-model-view"
    )
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
    engine = Engine(**config)
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": new_tokens,
        "ignore_eos": True,
    }

    def execute(batch_size: int) -> list[list[int]]:
        responses = engine.generate(
            input_ids=[[HELLO_TOKEN] for _ in range(batch_size)],
            sampling_params=sampling,
        )
        if isinstance(responses, dict):
            responses = [responses]
        return [list(response.get("output_ids") or []) for response in responses]

    return execute, engine.shutdown, {"config": config}


def _wait_for_reclaim(
    initial_idle_mb: int, *, tolerance_mb: int, timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    samples = []
    while True:
        used, capacity = sample_hbm_mb()
        samples.append(used)
        if used <= initial_idle_mb + tolerance_mb or time.monotonic() >= deadline:
            return {
                "initial_idle_hbm_mb": initial_idle_mb,
                "post_shutdown_hbm_mb": used,
                "capacity_mb": capacity,
                "tolerance_mb": tolerance_mb,
                "poll_samples_mb": samples,
                "gate": used <= initial_idle_mb + tolerance_mb,
            }
        time.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", default="/data/models/fla-hub-rwkv7-7.2B-g0a")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=1800)
    parser.add_argument("--minimum-cycles", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--max-hbm-growth-mb", type=float, default=256)
    parser.add_argument("--max-hbm-slope-mb-per-hour", type=float, default=128)
    parser.add_argument("--minimum-throughput-tail-ratio", type=float, default=0.8)
    parser.add_argument("--reclaim-tolerance-mb", type=int, default=256)
    parser.add_argument("--reclaim-timeout-seconds", type=float, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require(args.duration_seconds > 0, "--duration-seconds must be positive")
    _require(args.minimum_cycles >= 8, "--minimum-cycles must be at least 8")
    _require(args.warmup_cycles >= 1, "--warmup-cycles must be positive")
    _require(
        args.new_tokens >= len(HELLO_GREEDY_PREFIX),
        "--new-tokens must be at least 3",
    )
    output = args.output.resolve()
    trace = (
        args.trace.resolve()
        if args.trace
        else output.with_name(output.stem + ".trace.jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("", encoding="utf-8")
    trace_env = (
        "RWKV7_TRACE_PATH" if args.backend == "vllm" else "RWKV_SGLANG_ACCEPTANCE_TRACE"
    )
    os.environ[trace_env] = str(trace)

    initial_idle_hbm, hbm_capacity = sample_hbm_mb()
    started_at = time.time()
    engine_started = time.perf_counter()
    execute = None
    shutdown = None
    runtime: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    canonical: list[int] | None = None
    error: dict[str, str] | None = None
    engine_load_s = None
    measured_duration_s = 0.0
    reclaim: dict[str, Any] = {"gate": False}

    try:
        factory = create_vllm_engine if args.backend == "vllm" else create_sglang_engine
        execute, shutdown, runtime = factory(args.model, args.new_tokens)
        engine_load_s = time.perf_counter() - engine_started

        for cycle in range(args.warmup_cycles):
            batch = BATCH_PATTERN[cycle % len(BATCH_PATTERN)]
            token_ids = execute(batch)
            _append_marker(trace, phase="warmup", cycle=cycle, batch_size=batch)
            _require(len(token_ids) == batch, "warm-up response count mismatch")
            if canonical is None:
                canonical = token_ids[0]
                _require(
                    canonical[: len(HELLO_GREEDY_PREFIX)] == HELLO_GREEDY_PREFIX,
                    "warm-up greedy oracle failed",
                )
                _require(
                    len(canonical) == args.new_tokens,
                    "warm-up generation length mismatch",
                )
            _require(
                all(ids == canonical for ids in token_ids),
                "warm-up output is nondeterministic",
            )

        measured_started = time.perf_counter()
        cycle = 0
        while (
            time.perf_counter() - measured_started < args.duration_seconds
            or cycle < args.minimum_cycles
        ):
            batch = BATCH_PATTERN[cycle % len(BATCH_PATTERN)]
            cycle_started = time.perf_counter()
            token_ids = execute(batch)
            elapsed = time.perf_counter() - cycle_started
            _append_marker(trace, phase="measured", cycle=cycle, batch_size=batch)
            used_hbm, capacity = sample_hbm_mb()
            _require(capacity == hbm_capacity, "HBM capacity changed during soak")
            exact = len(token_ids) == batch and all(
                ids == canonical for ids in token_ids
            )
            output_tokens = sum(len(ids) for ids in token_ids)
            row = {
                "cycle": cycle,
                "batch_size": batch,
                "elapsed_s": time.perf_counter() - measured_started,
                "cycle_elapsed_s": elapsed,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / elapsed,
                "exact_canonical_output": exact,
                "hbm_used_mb": used_hbm,
            }
            rows.append(row)
            print("SOAK_ROW", json.dumps(row, sort_keys=True), flush=True)
            _require(exact, f"canonical output mismatch at cycle {cycle}")
            cycle += 1
        measured_duration_s = time.perf_counter() - measured_started
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if shutdown is not None:
            try:
                shutdown()
            except BaseException as exc:
                if error is None:
                    error = {
                        "type": type(exc).__name__,
                        "message": f"engine shutdown failed: {exc}",
                    }
        try:
            reclaim = _wait_for_reclaim(
                initial_idle_hbm,
                tolerance_mb=args.reclaim_tolerance_mb,
                timeout_s=args.reclaim_timeout_seconds,
            )
        except BaseException as exc:
            if error is None:
                error = {
                    "type": type(exc).__name__,
                    "message": f"HBM reclaim check failed: {exc}",
                }

    hbm_analysis: dict[str, Any] = {"growth_gate": False, "slope_gate": False}
    throughput: dict[str, Any] = {}
    trace_summary: dict[str, Any] = {"state_reuse_gate": False}
    if len(rows) >= 2:
        try:
            hbm_analysis = analyze_hbm(
                rows,
                max_growth_mb=args.max_hbm_growth_mb,
                max_slope_mb_per_hour=args.max_hbm_slope_mb_per_hour,
            )
            throughput = analyze_throughput(
                rows,
                minimum_tail_ratio=args.minimum_throughput_tail_ratio,
            )
            trace_summary = summarize_trace(trace, args.backend)
        except BaseException as exc:
            if error is None:
                error = {"type": type(exc).__name__, "message": str(exc)}

    gates = {
        "no_runtime_error": error is None,
        "duration_reached": measured_duration_s >= args.duration_seconds,
        "minimum_cycles_reached": len(rows) >= args.minimum_cycles,
        "all_outputs_exact": bool(rows)
        and all(row["exact_canonical_output"] for row in rows),
        "all_rows_finite_positive": bool(rows)
        and all(
            math.isfinite(row["output_tokens_per_second"])
            and row["output_tokens_per_second"] > 0
            for row in rows
        ),
        "throughput_tail_stable": bool(throughput)
        and all(value["gate"] for value in throughput.values()),
        "hbm_tail_growth_bounded": hbm_analysis.get("growth_gate") is True,
        "hbm_linear_slope_bounded": hbm_analysis.get("slope_gate") is True,
        "recurrent_state_slots_reused": trace_summary.get("state_reuse_gate") is True,
        "post_shutdown_hbm_reclaimed": reclaim.get("gate") is True,
    }
    record = {
        "schema": "rwkv7-ascend-serving-soak-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "backend": args.backend,
        "model": str(Path(args.model).resolve()),
        "hardware": "Ascend910B3 64GB",
        "started_at_unix": started_at,
        "engine_load_s": engine_load_s,
        "measured_duration_s": measured_duration_s,
        "cycle_count": len(rows),
        "request_count": sum(row["batch_size"] for row in rows),
        "generated_token_count": sum(row["output_tokens"] for row in rows),
        "batch_pattern": list(BATCH_PATTERN),
        "new_tokens_per_request": args.new_tokens,
        "canonical_output_token_ids": canonical,
        "runtime": runtime,
        "hbm": hbm_analysis,
        "throughput_stability": throughput,
        "trace": trace_summary,
        "reclaim": reclaim,
        "gates": gates,
        "error": error,
        "rows": rows,
    }
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SOAK_RESULT", json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
