#!/usr/bin/env python3
"""Fail-closed verifier for the committed vLLM and SGLang NPU evidence.

This verifier does not benchmark the local host.  It authenticates the
committed clean-rebuild artifacts and checks the serving invariants that those
real-engine runs are required to prove.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VLLM_EVIDENCE = ROOT / "vllm-rwkv-ascend" / "evidence" / "rebuild"
SGLANG_EVIDENCE = ROOT / "rwkv7-sglang-ascend" / "evidence" / "rebuild"
SOAK_EVIDENCE = ROOT / "benchmarks" / "results" / "serving_soak_20260724"
EXPECTED_BATCHES = (1, 4, 8)
EXPECTED_GREEDY_PREFIX = [45, 308, 459]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceError(RuntimeError):
    """An artifact is missing, corrupt, or does not satisfy a serving gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot read valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AcceptanceError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def verify_sha256sums(directory: Path, manifest_name: str = "SHA256SUMS") -> list[str]:
    """Verify every safe relative path named by a standard SHA256SUMS file."""

    manifest = directory / manifest_name
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AcceptanceError(f"cannot read hash manifest: {manifest}") from exc

    verified: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        _require(
            len(fields) == 2 and SHA256_RE.fullmatch(fields[0]) is not None,
            f"invalid {manifest_name} line {line_number}",
        )
        expected, filename = fields
        filename = filename.lstrip("*")
        relative = Path(filename)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe artifact path in {manifest_name}: {filename}",
        )
        artifact = directory / relative
        _require(artifact.is_file(), f"missing hashed artifact: {artifact}")
        actual = _sha256(artifact)
        _require(
            actual == expected,
            f"SHA256 mismatch for {artifact}: expected {expected}, got {actual}",
        )
        verified.append(relative.as_posix())

    _require(bool(verified), f"empty hash manifest: {manifest}")
    _require(len(verified) == len(set(verified)), f"duplicate path in {manifest}")
    return verified


def _verify_performance(
    path: Path, *, expected_axis: str, expected_hardware: str, expected_model: str
) -> dict[str, Any]:
    artifact = _read_json(path)
    _require(artifact.get("axis") == expected_axis, f"{path}: unexpected axis")
    _require(artifact.get("status") == "PASS", f"{path}: status is not PASS")
    _require(artifact.get("hardware") == expected_hardware, f"{path}: hardware drift")
    _require(artifact.get("model") == expected_model, f"{path}: model drift")

    gates = artifact.get("gates")
    _require(isinstance(gates, dict) and gates, f"{path}: missing gates")
    _require(
        all(value is True for value in gates.values()),
        f"{path}: at least one performance gate failed",
    )

    rows = artifact.get("rows")
    _require(isinstance(rows, list), f"{path}: rows must be a list")
    rows_by_batch = {
        row.get("batch_size"): row for row in rows if isinstance(row, dict)
    }
    _require(
        tuple(sorted(rows_by_batch)) == EXPECTED_BATCHES,
        f"{path}: expected exactly B1/B4/B8",
    )
    _require(len(rows) == len(rows_by_batch), f"{path}: duplicate performance batch")

    new_tokens = artifact.get("new_tokens_per_request")
    _require(
        isinstance(new_tokens, int) and new_tokens > 0,
        f"{path}: invalid generation length",
    )
    for batch in EXPECTED_BATCHES:
        row = rows_by_batch[batch]
        elapsed = row.get("elapsed_s")
        throughput = row.get("output_tokens_per_second")
        _require(row.get("measured") is True, f"{path}: B{batch} is not measured")
        _require(
            isinstance(elapsed, (int, float))
            and math.isfinite(elapsed)
            and elapsed > 0,
            f"{path}: B{batch} elapsed time is invalid",
        )
        _require(
            isinstance(throughput, (int, float))
            and math.isfinite(throughput)
            and throughput > 0,
            f"{path}: B{batch} throughput is invalid",
        )
        _require(
            row.get("output_tokens") == batch * new_tokens,
            f"{path}: B{batch} output-token count is inconsistent",
        )
        _require(
            row.get("batch_outputs_identical") is True,
            f"{path}: B{batch} outputs differ inside the batch",
        )
        _require(
            row.get("exact_hello_prefix") is True
            and row.get("first_output_token_ids", [])[:3] == EXPECTED_GREEDY_PREFIX,
            f"{path}: B{batch} common greedy oracle failed",
        )
        computed = row["output_tokens"] / elapsed
        _require(
            math.isclose(throughput, computed, rel_tol=1e-9, abs_tol=1e-9),
            f"{path}: B{batch} throughput is not reproducible from token/time totals",
        )

    b1 = rows_by_batch[1]["output_tokens_per_second"]
    b4 = rows_by_batch[4]["output_tokens_per_second"]
    b8 = rows_by_batch[8]["output_tokens_per_second"]
    scaling_b4 = b4 / b1
    scaling_b8 = b8 / b1
    floor = artifact.get("minimum_b4_scaling")
    _require(
        isinstance(floor, (int, float)) and scaling_b4 >= floor,
        f"{path}: B4 dynamic-batching scaling is below the declared floor",
    )
    _require(b8 >= b4, f"{path}: B8 aggregate throughput is below B4")
    _require(
        math.isclose(
            artifact.get("scaling_b4_over_b1", math.nan),
            scaling_b4,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        f"{path}: B4 scaling field is inconsistent",
    )
    _require(
        math.isclose(
            artifact.get("scaling_b8_over_b1", math.nan),
            scaling_b8,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        f"{path}: B8 scaling field is inconsistent",
    )
    return {
        "batch_output_tokens_per_second": {
            str(batch): rows_by_batch[batch]["output_tokens_per_second"]
            for batch in EXPECTED_BATCHES
        },
        "scaling_b4_over_b1": scaling_b4,
        "scaling_b8_over_b1": scaling_b8,
    }


def _verify_vllm() -> dict[str, Any]:
    hashed = verify_sha256sums(VLLM_EVIDENCE)
    required_hashes = {
        "real_engine_acceptance.json",
        "real_engine_scheduler_trace.jsonl",
        "e2e_performance.json",
        "e2e_performance.log",
    }
    _require(required_hashes <= set(hashed), "vLLM hash manifest is incomplete")

    artifact = _read_json(VLLM_EVIDENCE / "real_engine_acceptance.json")
    _require(artifact.get("status") == "ACCEPTANCE_OK", "vLLM acceptance failed")
    _require(artifact.get("engine") == "vllm-v1-ascend", "unexpected vLLM engine")
    _require(artifact.get("hardware") == "Ascend910B3 64GB", "vLLM hardware drift")
    config = artifact.get("config", {})
    _require(config.get("enable_chunked_prefill") is True, "vLLM chunking disabled")
    _require(config.get("enable_prefix_caching") is False, "vLLM prefix cache enabled")
    _require(config.get("max_num_seqs", 0) >= 3, "vLLM sequence budget too small")

    first = artifact.get("first_batch")
    second = artifact.get("slot_reuse_reverse_batch")
    _require(
        isinstance(first, dict) and isinstance(second, dict) and first == second,
        "vLLM reverse-order slot-reuse outputs differ",
    )
    _require(set(first) == {"short", "medium", "long"}, "vLLM request set drift")
    _require(
        sorted(row.get("input_tokens") for row in first.values()) == [1, 47, 180],
        "vLLM chunk-admission prompt lengths drifted",
    )
    _require(
        all(row.get("finished") is True for row in first.values()),
        "vLLM request did not finish",
    )
    _require(
        first["short"].get("output_token_ids") == EXPECTED_GREEDY_PREFIX,
        "vLLM greedy oracle failed",
    )
    _require(artifact.get("hello_greedy_exact") is True, "vLLM Hello gate failed")
    _require(
        artifact.get("physical_slot_reuse_proven") is True
        and artifact.get("reuse_outputs_identical") is True,
        "vLLM physical state-slot reuse was not proven",
    )

    trace = artifact.get("scheduler_trace")
    chunk_gate = artifact.get("chunk_gate")
    _require(isinstance(trace, dict), "vLLM scheduler summary missing")
    _require(isinstance(chunk_gate, dict), "vLLM chunk gate missing")
    budget = config.get("max_num_batched_tokens")
    _require(
        budget == chunk_gate.get("max_num_batched_tokens"),
        "vLLM scheduler/chunk budget mismatch",
    )
    _require(
        trace.get("max_prefill_tokens_in_step", math.inf) <= budget,
        "vLLM prefill step exceeded token budget",
    )
    _require(
        trace.get("prefill_steps", 0)
        >= chunk_gate.get("minimum_prefill_steps", math.inf),
        "vLLM did not cross the required number of prefill chunks",
    )
    for field in (
        "continuation_prefill_segments",
        "mixed_decode_prefill_steps",
        "multi_prefill_request_steps",
    ):
        _require(trace.get(field, 0) > 0, f"vLLM did not prove {field}")

    reused = set(trace.get("physically_reused_fresh_slots", []))
    stale_before_clear = set(
        trace.get("reused_slots_with_nonzero_state_before_clear", [])
    )
    _require(bool(reused), "vLLM no physical cache slot was reused")
    _require(
        reused <= stale_before_clear,
        "vLLM did not observe stale state before clearing every reused slot",
    )
    _require(
        trace.get("fresh_zero_events", 0) >= len(reused),
        "vLLM reused slots were not covered by zeroization events",
    )

    performance = _verify_performance(
        VLLM_EVIDENCE / "e2e_performance.json",
        expected_axis="vllm_ascend_real_7p2b_e2e_performance",
        expected_hardware=artifact["hardware"],
        expected_model=artifact["model"],
    )
    return {
        "status": "PASS",
        "engine": artifact["engine"],
        "hardware": artifact["hardware"],
        "model": artifact["model"],
        "dynamic_batching": True,
        "chunked_prefill": True,
        "mixed_decode_prefill_steps": trace["mixed_decode_prefill_steps"],
        "continuation_prefill_segments": trace["continuation_prefill_segments"],
        "physical_state_slot_reuse": sorted(reused),
        "performance": performance,
    }


def _verify_sglang(expected_model: str, expected_hardware: str) -> dict[str, Any]:
    hashed = verify_sha256sums(SGLANG_EVIDENCE)
    required_hashes = {
        "acceptance.json",
        "acceptance.trace.jsonl",
        "acceptance.worker.json",
        "e2e_performance.json",
        "e2e_performance.log",
    }
    _require(required_hashes <= set(hashed), "SGLang hash manifest is incomplete")

    artifact = _read_json(SGLANG_EVIDENCE / "acceptance.json")
    _require(artifact.get("passed") is True, "SGLang acceptance failed")
    _require(artifact.get("server_returncode") == 0, "SGLang engine exited nonzero")
    _require(artifact.get("model") == expected_model, "cross-backend model drift")

    checks = artifact.get("checks")
    _require(isinstance(checks, dict), "SGLang checks missing")
    for field in (
        "two_concurrent_requests_completed",
        "long_prompt_crossed_prefill_chunk",
        "slot_reuse_is_deterministic",
        "backend_observed_dynamic_batch",
        "backend_observed_mixed_decode_prefill",
        "backend_observed_chunk_state_continuation",
        "backend_observed_physical_slot_reuse",
        "radix_cache_was_disabled",
        "hello_matches_vllm_dense_output_ids",
    ):
        _require(checks.get(field) is True, f"SGLang gate failed: {field}")

    config = artifact.get("engine_config")
    _require(isinstance(config, dict), "SGLang engine config missing")
    _require(config.get("disable_radix_cache") is True, "SGLang radix cache enabled")
    _require(config.get("enable_mixed_chunk") is True, "SGLang mixed chunk disabled")
    chunk_size = config.get("chunked_prefill_size")
    _require(
        isinstance(chunk_size, int) and chunk_size > 0,
        "SGLang chunk size invalid",
    )
    _require(
        config.get("max_running_requests", 0) >= 2,
        "SGLang request budget cannot prove dynamic batching",
    )

    oracle = artifact.get("cross_backend_oracle")
    _require(isinstance(oracle, dict), "SGLang cross-backend oracle missing")
    _require(
        oracle.get("output_ids") == EXPECTED_GREEDY_PREFIX
        and oracle.get("expected_vllm_dense_output_ids") == EXPECTED_GREEDY_PREFIX,
        "SGLang/vLLM common greedy oracle failed",
    )
    responses = artifact.get("responses", {})
    _require(
        responses.get("short") == responses.get("short_after_slot_release"),
        "SGLang output changed after physical slot reuse",
    )

    trace = artifact.get("trace_evidence")
    _require(isinstance(trace, dict), "SGLang trace summary missing")
    dynamic_events = trace.get("dynamic_batch_events")
    mixed_events = trace.get("mixed_decode_prefill_events")
    _require(
        isinstance(dynamic_events, list)
        and any(event.get("real_batch_size", 0) >= 2 for event in dynamic_events),
        "SGLang trace has no real dynamic batch",
    )
    _require(
        isinstance(mixed_events, list)
        and any(event.get("mode") == "MIXED" for event in mixed_events),
        "SGLang trace has no mixed decode/prefill event",
    )

    long_tokens = responses.get("long", {}).get("prompt_tokens")
    continuations = trace.get("continued_slot_prefixes")
    _require(
        isinstance(continuations, dict) and continuations,
        "SGLang continuation prefixes missing",
    )
    for slot, prefixes in continuations.items():
        _require(
            isinstance(prefixes, list)
            and len(prefixes) >= 2
            and prefixes[0] == 0
            and prefixes[-1] == long_tokens,
            f"SGLang slot {slot} did not carry state across the full prompt",
        )
        _require(
            all(
                0 < current - previous <= chunk_size
                for previous, current in zip(prefixes, prefixes[1:])
            ),
            f"SGLang slot {slot} continuation offsets violate chunk size",
        )

    before = set(trace.get("slots_before_release", []))
    after = set(trace.get("slots_after_release", []))
    reused = set(trace.get("reused_slots", []))
    _require(bool(reused), "SGLang no physical state slot was reused")
    _require(
        reused == before & after,
        "SGLang reused-slot summary is inconsistent with the trace",
    )

    performance = _verify_performance(
        SGLANG_EVIDENCE / "e2e_performance.json",
        expected_axis="sglang_ascend_real_7p2b_e2e_performance",
        expected_hardware=expected_hardware,
        expected_model=expected_model,
    )
    return {
        "status": "PASS",
        "engine": "sglang-engine-eager",
        "hardware": expected_hardware,
        "model": expected_model,
        "dynamic_batching": True,
        "chunked_prefill": True,
        "mixed_decode_prefill_events": len(mixed_events),
        "continuation_state_slots": sorted(continuations),
        "physical_state_slot_reuse": sorted(reused),
        "performance": performance,
    }


def _soak_trace_summary(path: Path, backend: str) -> dict[str, Any]:
    segments = 0
    if backend == "vllm":
        zero_events = 0
        reused_events = 0
        reused_slots: set[int] = set()
        post_zero_failures = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("kind") == "soak_marker":
                    segments += 1
                elif event.get("event") == "fresh_state_zero":
                    zero_events += 1
                    if event.get("pre_zero_had_nonzero") is True:
                        reused_events += 1
                        reused_slots.add(int(event["slot"]))
                    if event.get("post_zero_nonzero") is not False:
                        post_zero_failures += 1
        return {
            "segment_count": segments,
            "fresh_zero_events": zero_events,
            "reused_nonzero_slots_before_clear": reused_events,
            "reused_slot_ids": sorted(reused_slots),
            "post_zero_failures": post_zero_failures,
            "state_reuse_gate": reused_events > 0 and post_zero_failures == 0,
        }

    _require(backend == "sglang", f"unsupported soak backend: {backend}")
    appearances: dict[int, int] = {}
    fresh_slots: set[int] = set()
    fresh_assignments = 0

    def finish_segment() -> None:
        nonlocal fresh_slots
        for slot in fresh_slots:
            appearances[slot] = appearances.get(slot, 0) + 1
        fresh_slots = set()

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("kind") == "soak_marker":
                finish_segment()
                segments += 1
                continue
            if event.get("kind") != "forward":
                continue
            prefixes = event.get("extend_prefix_lens")
            if prefixes is None:
                continue
            for slot, prefix in zip(event.get("state_slot_ids") or [], prefixes):
                if int(slot) >= 0 and int(prefix) == 0:
                    fresh_slots.add(int(slot))
    if fresh_slots:
        finish_segment()
        segments += 1
    reused = sorted(slot for slot, count in appearances.items() if count >= 2)
    fresh_assignments = sum(appearances.values())
    return {
        "segment_count": segments,
        "fresh_slot_assignments": fresh_assignments,
        "unique_fresh_slots": sorted(appearances),
        "reused_slot_ids": reused,
        "state_reuse_gate": bool(reused),
    }


def _soak_slope_per_hour(rows: list[dict[str, Any]]) -> float:
    xs = [float(row["elapsed_s"]) for row in rows]
    ys = [float(row["hbm_used_mb"]) for row in rows]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    _require(denominator > 0, "soak HBM samples have no time span")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator * 3600


def _verify_soak_backend(
    backend: str, *, expected_model: str, expected_hardware: str
) -> dict[str, Any]:
    path = SOAK_EVIDENCE / f"{backend}.json"
    artifact = _read_json(path)
    _require(artifact.get("status") == "PASS", f"{path}: soak status is not PASS")
    _require(artifact.get("backend") == backend, f"{path}: backend drift")
    _require(artifact.get("model") == expected_model, f"{path}: model drift")
    _require(artifact.get("hardware") == expected_hardware, f"{path}: hardware drift")
    gates = artifact.get("gates")
    _require(
        isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values()),
        f"{path}: at least one soak gate failed",
    )
    _require(
        artifact.get("measured_duration_s", 0) >= 1800,
        f"{path}: measured soak is shorter than 30 minutes",
    )
    rows = artifact.get("rows")
    _require(isinstance(rows, list) and len(rows) >= 32, f"{path}: too few cycles")
    _require(
        artifact.get("cycle_count") == len(rows),
        f"{path}: cycle count is inconsistent",
    )
    _require(
        [row.get("cycle") for row in rows] == list(range(len(rows))),
        f"{path}: cycle sequence is not contiguous",
    )
    _require(
        {row.get("batch_size") for row in rows} == set(EXPECTED_BATCHES),
        f"{path}: soak does not cover B1/B4/B8",
    )
    canonical = artifact.get("canonical_output_token_ids")
    _require(
        isinstance(canonical, list)
        and canonical[:3] == EXPECTED_GREEDY_PREFIX
        and len(canonical) == artifact.get("new_tokens_per_request"),
        f"{path}: canonical output is invalid",
    )
    _require(
        all(
            row.get("exact_canonical_output") is True
            and isinstance(row.get("output_tokens_per_second"), (int, float))
            and math.isfinite(row["output_tokens_per_second"])
            and row["output_tokens_per_second"] > 0
            for row in rows
        ),
        f"{path}: invalid or inexact cycle",
    )
    _require(
        artifact.get("request_count") == sum(int(row["batch_size"]) for row in rows),
        f"{path}: request count is inconsistent",
    )
    _require(
        artifact.get("generated_token_count")
        == sum(int(row["output_tokens"]) for row in rows),
        f"{path}: generated-token count is inconsistent",
    )

    hbm = artifact.get("hbm")
    _require(isinstance(hbm, dict), f"{path}: HBM analysis missing")
    _require(hbm.get("sample_count") == len(rows), f"{path}: HBM sample drift")
    window = min(8, max(1, len(rows) // 4))
    head = statistics.median(row["hbm_used_mb"] for row in rows[:window])
    tail = statistics.median(row["hbm_used_mb"] for row in rows[-window:])
    slope = _soak_slope_per_hour(rows)
    for label, actual, recorded in (
        ("head median", head, hbm.get("head_median_mb")),
        ("tail median", tail, hbm.get("tail_median_mb")),
        ("tail growth", tail - head, hbm.get("tail_growth_mb")),
        ("linear slope", slope, hbm.get("linear_slope_mb_per_hour")),
    ):
        _require(
            isinstance(recorded, (int, float))
            and math.isclose(actual, recorded, rel_tol=1e-9, abs_tol=1e-9),
            f"{path}: {label} is inconsistent",
        )
    _require(
        tail - head <= hbm.get("max_growth_mb", -math.inf)
        and slope <= hbm.get("max_slope_mb_per_hour", -math.inf),
        f"{path}: recomputed HBM gate failed",
    )

    stability = artifact.get("throughput_stability")
    _require(isinstance(stability, dict), f"{path}: throughput stability missing")
    grouped: dict[int, list[dict[str, Any]]] = {
        batch: [row for row in rows if row["batch_size"] == batch]
        for batch in EXPECTED_BATCHES
    }
    for batch, batch_rows in grouped.items():
        record = stability.get(str(batch))
        _require(isinstance(record, dict), f"{path}: B{batch} stability missing")
        throughput_window = min(8, max(1, len(batch_rows) // 4))
        first = statistics.median(
            row["output_tokens_per_second"] for row in batch_rows[:throughput_window]
        )
        last = statistics.median(
            row["output_tokens_per_second"] for row in batch_rows[-throughput_window:]
        )
        ratio = last / first
        _require(
            math.isclose(
                ratio,
                record.get("tail_over_head", math.nan),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and ratio >= record.get("minimum_tail_ratio", math.inf),
            f"{path}: B{batch} tail-throughput gate failed",
        )

    trace_summary = _soak_trace_summary(
        SOAK_EVIDENCE / f"{backend}.trace.jsonl", backend
    )
    _require(
        trace_summary == artifact.get("trace"),
        f"{path}: scheduler trace summary is inconsistent",
    )
    reclaim = artifact.get("reclaim")
    _require(
        isinstance(reclaim, dict)
        and reclaim.get("post_shutdown_hbm_mb", math.inf)
        <= reclaim.get("initial_idle_hbm_mb", -math.inf)
        + reclaim.get("tolerance_mb", -math.inf),
        f"{path}: post-shutdown HBM reclaim failed",
    )
    return {
        "status": "PASS",
        "measured_duration_s": artifact["measured_duration_s"],
        "cycles": artifact["cycle_count"],
        "requests": artifact["request_count"],
        "generated_tokens": artifact["generated_token_count"],
        "hbm_head_median_mb": head,
        "hbm_tail_median_mb": tail,
        "hbm_slope_mb_per_hour": slope,
        "post_shutdown_hbm_mb": reclaim["post_shutdown_hbm_mb"],
        "reused_slot_ids": trace_summary["reused_slot_ids"],
    }


def _verify_soak(expected_model: str, expected_hardware: str) -> dict[str, Any]:
    hashed = verify_sha256sums(SOAK_EVIDENCE)
    required = {
        "README.md",
        "script.sha256",
        "vllm.json",
        "vllm.log",
        "vllm.trace.jsonl",
        "sglang.json",
        "sglang.log",
        "sglang.trace.jsonl",
    }
    _require(required <= set(hashed), "serving soak hash manifest is incomplete")
    script_pin = (SOAK_EVIDENCE / "script.sha256").read_text(encoding="utf-8").split()
    _require(
        len(script_pin) == 2
        and script_pin[1] == "benchmarks/run_serving_soak.py"
        and script_pin[0] == _sha256(ROOT / script_pin[1]),
        "serving soak runner does not match script.sha256",
    )
    return {
        backend: _verify_soak_backend(
            backend,
            expected_model=expected_model,
            expected_hardware=expected_hardware,
        )
        for backend in ("vllm", "sglang")
    }


def verify(root: Path | None = None) -> dict[str, Any]:
    """Return the consolidated serving report or raise ``AcceptanceError``."""

    global ROOT, VLLM_EVIDENCE, SGLANG_EVIDENCE, SOAK_EVIDENCE
    previous = (ROOT, VLLM_EVIDENCE, SGLANG_EVIDENCE, SOAK_EVIDENCE)
    try:
        if root is not None:
            ROOT = root.resolve()
            VLLM_EVIDENCE = ROOT / "vllm-rwkv-ascend" / "evidence" / "rebuild"
            SGLANG_EVIDENCE = ROOT / "rwkv7-sglang-ascend" / "evidence" / "rebuild"
            SOAK_EVIDENCE = ROOT / "benchmarks" / "results" / "serving_soak_20260724"

        vllm = _verify_vllm()
        sglang = _verify_sglang(vllm["model"], vllm["hardware"])
        soak = _verify_soak(vllm["model"], vllm["hardware"])
        vllm["soak"] = soak["vllm"]
        sglang["soak"] = soak["sglang"]
        return {
            "schema": "rwkv7-ascend-serving-acceptance-v1",
            "status": "PASS",
            "hardware_scope": ["Ascend910B3 64GB"],
            "common_greedy_prefix": EXPECTED_GREEDY_PREFIX,
            "backends": {"vllm": vllm, "sglang": sglang},
            "production_admission": {
                "dense_bf16_serving": ["vllm", "sglang"],
                "dynamic_batching": ["vllm", "sglang"],
                "chunked_prefill": ["vllm", "sglang"],
                "recurrent_state_cache": ["vllm", "sglang"],
                "thirty_minute_soak": ["vllm", "sglang"],
                "quantized_serving": [],
            },
            "fail_closed": {
                "vllm_quantized_e2e": "not admitted",
                "sglang_quantized_e2e": "not admitted",
                "unmeasured_hardware": "not admitted",
            },
        }
    finally:
        ROOT, VLLM_EVIDENCE, SGLANG_EVIDENCE, SOAK_EVIDENCE = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report")
    args = parser.parse_args()

    try:
        report = verify(args.root)
    except AcceptanceError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
