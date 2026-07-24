#!/usr/bin/env python3
"""One-command real-engine acceptance for RWKV-7/SGLang/Ascend.

The test submits a long and a short request as one Engine batch, while the long
prompt crosses multiple configured prefill chunks. After both slots are
released it repeats the short request and requires deterministic equality,
catching stale recurrent state on slot reuse.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _trim_response(response: dict) -> dict:
    meta = response.get("meta_info") or {}
    return {
        "text": response.get("text"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="acceptance-engine.json")
    ap.add_argument("--server-log", default="acceptance-engine.server.log")
    ap.add_argument("--port", type=int, default=30110)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--startup-timeout", type=float, default=420.0)
    args = ap.parse_args()

    output = Path(args.output).resolve()
    server_log = Path(args.server_log).resolve()
    trace_file = output.with_suffix(".trace.jsonl")
    worker_output = output.with_suffix(".worker.json")
    worker = Path(__file__).with_name("engine_worker.py")
    command = [
        sys.executable,
        str(worker),
        "--model",
        str(Path(args.model).resolve()),
        "--output",
        str(worker_output),
        "--chunk-size",
        str(args.chunk_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    result = {
        "schema": "rwkv7-sglang-ascend-engine-acceptance-v1",
        "started_at_unix": time.time(),
        "model": str(Path(args.model).resolve()),
        "command": command,
        "server_log": str(server_log),
        "backend_trace": str(trace_file),
        "checks": {},
        "responses": {},
        "passed": False,
    }
    proc = None
    started = time.monotonic()
    try:
        server_log.parent.mkdir(parents=True, exist_ok=True)
        with server_log.open("w") as log:
            child_env = os.environ.copy()
            child_env["RWKV_SGLANG_ACCEPTANCE_TRACE"] = str(trace_file)
            trace_file.unlink(missing_ok=True)
            worker_output.unlink(missing_ok=True)
            proc = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=child_env,
            )

        proc.wait(timeout=args.startup_timeout)
        if proc.returncode:
            raise RuntimeError(f"real Engine worker exited: rc={proc.returncode}")
        worker_result = json.loads(worker_output.read_text(encoding="utf-8"))
        long_response, short_response = worker_result["batch"]
        repeated_response = worker_result["repeated"]
        batch_elapsed = worker_result["batch_elapsed"]
        oracle = worker_result["oracle"]
        long_view = _trim_response(long_response)
        short_view = _trim_response(short_response)
        repeated_view = _trim_response(repeated_response)
        prompt_tokens = long_view["prompt_tokens"]
        result["responses"] = {
            "long": long_view,
            "short": short_view,
            "short_after_slot_release": repeated_view,
        }
        result["cross_backend_oracle"] = {
            **oracle,
            "expected_vllm_dense_output_ids": [45, 308, 459],
        }
        result["engine_config"] = worker_result["engine_config"]
        result["slot_probe_count"] = worker_result["slot_probe_count"]
        result["batch_elapsed_seconds"] = batch_elapsed
        events = [
            json.loads(line)
            for line in trace_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        marker_index = next(
            i
            for i, event in enumerate(events)
            if event.get("kind") == "marker"
            and event.get("name") == "before_slot_reuse"
        )
        before = [e for e in events[:marker_index] if e.get("kind") == "forward"]
        after = [e for e in events[marker_index + 1 :] if e.get("kind") == "forward"]
        dynamic_events = [e for e in before if e.get("real_batch_size", 0) >= 2]
        mixed_events = [
            e
            for e in before
            if e.get("mode") == "MIXED" and e.get("real_batch_size", 0) >= 2
        ]

        # Strong chunk evidence: the same physical state slot occurs in two
        # extend/mixed forwards and its prefix progresses from fresh(0) to >0.
        prefixes_by_slot: dict[int, list[int]] = {}
        for event in before:
            if event.get("mode") not in ("EXTEND", "MIXED", "SPLIT_PREFILL"):
                continue
            slots = event.get("state_slot_ids") or []
            prefixes = event.get("extend_prefix_lens") or []
            for slot, prefix in zip(slots, prefixes):
                prefixes_by_slot.setdefault(int(slot), []).append(int(prefix))
        continued_slots = {
            slot: prefixes
            for slot, prefixes in prefixes_by_slot.items()
            if 0 in prefixes and any(prefix > 0 for prefix in prefixes)
        }
        before_slots = {
            int(slot)
            for event in before
            for slot in (event.get("state_slot_ids") or [])
            if int(slot) >= 0
        }
        after_slots = {
            int(slot)
            for event in after
            for slot in (event.get("state_slot_ids") or [])
            if int(slot) >= 0
        }
        reused_slots = sorted(before_slots & after_slots)
        result["trace_evidence"] = {
            "forward_event_count": len(before) + len(after),
            "dynamic_batch_events": dynamic_events[:3],
            "mixed_decode_prefill_events": mixed_events[:3],
            "continued_slot_prefixes": continued_slots,
            "slots_before_release": sorted(before_slots),
            "slots_after_release": sorted(after_slots),
            "reused_slots": reused_slots,
        }
        result["checks"] = {
            "two_concurrent_requests_completed": bool(
                long_view["text"] and short_view["text"]
            ),
            "long_prompt_crossed_prefill_chunk": bool(
                prompt_tokens is not None and prompt_tokens > args.chunk_size
            ),
            "slot_reuse_is_deterministic": (
                short_view["text"] == repeated_view["text"]
            ),
            "backend_observed_dynamic_batch": bool(dynamic_events),
            "backend_observed_mixed_decode_prefill": bool(mixed_events),
            "backend_observed_chunk_state_continuation": bool(continued_slots),
            "backend_observed_physical_slot_reuse": bool(reused_slots),
            "radix_cache_was_disabled": bool(
                worker_result.get("engine_config", {}).get("disable_radix_cache")
            ),
            "hello_matches_vllm_dense_output_ids": (
                oracle.get("output_ids") == [45, 308, 459]
            ),
            "state_mode": "MambaPool active-request state; radix prefix cache disabled",
        }
        result["passed"] = all(
            value for value in result["checks"].values() if isinstance(value, bool)
        )
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        result["server_returncode"] = None if proc is None else proc.poll()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
