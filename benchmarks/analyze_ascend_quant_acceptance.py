#!/usr/bin/env python3
"""Validate a clean 910B3 quant-dispatch profile.

The gate intentionally accepts only raw-operator *candidates*.  Module-path
results are reported separately and never populate production policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

POLICY = {4: {1, 8}, 8: {17, 28}}
SHAPES = {(4096, 16384), (16384, 4096)}
MIN_SPEED = 1.02
MIN_COS = {8: 0.9999, 4: 0.992}
MAX_RATIO = {8: 0.51, 4: 0.28}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lines = args.jsonl.read_text(encoding="utf-8").splitlines()
    malformed = [line for line in lines if line.strip() and not line.startswith("{")]
    rows = [json.loads(line) for line in lines if line.startswith("{")]
    env = next((row for row in rows if row.get("kind") == "environment"), None)
    errors: list[str] = []
    if malformed:
        errors.append(f"{len(malformed)} non-JSON line(s) in evidence")
    if (
        not env
        or env.get("device") != "Ascend910B3"
        or env.get("torch_npu") != "2.9.0"
        or env.get("cann") != "8.5.0"
    ):
        errors.append("wrong or missing environment row")

    results = {
        (row["bit"], row["M"], row["K"], row["N"]): row
        for row in rows
        if row.get("kind") == "dispatch_result"
    }
    summary = {
        "source": str(args.jsonl),
        "environment": env,
        "thresholds": {
            "raw_candidate_min_speedup": MIN_SPEED,
            "min_cosine": MIN_COS,
            "max_storage_ratio": MAX_RATIO,
        },
        "bits": {},
        "raw_candidate_gate_passed": False,
        "production_gate_passed": False,
        "errors": errors,
    }

    for bit, batches in POLICY.items():
        accepted = []
        for k, n in sorted(SHAPES):
            for m in sorted(batches):
                row = results.get((bit, m, k, n))
                if row is None:
                    errors.append(f"missing W{bit} M{m} K{k} N{n}")
                    continue
                ratio = row["quant_weight_bytes"] / row["fp16_weight_bytes"]
                if ratio > MAX_RATIO[bit]:
                    errors.append(
                        f"W{bit} storage failed M{m} K{k} N{n}: {ratio}"
                    )
                if row["raw_quant_speedup_vs_nn_linear"] < MIN_SPEED:
                    errors.append(
                        f"W{bit} raw speed failed M{m} K{k} N{n}: "
                        f"{row['raw_quant_speedup_vs_nn_linear']}"
                    )
                if row["cosine"] < MIN_COS[bit]:
                    errors.append(
                        f"W{bit} cosine failed M{m} K{k} N{n}: {row['cosine']}"
                    )
                accepted.append(row)

        summary["bits"][str(bit)] = {
            "raw_candidate_batches": sorted(batches),
            "shapes": [list(shape) for shape in sorted(SHAPES)],
            "count": len(accepted),
            "raw_speedup_min": min(
                (row["raw_quant_speedup_vs_nn_linear"] for row in accepted),
                default=None,
            ),
            "raw_speedup_max": max(
                (row["raw_quant_speedup_vs_nn_linear"] for row in accepted),
                default=None,
            ),
            "bound_speedup_min": min(
                (row["quant_bound_speedup_vs_nn_linear"] for row in accepted),
                default=None,
            ),
            "module_speedup_min": min(
                (row["quant_module_speedup_vs_nn_linear"] for row in accepted),
                default=None,
            ),
            "module_speedup_max": max(
                (row["quant_module_speedup_vs_nn_linear"] for row in accepted),
                default=None,
            ),
            "min_cosine": min(
                (row["cosine"] for row in accepted), default=None
            ),
            "storage_ratio": max(
                (
                    row["quant_weight_bytes"] / row["fp16_weight_bytes"]
                    for row in accepted
                ),
                default=None,
            ),
        }

    summary["raw_candidate_gate_passed"] = not errors
    # Deliberately false until a model/backend E2E latency and quality gate is
    # committed; this analyzer only measures isolated FFN dispatch.
    summary["production_gate_passed"] = False
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if summary["raw_candidate_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
