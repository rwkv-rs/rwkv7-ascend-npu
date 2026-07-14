"""Preflight and enumerate the fixed Qwen3.5 Dense comparison matrix.

The script intentionally refuses to manufacture a TP/PP result when the host
does not expose enough NPUs.  Benchmark launchers can consume its JSON plan;
an insufficient-device record is evidence of a blocked row, never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from model_matrix import load_manifest
except ImportError:  # package import in CPU-only tests
    from .model_matrix import load_manifest


def parse_visible_devices(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    devices = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit():
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must be a comma-separated integer list"
            )
        devices.append(int(item))
    if len(set(devices)) != len(devices):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES contains duplicate devices")
    return tuple(devices)


def detect_visible_device_count() -> int:
    configured = parse_visible_devices(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        or os.environ.get("ASCEND_VISIBLE_DEVICES")
    )
    if configured is not None:
        return len(configured)
    try:
        import torch
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count())
    except (ImportError, AttributeError, RuntimeError):
        return 0


def build_plan(manifest, visible_device_count: int) -> dict:
    rows = []
    for tier in manifest.tiers:
        required = max(
            tier.rwkv.minimum_fp16_devices,
            tier.qwen.minimum_fp16_devices,
        )
        for workload in manifest.workloads:
            status = "ready" if visible_device_count >= required else "blocked"
            reason = None
            if status == "blocked":
                reason = (
                    f"requires {required} visible NPUs, found "
                    f"{visible_device_count}"
                )
            rows.append(
                {
                    "tier_id": tier.tier_id,
                    "rwkv_model_key": tier.rwkv.model_key,
                    "qwen_model_key": tier.qwen.model_key,
                    "batch_size": workload.batch_size,
                    "prompt_length": workload.prompt_length,
                    "decode_length": workload.decode_length,
                    "dtype": "fp16",
                    "required_device_count": required,
                    "visible_device_count": visible_device_count,
                    "status": status,
                    "status_reason": reason,
                }
            )
    return {
        "benchmark": "qwen35_dense_matrix_preflight",
        "matrix_id": manifest.matrix_id,
        "visible_device_count": visible_device_count,
        "global_status": (
            "blocked"
            if any(row["status"] == "blocked" for row in rows)
            else "ready"
        ),
        "rows": rows,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(here / "qwen35_dense_matrix.json"),
    )
    parser.add_argument("--visible-device-count", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero if any matrix row lacks enough devices",
    )
    args = parser.parse_args()
    count = (
        detect_visible_device_count()
        if args.visible_device_count is None
        else args.visible_device_count
    )
    if count < 0:
        parser.error("visible-device-count must be non-negative")
    plan = build_plan(load_manifest(args.manifest), count)
    rendered = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 2 if args.require_ready and plan["global_status"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
