#!/usr/bin/env python3
"""Measure actual nn.Linear/module/unchecked/bound dispatch on Ascend 910B3.

This is intentionally separate from the raw sweep.  Production requires the
``quant_module`` and backend E2E rows to beat the same ``nn_linear`` baseline;
the ``raw_quant`` row alone can never enable ``should_quantize``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
import torch_npu

from rwkv7_ascend_quant import AscendWeightOnlyLinear


def sync() -> None:
    torch.npu.synchronize()


def timed(fn, warmup: int, iters: int, rounds: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        sync()
        samples.append((time.perf_counter_ns() - started) / 1e6 / iters)
    return statistics.median(samples), samples


def cosine(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "=" in line
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--rows", default="1,8,17,28,64")
    args = parser.parse_args()
    rows = tuple(sorted({int(item) for item in args.rows.split(",")}))
    script = Path(__file__).resolve()
    quant_module = script.parent.parent / "rwkv7_ascend_quant.py"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=script.parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = os.environ.get("RWKV7_BENCHMARK_COMMIT")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=script.parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        git_status = status.splitlines()
    except Exception:
        git_status = None
    toolkit_info_path = Path(
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    )
    driver_info_path = Path("/usr/local/Ascend/driver/version.info")
    toolkit_info = key_values(toolkit_info_path)
    driver_info = key_values(driver_info_path)
    npu_smi = subprocess.check_output(["npu-smi", "info"], text=True)
    schema = str(torch.ops.npu.npu_weight_quant_batchmatmul.default._schema)
    print(
        json.dumps(
            {
                "kind": "environment",
                "device": torch.npu.get_device_name(0),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "python": platform.python_version(),
                "operator_schema": schema,
                "operator_schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
                "script_sha256": sha256(script),
                "quant_module_sha256": sha256(quant_module),
                "benchmark_git_commit": commit,
                "git_dirty": bool(git_status) if git_status is not None else None,
                "git_status_porcelain": git_status,
                "cann": toolkit_info.get("version"),
                "cann_inner_version": toolkit_info.get("innerversion"),
                "cann_install_info_sha256": sha256(toolkit_info_path),
                "driver_version": driver_info.get("Version"),
                "driver_inner_version": driver_info.get("Innerversion"),
                "driver_package_version": driver_info.get("package_version"),
                "driver_version_info_sha256": sha256(driver_info_path),
                "npu_smi_version_line": npu_smi.splitlines()[1].strip(),
                "npu_smi_output_sha256": hashlib.sha256(npu_smi.encode()).hexdigest(),
                "warmup": args.warmup,
                "iters": args.iters,
                "rounds": args.rounds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch.manual_seed(20260724)
    device = torch.device("npu:0")
    for bit in (8, 4):
        for k, n in ((4096, 16384), (16384, 4096)):
            fp = torch.nn.Linear(k, n, bias=False, device=device, dtype=torch.float16)
            torch.nn.init.normal_(fp.weight, std=k**-0.5)
            quant = AscendWeightOnlyLinear(
                k,
                n,
                bias=False,
                bit=bit,
                group_size=128,
                enforce_verified_shape=False,
            ).load_fp_weight(fp.weight.detach())
            op = torch_npu.npu_weight_quant_batchmatmul
            for m in rows:
                x = torch.randn(m, k, device=device, dtype=torch.float16)
                bound = quant.bind_npu_fastpath(m, scope="experiment")
                if bit == 8:
                    raw_quant = lambda: op(x, quant.qweight, quant.scales)
                else:
                    raw_quant = lambda: op(
                        x,
                        quant.qweight,
                        quant.scales,
                        quant.offsets,
                        None,
                        None,
                        None,
                        128,
                        1,
                    )
                calls = {
                    "nn_linear": lambda: fp(x),
                    "raw_quant": raw_quant,
                    "quant_module": lambda: quant(x),
                    "quant_unchecked": lambda: quant.forward_unchecked(x),
                    "quant_bound": lambda: bound(x),
                }
                reference = calls["nn_linear"]()
                candidate = calls["quant_bound"]()
                sync()
                timings = {
                    name: timed(fn, args.warmup, args.iters, args.rounds)
                    for name, fn in calls.items()
                }
                fp_ms = timings["nn_linear"][0]
                print(
                    json.dumps(
                        {
                            "kind": "dispatch_result",
                            "bit": bit,
                            "group_size": 0 if bit == 8 else 128,
                            "M": m,
                            "K": k,
                            "N": n,
                            "cosine": cosine(reference, candidate),
                            "fp16_weight_bytes": fp.weight.numel() * fp.weight.element_size(),
                            "quant_weight_bytes": quant.packed_weight_bytes(),
                            **{f"{name}_ms": result[0] for name, result in timings.items()},
                            **{f"{name}_samples_ms": result[1] for name, result in timings.items()},
                            **{
                                f"{name}_speedup_vs_nn_linear": fp_ms / result[0]
                                for name, result in timings.items()
                                if name != "nn_linear"
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del x, reference, candidate, bound
            del fp, quant
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
