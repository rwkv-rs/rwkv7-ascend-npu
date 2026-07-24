from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_serving_soak.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_serving_soak", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


soak = _load()


def test_parse_npu_smi_hbm_uses_largest_capacity():
    text = """
    | DDR  0 / 0 |
    | HBM  3451 / 65536 |
    | cache 8 / 64 |
    """
    assert soak.parse_npu_smi_hbm(text) == (3451, 65536)
    with pytest.raises(soak.SoakError, match="no HBM"):
        soak.parse_npu_smi_hbm("no memory counters")


def _rows(hbm, throughputs=None):
    if throughputs is None:
        throughputs = [100.0] * len(hbm)
    return [
        {
            "elapsed_s": index * 60.0,
            "hbm_used_mb": value,
            "batch_size": 1 if index % 2 == 0 else 4,
            "output_tokens_per_second": throughputs[index],
        }
        for index, value in enumerate(hbm)
    ]


def test_hbm_and_throughput_analysis_are_fail_closed():
    stable = _rows([1000, 1002, 1001, 1003, 1002, 1001, 1002, 1003])
    report = soak.analyze_hbm(stable, max_growth_mb=16, max_slope_mb_per_hour=16)
    assert report["growth_gate"]
    assert report["slope_gate"]

    growing = _rows([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070])
    report = soak.analyze_hbm(growing, max_growth_mb=16, max_slope_mb_per_hour=16)
    assert not report["growth_gate"]
    assert not report["slope_gate"]

    perf = soak.analyze_throughput(
        _rows(
            [1000] * 8,
            [100, 100, 100, 100, 100, 100, 100, 100],
        ),
        minimum_tail_ratio=0.8,
    )
    assert all(row["gate"] for row in perf.values())


def test_vllm_trace_requires_zeroized_reuse(tmp_path):
    path = tmp_path / "vllm.jsonl"
    events = [
        {
            "event": "fresh_state_zero",
            "slot": 1,
            "pre_zero_had_nonzero": False,
            "post_zero_nonzero": False,
        },
        {"kind": "soak_marker"},
        {
            "event": "fresh_state_zero",
            "slot": 1,
            "pre_zero_had_nonzero": True,
            "post_zero_nonzero": False,
        },
        {"kind": "soak_marker"},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    report = soak.summarize_trace(path, "vllm")
    assert report["state_reuse_gate"]
    assert report["reused_slot_ids"] == [1]

    events[-2]["post_zero_nonzero"] = True
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    assert not soak.summarize_trace(path, "vllm")["state_reuse_gate"]


def test_sglang_trace_requires_fresh_physical_slot_reuse(tmp_path):
    path = tmp_path / "sglang.jsonl"
    events = [
        {
            "kind": "forward",
            "state_slot_ids": [4],
            "extend_prefix_lens": [0],
        },
        {"kind": "soak_marker"},
        {
            "kind": "forward",
            "state_slot_ids": [4],
            "extend_prefix_lens": [0],
        },
        {"kind": "soak_marker"},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    report = soak.summarize_trace(path, "sglang")
    assert report["state_reuse_gate"]
    assert report["reused_slot_ids"] == [4]
