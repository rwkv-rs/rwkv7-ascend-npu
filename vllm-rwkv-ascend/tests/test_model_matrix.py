import json
import subprocess
import sys
from pathlib import Path

import pytest

from perf.model_matrix import (
    MatrixValidationError,
    NormalizedRow,
    evaluate_matrix,
    load_manifest,
    normalize_result_document,
    render_markdown,
)
from perf.run_qwen35_dense_matrix import build_plan, parse_visible_devices


PERF_DIR = Path(__file__).resolve().parents[1] / "perf"
MANIFEST = PERF_DIR / "qwen35_dense_matrix.json"


def test_default_manifest_contains_all_dense_deployment_tiers():
    matrix = load_manifest(MANIFEST)

    assert {
        tier.tier_id: (tier.rwkv.model_key, tier.qwen.model_key)
        for tier in matrix.tiers
    } == {
        "dense-0": ("rwkv7-0.4b", "qwen3.5-0.8b"),
        "dense-1": ("rwkv7-1.5b", "qwen3.5-2b"),
        "dense-2": ("rwkv7-2.9b", "qwen3.5-4b"),
        "dense-3": ("rwkv7-7.2b", "qwen3.5-9b"),
        "dense-4": ("rwkv7-13.3b", "qwen3.5-27b"),
    }
    assert {
        (workload.batch_size, workload.prompt_length)
        for workload in matrix.workloads
    } == {(1, 512), (4, 512), (1, 2048), (4, 2048)}
    assert all(workload.decode_length == 128 for workload in matrix.workloads)


def test_multicard_preflight_blocks_27b_on_one_card():
    matrix = load_manifest(MANIFEST)

    plan = build_plan(matrix, visible_device_count=1)

    assert plan["global_status"] == "blocked"
    tier_27b = [row for row in plan["rows"] if row["tier_id"] == "dense-4"]
    assert tier_27b
    assert all(row["status"] == "blocked" for row in tier_27b)
    assert all("requires 2 visible NPUs" in row["status_reason"] for row in tier_27b)
    assert all(
        row["status"] == "ready"
        for row in plan["rows"]
        if row["tier_id"] != "dense-4"
    )


def test_visible_device_parser_rejects_ambiguous_input():
    assert parse_visible_devices("0, 2") == (0, 2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_visible_devices("0,0")
    with pytest.raises(ValueError, match="integer list"):
        parse_visible_devices("0,npu1")


def test_manifest_rejects_duplicate_tier_ids(tmp_path):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["tiers"].append(raw["tiers"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="duplicate tier_id dense-0"):
        load_manifest(path)


def test_manifest_rejects_non_positive_parameters(tmp_path):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["tiers"][0]["rwkv"]["parameters_billions"] = 0
    path = tmp_path / "invalid-parameters.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        MatrixValidationError,
        match="dense-0.rwkv.parameters_billions must be positive",
    ):
        load_manifest(path)


def test_manifest_rejects_workload_without_b4(tmp_path):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["workloads"] = [
        workload for workload in raw["workloads"]
        if workload["batch_size"] == 1
    ]
    path = tmp_path / "missing-b4.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="workloads must include B1 and B4"):
        load_manifest(path)


def _passing_rows(matrix):
    rows = []
    for tier in matrix.tiers:
        for workload in matrix.workloads:
            common = {
                "tier_id": tier.tier_id,
                "device_name": "Ascend 910B2C",
                "device_count": max(
                    tier.rwkv.minimum_fp16_devices,
                    tier.qwen.minimum_fp16_devices,
                ),
                "dtype": "fp16",
                "batch_size": workload.batch_size,
                "prompt_length": workload.prompt_length,
                "decode_length": workload.decode_length,
                "peak_memory_mib": 2000.0,
                "memory_scope": "all_npu_processes_on_selected_devices",
                "run_status": "ok",
            }
            rows.append(
                NormalizedRow(
                    engine="rwkv7_ascendc",
                    model_key=tier.rwkv.model_key,
                    model_family="rwkv",
                    prefill_tokens_per_second=1200.0,
                    decode_tokens_per_second=200.0,
                    correctness_passed=True,
                    source_path="rwkv.json",
                    **common,
                )
            )
            rows.append(
                NormalizedRow(
                    engine="vllm_ascend",
                    model_key=tier.qwen.model_key,
                    model_family="qwen",
                    prefill_tokens_per_second=1000.0,
                    decode_tokens_per_second=100.0,
                    correctness_passed=None,
                    source_path="qwen.json",
                    **common,
                )
            )
    return rows


def test_matrix_pass_requires_every_dense_tier_and_workload():
    matrix = load_manifest(MANIFEST)

    report = evaluate_matrix(matrix, _passing_rows(matrix))

    assert report.global_status == "pass"
    assert len(report.rows) == len(matrix.tiers) * len(matrix.workloads)
    assert all(row.status == "pass" for row in report.rows)


@pytest.mark.parametrize(
    ("mutation", "expected_status", "expected_reason"),
    [
        ("missing", "missing", "paired result is missing"),
        ("blocked", "blocked", "insufficient devices"),
        ("correctness", "fail", "RWKV correctness failed"),
        ("prefill", "fail", "RWKV prefill is not faster"),
        ("decode", "fail", "RWKV decode is not faster"),
        ("memory", "fail", "RWKV peak memory is higher"),
        ("hardware", "fail", "paired hardware or dtype differs"),
        ("memory_scope", "fail", "paired memory scope is missing or differs"),
    ],
)
def test_matrix_rejects_incomplete_or_non_winning_rows(
    mutation, expected_status, expected_reason
):
    matrix = load_manifest(MANIFEST)
    rows = _passing_rows(matrix)
    target = rows[0]
    if mutation == "missing":
        rows.pop(1)
    elif mutation == "blocked":
        rows[0] = target.with_updates(
            run_status="blocked", status_reason="insufficient devices"
        )
    elif mutation == "correctness":
        rows[0] = target.with_updates(correctness_passed=False)
    elif mutation == "prefill":
        rows[0] = target.with_updates(prefill_tokens_per_second=900.0)
    elif mutation == "decode":
        rows[0] = target.with_updates(decode_tokens_per_second=90.0)
    elif mutation == "memory":
        rows[0] = target.with_updates(peak_memory_mib=2100.0)
    elif mutation == "hardware":
        rows[0] = target.with_updates(device_name="Different NPU")
    elif mutation == "memory_scope":
        rows[0] = target.with_updates(memory_scope="torch_allocator")

    report = evaluate_matrix(matrix, rows)
    first = next(row for row in report.rows if row.tier_id == "dense-0")

    assert report.global_status == expected_status
    assert first.status == expected_status
    assert expected_reason in first.reasons


def test_normalize_existing_rwkv_prefill_result():
    matrix = load_manifest(MANIFEST)
    document = {
        "benchmark": "rwkv7_pth_prefill_npu",
        "model": "/models/RWKV-x070-World-0.4B.pth",
        "dtype": "fp16",
        "device_name": "Ascend 910B2C",
        "device_count": 1,
        "shape": {"batch_size": 1, "prompt_length": 512},
        "decode_length": 128,
        "correctness": {
            "greedy_match": True,
            "logits_cosine": 0.9999996,
        },
        "layer_major_tokens_per_second": 7338.37,
        "peak_memory_mib": 1160.65,
        "peak_memory_scope": "all_npu_processes_on_selected_devices",
    }

    rows = normalize_result_document(matrix, document, "rwkv.json")

    assert len(rows) == 1
    assert rows[0].tier_id == "dense-0"
    assert rows[0].model_key == "rwkv7-0.4b"
    assert rows[0].prefill_tokens_per_second == 7338.37
    assert rows[0].decode_tokens_per_second is None
    assert rows[0].correctness_passed is True


def test_normalize_existing_qwen_vllm_result():
    matrix = load_manifest(MANIFEST)
    document = {
        "benchmark": "qwen35_vllm_ascend",
        "model": "/models/Qwen3.5-2B",
        "dtype": "fp16",
        "device_name": "Ascend 910B2C",
        "device_count": 1,
        "batch_size": 4,
        "prompt_length": 512,
        "decode_length": 128,
        "prefill_tokens_per_second": 20000.0,
        "decode_tokens_per_second": 160.0,
        "peak_memory_mib": 5000.0,
        "peak_memory_scope": "all_npu_processes_on_selected_devices",
    }

    rows = normalize_result_document(matrix, document, "qwen.json")

    assert len(rows) == 1
    assert rows[0].tier_id == "dense-1"
    assert rows[0].model_key == "qwen3.5-2b"
    assert rows[0].batch_size == 4
    assert rows[0].decode_tokens_per_second == 160.0


def test_normalize_transformers_document_expands_rows():
    matrix = load_manifest(MANIFEST)
    document = {
        "benchmark": "qwen35_transformers_npu",
        "model": "/models/Qwen3.5-4B",
        "dtype": "fp16",
        "device_name": "Ascend 910B2C",
        "device_count": 1,
        "rows": [
            {
                "batch_size": 1,
                "prompt_length": 512,
                "decode_length": 128,
                "prefill_tokens_per_second": 3000.0,
                "decode_tokens_per_second": 30.0,
                "peak_memory_mib": 9000.0,
                "peak_memory_scope": "all_npu_processes_on_selected_devices",
            },
            {
                "batch_size": 4,
                "prompt_length": 512,
                "decode_length": 128,
                "prefill_tokens_per_second": 10000.0,
                "decode_tokens_per_second": 120.0,
                "peak_memory_mib": 10000.0,
                "peak_memory_scope": "all_npu_processes_on_selected_devices",
            },
        ],
    }

    rows = normalize_result_document(matrix, document, "transformers.json")

    assert [row.batch_size for row in rows] == [1, 4]
    assert all(row.tier_id == "dense-2" for row in rows)


def test_normalize_curated_dense_evidence_rows():
    matrix = load_manifest(MANIFEST)
    document = {
        "benchmark": "qwen35_dense_evidence",
        "model": "fixed-five-tier-matrix",
        "device_name": "Ascend910B2C",
        "device_count": 1,
        "dtype": "fp16",
        "peak_memory_scope": "all_npu_processes_on_selected_devices",
        "rows": [
            {
                "tier_id": "dense-0",
                "model_family": "rwkv",
                "model_key": "rwkv7-0.4b",
                "engine": "rwkv7_ascendc",
                "batch_size": 1,
                "prompt_length": 512,
                "decode_length": 128,
                "prefill_tokens_per_second": 1200.0,
                "decode_tokens_per_second": 200.0,
                "peak_memory_mib": 1000.0,
                "correctness_passed": True,
                "remote_source": "rwkv.json",
            }
        ],
    }

    rows = normalize_result_document(matrix, document, "evidence.json")

    assert len(rows) == 1
    assert rows[0].tier_id == "dense-0"
    assert rows[0].correctness_passed is True
    assert rows[0].source_path.endswith("#rwkv.json")


def test_markdown_report_exposes_global_and_each_tier():
    matrix = load_manifest(MANIFEST)
    report = evaluate_matrix(matrix, _passing_rows(matrix))

    markdown = render_markdown(report)

    assert "Global status: **PASS**" in markdown
    for tier in matrix.tiers:
        assert tier.tier_id in markdown


def test_analyzer_cli_strict_mode_rejects_incomplete_matrix(tmp_path):
    result = tmp_path / "qwen.json"
    result.write_text(
        json.dumps(
            {
                "benchmark": "qwen35_vllm_ascend",
                "model": "/models/Qwen3.5-0.8B",
                "dtype": "fp16",
                "device_name": "Ascend 910B2C",
                "device_count": 1,
                "batch_size": 1,
                "prompt_length": 512,
                "decode_length": 128,
                "prefill_tokens_per_second": 1000.0,
                "decode_tokens_per_second": 100.0,
                "peak_memory_mib": 2000.0,
                "peak_memory_scope": "all_npu_processes_on_selected_devices",
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PERF_DIR / "analyze_qwen35_matrix.py"),
            str(result),
            "--json-output",
            str(json_output),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Global status: **MISSING**" in completed.stdout
    report = json.loads(json_output.read_text(encoding="utf-8"))
    assert report["global_status"] == "missing"
    assert report["normalized_result_count"] == 1
