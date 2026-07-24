from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "benchmarks" / "verify_serving_acceptance.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load(VERIFIER_PATH, "verify_serving_acceptance")


def test_committed_serving_evidence_passes_all_gates():
    report = verifier.verify(ROOT)
    assert report["status"] == "PASS"
    assert report["common_greedy_prefix"] == [45, 308, 459]
    assert report["production_admission"]["dynamic_batching"] == ["vllm", "sglang"]
    assert report["production_admission"]["chunked_prefill"] == ["vllm", "sglang"]
    assert report["production_admission"]["recurrent_state_cache"] == [
        "vllm",
        "sglang",
    ]
    assert report["production_admission"]["quantized_serving"] == []


def test_corrupt_hashed_artifact_fails_closed(tmp_path):
    target = tmp_path / "repo"
    for component in ("vllm-rwkv-ascend", "rwkv7-sglang-ascend"):
        source = ROOT / component / "evidence" / "rebuild"
        destination = target / component / "evidence" / "rebuild"
        shutil.copytree(source, destination)

    artifact = target / "vllm-rwkv-ascend/evidence/rebuild/e2e_performance.json"
    data = json.loads(artifact.read_text())
    data["status"] = "FAIL"
    artifact.write_text(json.dumps(data))
    with pytest.raises(verifier.AcceptanceError, match="SHA256 mismatch"):
        verifier.verify(target)


@pytest.mark.parametrize(
    ("backend", "path"),
    [
        (
            "vllm",
            ROOT / "vllm-rwkv-ascend" / "rwkv7_vllm_ascend" / "ascend_quant.py",
        ),
        (
            "sglang",
            ROOT / "rwkv7-sglang-ascend" / "sglang_rwkv7_ascend" / "ascend_quant.py",
        ),
    ],
)
def test_serving_quant_policy_is_default_off_and_rejects_overclaim(
    backend, path, monkeypatch
):
    module = _load(path, f"_serving_quant_policy_{backend}")
    for name in (module.ENABLE_ENV, module.MANIFEST_ENV, module.RAW_ACK_ENV):
        monkeypatch.delenv(name, raising=False)
    assert module.activate_quant_from_env(backend=backend) is None

    raw = {
        "format": module.MANIFEST_FORMAT,
        "version": module.MANIFEST_VERSION,
        "backend": backend,
        "bit": 8,
        "group_size": 0,
        "activation_dtype": "float16",
        "acceptance_scope": "raw-kernel-candidate-only",
        "production_accepted": True,
        "operator_schema_sha256": module.EXPECTED_OPERATOR_SCHEMA_SHA256,
        "verified_stack": {
            "device_name": module.EXPECTED_DEVICE_NAME,
            "torch_version": module.EXPECTED_TORCH_VERSION,
            "torch_npu_version": module.EXPECTED_TORCH_NPU_VERSION,
            "cann_version": module.EXPECTED_CANN_VERSION,
        },
        "verified_ffn_shapes": [list(module.VERIFIED_FFN_SHAPES[0])],
        "admitted_rows": [module.RAW_CANDIDATE_ROWS[8][0]],
        "ffn": {"key_layers": [0], "value_layers": []},
        "source": "fp-checkpoint",
        "tensors": {},
    }
    with pytest.raises(module.AscendQuantConfigError, match="production_accepted"):
        module.AscendQuantManifest.from_mapping(raw, backend=backend)
