from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "2026.07.24"
SCRIPT = ROOT / "tools" / "build_release_wheels.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_release_wheels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_committed_wheels_match_manifest_and_source_tree() -> None:
    builder = _load_builder()
    manifest = json.loads((RELEASE / "release_manifest.json").read_text())

    assert manifest["schema"] == "rwkv7-ascend-wheel-release-v1"
    assert manifest["status"] == "PASS"
    assert manifest["source_date_epoch"] == builder.DEFAULT_SOURCE_DATE_EPOCH
    assert len(manifest["wheels"]) == len(builder.PACKAGES) == 3

    by_filename = {record["filename"]: record for record in manifest["wheels"]}
    for package in builder.PACKAGES:
        inspected = builder.inspect_wheel(RELEASE / package.wheel, package)
        recorded = by_filename[package.wheel]
        for key in (
            "filename",
            "sha256",
            "bytes",
            "distribution",
            "version",
            "requires_python",
            "pure_python",
            "tag",
            "archive_file_count",
            "source_tree_sha256",
            "entry_points",
        ):
            assert inspected[key] == recorded[key]
        assert recorded["install_smoke"]["passed"] is True
        assert recorded["install_smoke"]["installed_from_wheel"] is True


def test_sha256sums_covers_release_payload() -> None:
    expected = {}
    for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        expected[filename] = digest

    assert set(expected) == {
        "release_manifest.json",
        "rwkv7_hf_adapter-0.6.0-py3-none-any.whl",
        "rwkv7_vllm_ascend-0.3.0-py3-none-any.whl",
        "sglang_rwkv7_ascend-0.2.0-py3-none-any.whl",
    }
    for filename, digest in expected.items():
        assert _sha256(RELEASE / filename) == digest


def test_hardware_install_smoke_matches_wheels() -> None:
    evidence = json.loads((RELEASE / "ascend_install_smoke.json").read_text())
    manifest = json.loads((RELEASE / "release_manifest.json").read_text())
    hashes = {record["distribution"]: record["sha256"] for record in manifest["wheels"]}

    assert evidence["status"] == "PASS"
    assert evidence["hardware"]["device"] == "Ascend910B3"
    assert evidence["hardware"]["npu_available"] is True
    assert evidence["install"] == {
        "isolated_target": True,
        "pip_no_deps": True,
        "module_origin_asserted_inside_target": True,
    }
    assert len(evidence["components"]) == 3
    for component in evidence["components"]:
        assert component["sha256"] == hashes[component["distribution"]]
        assert component["import"] == "PASS"


def test_wheel_inspection_fails_closed_on_duplicate_member(tmp_path: Path) -> None:
    builder = _load_builder()
    package = builder.PACKAGES[1]
    corrupt = tmp_path / package.wheel
    shutil.copy2(RELEASE / package.wheel, corrupt)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(corrupt, "a") as archive:
            archive.writestr(f"{package.module}/__init__.py", "# duplicate\n")

    with pytest.raises(builder.ReleaseError, match="duplicate paths"):
        builder.inspect_wheel(corrupt, package)
