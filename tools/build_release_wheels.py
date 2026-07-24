#!/usr/bin/env python3
"""Build, inspect and install-smoke the three reproducible monorepo wheels."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import email.parser
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATE_EPOCH = 1784894901


@dataclass(frozen=True)
class PackageSpec:
    source: str
    distribution: str
    version: str
    module: str
    wheel: str
    entry_points: dict[str, dict[str, str]]


PACKAGES = (
    PackageSpec(
        source="rwkv7-hf-ascend",
        distribution="rwkv7-hf-adapter",
        version="0.6.0",
        module="rwkv7_hf",
        wheel="rwkv7_hf_adapter-0.6.0-py3-none-any.whl",
        entry_points={},
    ),
    PackageSpec(
        source="vllm-rwkv-ascend",
        distribution="rwkv7-vllm-ascend",
        version="0.3.0",
        module="rwkv7_vllm_ascend",
        wheel="rwkv7_vllm_ascend-0.3.0-py3-none-any.whl",
        entry_points={
            "vllm.general_plugins": {
                "rwkv7_ascend_model": "rwkv7_vllm_ascend.plugin:register"
            }
        },
    ),
    PackageSpec(
        source="rwkv7-sglang-ascend",
        distribution="sglang-rwkv7-ascend",
        version="0.2.0",
        module="sglang_rwkv7_ascend",
        wheel="sglang_rwkv7_ascend-0.2.0-py3-none-any.whl",
        entry_points={
            "console_scripts": {
                "sglang-rwkv7-ascend-serve": "sglang_rwkv7_ascend.serve:main"
            },
            "sglang.srt.plugins": {"rwkv7_ascend": "sglang_rwkv7_ascend:register"},
        },
    ),
)


class ReleaseError(RuntimeError):
    """A build, wheel-integrity or isolated-install gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(spec: PackageSpec) -> list[Path]:
    source = ROOT / spec.source
    package = source / spec.module
    files = [source / "pyproject.toml"]
    readme = source / "README.md"
    if readme.is_file():
        files.append(readme)
    files.extend(
        path
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    return sorted(files)


def source_tree_sha256(spec: PackageSpec) -> str:
    digest = hashlib.sha256()
    for path in _source_files(spec):
        relative = path.relative_to(ROOT).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _clean_build_state(spec: PackageSpec) -> None:
    source = ROOT / spec.source
    shutil.rmtree(source / "build", ignore_errors=True)
    shutil.rmtree(source / "dist", ignore_errors=True)
    for egg_info in source.glob("*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _zip_timestamp(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    parts = list(time.gmtime(max(source_date_epoch, 315532800))[:6])
    parts[5] -= parts[5] % 2  # DOS ZIP timestamps have two-second resolution.
    return tuple(parts)


def canonicalize_wheel(path: Path, source_date_epoch: int) -> None:
    """Repack a wheel as a host-independent, store-only ZIP archive."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "built wheel has duplicate paths")
        _require(
            all(not name.endswith("/") for name in names),
            "built wheel unexpectedly contains directory entries",
        )
        payloads = {name: archive.read(name) for name in names}

    timestamp = _zip_timestamp(source_date_epoch)
    temporary = path.with_suffix(".canonical.whl")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payloads[name])
    temporary.replace(path)


def _inspect_record(archive: zipfile.ZipFile, names: list[str], dist_info: str) -> None:
    record_name = f"{dist_info}/RECORD"
    _require(record_name in names, "wheel RECORD is missing")
    rows = list(csv.reader(archive.read(record_name).decode().splitlines()))
    _require(len(rows) == len(names), "wheel RECORD/file count mismatch")
    by_name = {row[0]: row for row in rows}
    _require(set(by_name) == set(names), "wheel RECORD paths do not match archive")
    for name in names:
        row = by_name[name]
        _require(len(row) == 3, f"invalid RECORD row: {name}")
        if name == record_name:
            _require(row[1:] == ["", ""], "RECORD must not hash itself")
            continue
        data = archive.read(name)
        _require(
            row[1] == f"sha256={_record_digest(data)}",
            f"RECORD hash mismatch: {name}",
        )
        _require(row[2] == str(len(data)), f"RECORD size mismatch: {name}")


def inspect_wheel(path: Path, spec: PackageSpec) -> dict[str, Any]:
    _require(path.name == spec.wheel, f"unexpected wheel filename: {path.name}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _require(len(names) == len(set(names)), "wheel has duplicate paths")
        for info in infos:
            pure = PurePosixPath(info.filename)
            _require(
                not pure.is_absolute() and ".." not in pure.parts,
                f"unsafe wheel path: {info.filename}",
            )
            mode = info.external_attr >> 16
            _require(not stat.S_ISLNK(mode), f"wheel contains symlink: {info.filename}")
            _require(
                not info.filename.lower().endswith((".so", ".dll", ".dylib", ".pyd")),
                f"wheel unexpectedly contains compiled code: {info.filename}",
            )

        normalized = spec.distribution.replace("-", "_")
        dist_info = f"{normalized}-{spec.version}.dist-info"
        metadata_name = f"{dist_info}/METADATA"
        wheel_name = f"{dist_info}/WHEEL"
        _require(
            metadata_name in names and wheel_name in names, "wheel metadata missing"
        )
        _require(f"{spec.module}/__init__.py" in names, "package __init__ missing")

        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_name))
        _require(metadata["Name"] == spec.distribution, "distribution name drift")
        _require(metadata["Version"] == spec.version, "distribution version drift")
        requires_python = metadata["Requires-Python"]
        _require(bool(requires_python), "Requires-Python is missing")

        wheel_metadata = email.parser.BytesParser().parsebytes(archive.read(wheel_name))
        _require(
            wheel_metadata["Root-Is-Purelib"] == "true",
            "release wheel is not pure Python",
        )
        _require(
            wheel_metadata.get_all("Tag") == ["py3-none-any"],
            "release wheel tag drift",
        )

        entry_points_name = f"{dist_info}/entry_points.txt"
        actual_entry_points: dict[str, dict[str, str]] = {}
        if entry_points_name in names:
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read_string(archive.read(entry_points_name).decode())
            actual_entry_points = {
                section: dict(parser.items(section)) for section in parser.sections()
            }
        _require(
            actual_entry_points == spec.entry_points,
            f"entry-point drift for {spec.distribution}",
        )
        _inspect_record(archive, names, dist_info)
        return {
            "filename": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "distribution": spec.distribution,
            "version": spec.version,
            "requires_python": requires_python,
            "pure_python": True,
            "tag": "py3-none-any",
            "archive_file_count": len(names),
            "source_tree_sha256": source_tree_sha256(spec),
            "entry_points": actual_entry_points,
        }


def install_smoke(path: Path, spec: PackageSpec, python: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{spec.module}-wheel-smoke-") as directory:
        target = Path(directory) / "site"
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(target),
                str(path),
            ],
            check=True,
        )
        code = f"""
import importlib
import importlib.metadata
import json
from pathlib import Path
target = Path({str(target)!r}).resolve()
dist = importlib.metadata.distribution({spec.distribution!r})
module = importlib.import_module({spec.module!r})
origin = Path(module.__file__).resolve()
assert target in origin.parents, (target, origin)
assert dist.version == {spec.version!r}, dist.version
entry_points = {{
    ep.group + ':' + ep.name: ep.value
    for ep in dist.entry_points
}}
print(json.dumps({{'origin': str(origin), 'entry_points': entry_points}}))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        result = subprocess.run(
            [python, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(result.stdout.splitlines()[-1])
        return {
            "passed": True,
            "module": spec.module,
            "installed_from_wheel": True,
            "entry_points": payload["entry_points"],
        }


def build_release(
    output: Path, *, python: str, source_date_epoch: int
) -> dict[str, Any]:
    output = output.resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    records = []
    for spec in PACKAGES:
        _clean_build_state(spec)
        subprocess.run(
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
                str(ROOT / spec.source),
            ],
            check=True,
            env=env,
        )
        wheel = output / spec.wheel
        canonicalize_wheel(wheel, source_date_epoch)
        record = inspect_wheel(wheel, spec)
        record["install_smoke"] = install_smoke(wheel, spec, python)
        records.append(record)

    actual_wheels = sorted(path.name for path in output.glob("*.whl"))
    _require(
        actual_wheels == sorted(spec.wheel for spec in PACKAGES),
        "release output contains an unexpected wheel set",
    )
    manifest = {
        "schema": "rwkv7-ascend-wheel-release-v1",
        "status": "PASS",
        "source_date_epoch": source_date_epoch,
        "build_frontend": "build==1.3.0",
        "build_backend": "setuptools==79.0.1",
        "wheel_tool": "wheel==0.46.3",
        "archive_normalization": "sorted-zip-stored-v1",
        "wheels": records,
    }
    manifest_path = output / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums = output / "SHA256SUMS"
    artifacts = [output / spec.wheel for spec in PACKAGES] + [manifest_path]
    sums.write_text(
        "".join(f"{_sha256_file(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--source-date-epoch", type=int, default=DEFAULT_SOURCE_DATE_EPOCH
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_release(
            args.output,
            python=args.python,
            source_date_epoch=args.source_date_epoch,
        )
    except (OSError, subprocess.SubprocessError, ReleaseError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
