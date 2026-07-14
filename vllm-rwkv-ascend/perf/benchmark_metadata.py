"""Shared, side-effect-free metadata helpers for Ascend benchmark rows."""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Any


def npu_device_id(device: str) -> int:
    """Parse a logical ``npu:N`` device string for telemetry selection."""
    prefix, separator, index = device.rpartition(":")
    if separator != ":" or prefix != "npu" or not index.isdigit():
        raise ValueError(f"expected npu:N device, got {device!r}")
    return int(index)


def collect_npu_metadata(
    torch_module: Any,
    torch_npu_module: Any,
    device: str,
    *,
    device_count: int = 1,
) -> dict[str, Any]:
    """Return the hardware/runtime identity required by paired comparisons."""
    npu = torch_module.npu
    return {
        "device": device,
        "device_name": str(npu.get_device_name(device)),
        "device_count": int(device_count),
        "visible_device_count": int(npu.device_count()),
        "torch": str(torch_module.__version__),
        "torch_npu": str(torch_npu_module.__version__),
        "python": platform.python_version(),
    }


def infer_huggingface_revision(path: str | Path) -> str | None:
    """Infer one pinned commit from local-dir Hugging Face metadata files."""
    metadata_root = Path(path) / ".cache" / "huggingface" / "download"
    revisions = set()
    if metadata_root.is_dir():
        for item in metadata_root.rglob("*.metadata"):
            try:
                first_line = item.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if re.fullmatch(r"[0-9a-f]{40}", first_line):
                revisions.add(first_line)
    return next(iter(revisions)) if len(revisions) == 1 else None


def collect_cann_metadata() -> dict[str, str]:
    """Read the installed CANN identity from its resolved toolkit home."""
    configured = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    home = Path(configured).resolve()
    version = None
    version_file = home / "share" / "info" / "runtime" / "version.info"
    try:
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version="):
                version = line.partition("=")[2].strip()
                break
    except OSError:
        pass
    if not version and home.name.startswith("cann-"):
        version = home.name.removeprefix("cann-")
    return {
        "cann": version or "unknown",
        "ascend_home_path": str(home),
    }


def checkpoint_metadata(
    path: str,
    *,
    revision: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Return stable local checkpoint facts without reading model contents."""
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint_bytes = sum(
            item.stat().st_size for item in checkpoint.rglob("*") if item.is_file()
        )
    else:
        checkpoint_bytes = checkpoint.stat().st_size
    metadata = {
        "model": os.path.abspath(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_revision": revision or infer_huggingface_revision(checkpoint),
    }
    if sha256:
        metadata["checkpoint_sha256"] = sha256
    return metadata
