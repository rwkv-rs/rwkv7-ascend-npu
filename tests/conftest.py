"""pytest config: put serving/ on sys.path; skip NPU integration tests in CI."""
import sys, os
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "serving"))  # repo layout
sys.path.insert(0, os.path.join(_HERE, ".."))              # flat layout (910B3 dev)

import pytest


def _has_npu():
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(getattr(torch.npu, "is_available", lambda: False)())
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "npu: requires an Ascend NPU (skipped in CI)")


def pytest_ignore_collect(collection_path, config):
    # Don't even IMPORT test_integration when there's no NPU — it pulls torch_npu
    # + rwkv7_hf, which CI doesn't have. test_sampler stays (pure torch).
    name = os.path.basename(str(collection_path))
    if "test_integration" in name and not _has_npu():
        return True
    return False
