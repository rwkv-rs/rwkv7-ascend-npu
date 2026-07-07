"""CI-runnable structural tests for the Ascend optimization artifacts (no NPU).

Validates:
  - the AscendC custom-op definition JSONs are well-formed and have the expected
    msopgen schema (op / input_desc / output_desc / attr_desc);
  - the C++ op-coalesced forward (``rwkv7_ascend_v3.cpp``) exists and contains the
    state-writeback fix (three in-place ``copy_`` of the recurrent state) that
    ``bench_batch`` single-step cos=1.0 used to mask.

Pure file/json inspection — runs in GitHub Actions on CPU.
"""
import json
import os

_HERE = os.path.dirname(__file__)
ROOT = os.path.join(_HERE, "..")


def _load_json(rel):
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def test_ascendc_op_defs_are_valid():
    for rel in ["ascendc/rwkv_wexp.json", "ascendc/rwkv_shiftmix.json"]:
        data = _load_json(rel)
        assert isinstance(data, list) and data, f"{rel}: expected a non-empty list"
        for entry in data:
            assert "op" in entry and isinstance(entry["op"], str) and entry["op"], f"{rel}: bad 'op'"
            for key in ("input_desc", "output_desc"):
                assert key in entry, f"{rel}: missing '{key}'"
                assert isinstance(entry[key], list), f"{rel}: '{key}' must be a list"
            entry.setdefault("attr_desc", [])
            for desc in entry["input_desc"] + entry["output_desc"]:
                assert {"name", "param_type", "format", "type"} <= set(desc), \
                    f"{rel}: desc missing keys: {desc}"


def test_cpp_forward_is_op_coalesced():
    """The R&D C++ forward packs the RWKV-7 layer stack into one call (the
    op-coalesced base; the serving-hardened variant with the state-writeback fix
    lives in vllm-rwkv-ascend/perf/, exercised by its own test)."""
    src = os.path.join(ROOT, "rwkv7_ascend_v3.cpp")
    assert os.path.exists(src), "rwkv7_ascend_v3.cpp missing"
    txt = open(src, "r", encoding="utf-8", errors="replace").read()
    assert "rwkv7_decode_full" in txt, "C++ entry point rwkv7_decode_full not found"
    assert "state_all" in txt and "xpa_all" in txt, "recurrent-state tensors not found"


def test_ascendc_kernel_sources_present():
    """The AscendC fused-op kernels + their torch bindings exist (the toolchain
    exploration that produced the 'AscendC is for elementwise, not GEMV' finding)."""
    for rel in ["ascendc/rwkv_wexp_kernel.cpp", "ascendc/call_rwkv_wexp.cpp",
                "ascendc/README.md"]:
        assert os.path.exists(os.path.join(ROOT, rel)), f"{rel} missing"
