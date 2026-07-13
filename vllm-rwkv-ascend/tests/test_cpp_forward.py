"""CI-runnable structural test for the serving C++ forward (no NPU).

Verifies the **state-writeback correctness fix** is present in
`perf/rwkv7_ascend_v3.cpp`: the forward must copy the evolved recurrent state back
to the Python-passed tensors (three `.copy_()`: state, xpa, xpf). Without it, the
macro reassigns a local C++ variable and multi-step generation collapses to a fixed
cycle — a bug masked by single-step cos=1.0 and `bench_batch` re-zeroing state.

Pure file inspection — runs in GitHub Actions on CPU.
"""
import os

_HERE = os.path.dirname(__file__)
SRC = os.path.join(_HERE, "..", "perf", "rwkv7_ascend_v3.cpp")


def test_state_writeback_fix_present():
    assert os.path.exists(SRC), "perf/rwkv7_ascend_v3.cpp missing"
    txt = open(SRC, "r", encoding="utf-8", errors="replace").read()
    assert "rwkv7_decode_full" in txt, "C++ entry point rwkv7_decode_full not found"
    for needle in ("state_all[li].copy_(", "xpa_all[li].copy_(", "xpf_all[li].copy_("):
        assert needle in txt, f"state-writeback fix missing: '{needle}' not found"


def test_addcmul_shift_mix_remains_benchmark_only():
    txt = open(SRC, "r", encoding="utf-8", errors="replace").read()
    assert "#ifdef RWKV7_USE_ADDCMUL_SHIFT_MIX" in txt
    assert "RWKV7_SHIFT_MIX" in txt

    engine_path = os.path.join(_HERE, "..", "serving", "serve_engine.py")
    engine = open(engine_path, "r", encoding="utf-8", errors="replace").read()
    assert "RWKV7_ADDCMUL_SHIFT_MIX" not in engine
