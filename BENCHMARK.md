# Performance: RWKV7 on Ascend 910B3 vs RTX 5070 (CUDA)

## Headline

**Same pure-PyTorch RWKV7 code → the 910B3 NPU ≈ the RTX 5070 (NPU ~1.15×).** The
910B3 hardware is comparable to a modern consumer GPU for this workload. The
optimized-path gap (our C++ forward vs the HF adapter's `native_graph`) is a
**software** difference, not a hardware ceiling.

---

## Two comparisons — they measure different things, don't conflate them

### 1. Same code (op-shim, pure PyTorch) — isolates HARDWARE

The op-shim (`rwkv7_npu_ops.py` — pure-PyTorch `rwkv7_*` ops) + the Albatross
standalone, run on **both** the 910B3 NPU and the 5070 (CUDA). Same code, only the
backend differs.

| backend | tok/s (0.1B, 16-token forward) |
|---|---|
| **910B3 NPU** (CANN 8.5.0, torch_npu 2.9.0) | **63** |
| **RTX 5070 Laptop** (sm_120, torch 2.11+cu128) | **55** |
| **ratio** | **NPU ~1.15× CUDA** |

Both correct (greedy continuation `[0..15] → 16`). The NPU is ~15% **faster** than
the 5070 for the same pure-PyTorch code (both are launch-overhead-bound on this
path). **The hardware is comparable — the NPU is not a "3× slower chip."**

### 2. Optimized paths (different code) — the SOFTWARE gap

| path | tok/s (0.1B, B=64 aggregate) |
|---|---|
| **our C++ forward** (910B3 NPU, libtorch/aclnn) | **3,666** |
| **HF adapter `native_graph`** (5070 CUDA) | **10,219** |

These are **different code** (our `rwkv7_ascend_v3.cpp` vs the adapter's fused
`native_graph`). The 5070's 2.8× here is **software**: the adapter's
`native_graph` (fused ops + CUDA graph) is far more optimized than our C++ forward.
Same-code (comparison 1), the NPU ≈ the 5070 — so the NPU's lower optimized
throughput is a **software gap, closable with fusion work** (toward `native_graph`-
level), not a hardware ceiling.

---

## "2× Albatross" — the earlier framing, and its caveat

Earlier results (and older docs) framed the NPU's batched aggregate throughput as
">2× Albatross" (the Albatross faster3a single-stream reference ≈ 1500 tok/s for
0.1B). That is an **aggregate-vs-single-stream** comparison (many batched serving
users vs one single-stream engine) — a valid *serving* metric, but **not** a
same-code/same-hardware comparison. Comparison 1 above is the cleaner hardware
read. Both are true; don't conflate them.

---

## Takeaway

- **910B3 NPU hardware ≈ RTX 5070** for RWKV7 (same code, NPU slightly faster) —
  not a slow chip.
- Our C++ forward (3666) < HF adapter `native_graph` (10219): a **software gap**,
  closable with more fusion (the AscendC GEMV-Cube / native_graph-equivalent work,
  multi-month) — not a hardware limit.
- The NPU work's value: RWKV7 runs **on NPU** (where CUDA can't), at a usable rate
  (3666 tok/s aggregate), with a correct + tested serving framework. It doesn't beat
  a heavily-optimized CUDA path, but it doesn't need to — it's a different platform.

---

## Reproduce

- **NPU op-shim** (same-code): `python npu_op_shim_bench.py` on 910B3 → 63 tok/s.
- **CUDA op-shim** (same-code): `python op_shim_cuda_bench.py` on the 5070 → 55 tok/s.
- **NPU C++ forward** (optimized): `/root/rwkv7-ascend/bench_batch.py` → 3666 tok/s @ B=64.
- **CUDA HF adapter** (optimized): load the HF model + `generate` on cuda:0 → 10219 tok/s @ B=64.
- **All 6 model sizes** (NPU C++ forward, aggregate): see `serving/SERVING.md`.

> Quantization note: W8A16 (`npu_weight_quant_batchmatmul`) was PoC'd on the NPU —
> correct but **measured slower** than fp16 (fp16 matmul already memory-BW-bound).
> See `serving/QUANT.md`.
