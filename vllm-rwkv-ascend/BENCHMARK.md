# Performance: RWKV7 on Ascend 910B3 vs RTX 5070 / A100 (CUDA)

## Headline

**`torch.npu.NPUGraph` decode brings single-sequence (B=1) latency to A100 parity.**
Capturing the C++ decode step as one device-side graph replay collapses the ~960
host dispatches per step → **0.1B B=1: 60 → 353 tok/s (5.9×), matching an A100's
CUDA-graph decode (368)**; 1.5B B=1: 30 → 113 tok/s (3.7×). Bit-exact vs eager.

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

## NPUGraph decode — B=1 latency at A100 parity (the launch-overhead fix)

The C++ forward (`rwkv7_ascend_v3.cpp`) coalesces ~960 ops into one call, but each op
is still a CANN kernel launch the host must dispatch (~17µs each on the 910B3). So
**eager decode latency is B-independent** — ~16ms for 0.1B whether B=1 or B=64 — i.e.
purely dispatch-bound, not compute (`at::linear` is 0.024ms at both B=1 and B=128).

`torch.npu.NPUGraph` records the whole step into one device-side graph; `replay()`
re-runs it with a single host launch, removing the dispatch overhead. Measured on the
910B3 (CANN 8.5.0, random-init, pure speed — `perf/bench_npugraph.py`):

| model | B | eager ms/step | graph ms/step | eager tok/s | graph tok/s | speedup |
|---|---|---|---|---|---|---|
| 0.1B | 1 | 16.7 | 2.8 | 60 | **353** | **5.9×** |
| 0.1B | 8 | 16.3 | 4.1 | 492 | 1969 | 4.0× |
| 0.1B | 64 | 16.6 | 6.7 | 3852 | **9508** | 2.5× |
| 1.5B | 1 | 35.3 | 8.9 | 28 | **113** | **3.7×** |
| 1.5B | 8 | 34.7 | 11.3 | 230 | 711 | 3.1× |
| 1.5B | 64 | 32.4 | 23.6 | 1978 | 2708 | 1.4× |

Bit-exact vs eager (single-step maxabs=0; multi-step greedy tokens identical —
`tests/test_npugraph_correctness.py`). Graph latency is also **stable across runs**
(immune to host contention — the 910B3 here is shared with sglang), where eager
latency drifts with host load.

**vs A100** (CUDA-graph decode, `rwkv7_forward_token` + `native_graph`, from the
rwkv7-hf-adapter `bench/results.jsonl`):

| model | B | 910B3 NPUGraph | A100 native_graph | ratio |
|---|---|---|---|---|
| 0.1B | 1 | 2.8ms / 353 tok/s | 2.71ms / 368 tok/s | **≈ parity (0.97×)** |
| 1.5B | 1 | 8.9ms / 113 tok/s | 6.08ms / 164.5 tok/s | 0.69× |

At small compute (0.1B B=1), both are dispatch-bound and NPUGraph reaches parity.
At larger compute (1.5B+), A100's raw kernel speed shows (the remaining gap is the
NPU's per-kernel execution, not dispatch) — the next lever there is op-coalescing /
fusion, **not** AscendC GEMV kernels (see the correction below).

Enable in serving with `--graph-decode` (or `RWKV7_GRAPH_DECODE=1`); B=1 steps replay
the graph, B>1 falls back to the eager batched forward. See `serving/graph_decode.py`.

## ⚠️ Correction: the "AscendC GEMV-Cube" direction is wrong for B=1

An earlier conclusion (and older docs) said the B=1 gap needed "AscendC GEMV-Cube
fusion (multi-month)." **Disproven by measurement**: a custom AscendC **Cube** matmul
kernel (the canonical `MatmulLeakyreluCustom` sample, built + run on the 910B3) is
**0.79ms vs `at::linear`'s 0.024ms for a B=1 GEMV — 33× SLOWER.** The Cube unit is a
dense-GEMM (compute) engine; a B=1 projection is a GEMV (memory-BW-bound), and CANN's
`at::linear` already uses an optimal GEMV path a Cube kernel can't beat. **The B=1
lever is launch-count reduction (NPUGraph, above) + op-coalescing — not AscendC Cube
kernels.** The AscendC toolchain is validated on the 910B3 (a fused elementwise op
runs 3.8× faster than eager), but it's for elementwise/recurrence fusion, not GEMV.

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
- Our C++ forward (3666) < HF adapter `native_graph` (10219): a **software gap**.
  The B=1 latency half is **closed** — `torch.npu.NPUGraph` reaches A100 parity (see
  the NPUGraph section above); the batched-aggregate half is closable with more
  op-coalescing/fusion. Not a hardware limit, and **not** AscendC GEMV-Cube (that path
  is 33× slower for B=1 — see the correction above).
- The NPU work's value: RWKV7 runs **on NPU** (where CUDA can't), at a usable rate
  (3666 tok/s aggregate), with a correct + tested serving framework. It doesn't beat
  a heavily-optimized CUDA path, but it doesn't need to — it's a different platform.

---

## Reproduce

- **NPU op-shim** (same-code): `python npu_op_shim_bench.py` on 910B3 → 63 tok/s.
- **CUDA op-shim** (same-code): `python op_shim_cuda_bench.py` on the 5070 → 55 tok/s.
- **NPU C++ forward** (optimized): `/root/rwkv7-ascend/bench_batch.py` → 3666 tok/s @ B=64.
- **NPU NPUGraph vs eager** (B=1 latency): `python perf/bench_npugraph.py` → 5.9× at 0.1B B=1 (353 tok/s), 3.7× at 1.5B B=1 (113 tok/s).
- **CUDA HF adapter** (optimized): load the HF model + `generate` on cuda:0 → 10219 tok/s @ B=64.
- **All 6 model sizes** (NPU C++ forward, aggregate): see `serving/SERVING.md`.

> Quantization note: W8A16 (`npu_weight_quant_batchmatmul`) was PoC'd on the NPU —
> correct but **measured slower** than fp16 (fp16 matmul already memory-BW-bound).
> See `serving/QUANT.md`.
