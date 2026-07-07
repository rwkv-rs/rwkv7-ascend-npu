# Contributions

RWKV-7 inference + serving on the **Huawei Ascend 910B3 NPU** (CANN 8.5.0). This repo
consolidates the NPU work across three projects. The contributions below are each
backed by a measurement you can reproduce — see [`BENCHMARK.md`](BENCHMARK.md) for the
full numbers + methodology.

> **The one-sentence story:** single-sequence (B=1) decode on the 910B3 was stuck at
> ~60 tok/s, and the prevailing view was that closing the gap to an A100 needed
> "multi-month AscendC GEMV-Cube kernel fusion." We **disproved that direction with a
> direct measurement** (an AscendC Cube kernel is 33× *slower* than `at::linear` for a
> B=1 GEMV), **found the real lever — host dispatch overhead** — and collapsed it with
> `torch.npu.NPUGraph`, reaching **A100 parity in days, bit-exact, with zero custom
> kernels.**

---

## 1. NPUGraph decode — B=1 latency at A100 parity *(headline)*

**What.** The C++ forward issues ~960 CANN kernel launches per decode step, and each
host dispatch costs ~17µs on the 910B3 — so eager B=1 latency (~16ms) is **dispatch-
bound, not compute** (`at::linear` measures the same 0.024ms at B=1 and B=128). We
capture the whole step as a `torch.npu.NPUGraph` and replay it per token — one
device-side replay instead of 960 host dispatches — with dedicated fixed-address state
buffers decoupled from the scheduler, so continuous-batching join/leave is safe.

**Result.**
| model | B | eager | **NPUGraph** | speedup |
|---|---|---|---|---|
| 0.1B | 1 | 60 tok/s (16.7ms) | **353 tok/s (2.8ms)** | **5.9×** |
| 1.5B | 1 | 28 tok/s (35.3ms) | **113 tok/s (8.9ms)** | **3.7×** |
| 0.1B | 64 | 3852 tok/s agg | **9508 tok/s agg** | 2.5× |

**Bit-exact** vs eager (single-step maxabs=0; multi-step greedy tokens identical).
Graph latency is also **stable across runs** (immune to host contention), where eager
drifts with host load.

**Why it matters.** This is the contribution that changes the position of RWKV-7 on
Ascend: **0.1B B=1 NPUGraph (353 tok/s) matches an A100's CUDA-graph decode (368)**.
The blocker was not hardware and not a kernel problem — it was host dispatch, removable
with a stock `torch.npu` API. No custom kernels, no multi-month effort.

**Where.** [`vllm-rwkv-ascend/serving/graph_decode.py`](vllm-rwkv-ascend/serving/graph_decode.py)
(`NpuGraphDecoder`), [`vllm-rwkv-ascend/perf/bench_npugraph.py`](vllm-rwkv-ascend/perf/bench_npugraph.py),
[`vllm-rwkv-ascend/tests/test_npugraph_correctness.py`](vllm-rwkv-ascend/tests/test_npugraph_correctness.py).

---

## 2. Disproved the "AscendC GEMV-Cube" direction for B=1

**What.** The earlier consensus (documented in this repo's predecessors and in the
sibling sglang repo) was that beating `F.linear` for the B=1 projection GEMV "needs a
custom AscendC GEMV kernel." We tested it directly: built and ran the canonical AscendC
**Cube** matmul kernel (`MatmulLeakyreluCustom`, from the official Ascend CANN samples)
on this same 910B3 and measured it against `at::linear` for a 1×768×768 fp16 GEMV.

**Result.** AscendC Cube kernel = **0.79ms vs `at::linear`'s 0.024ms — 33× slower**
(and the output was wrong at M=1, a degenerate-tiling artifact).

**Why it matters.** The Cube unit is a dense-GEMM (compute) engine; a B=1 projection is
a memory-bandwidth-bound GEMV that CANN's `at::linear` already optimally handles. A
custom Cube kernel cannot beat it. This **corrects a field-wide misconception** and
redirects effort away from a multi-month dead end — toward op-coalescing + graph (the
levers that actually work). We propagated the correction to both this repo's docs and
the sglang repo's `BENCH.md`.

**Where.** [`vllm-rwkv-ascend/BENCHMARK.md`](vllm-rwkv-ascend/BENCHMARK.md) (the
correction section), [`ascend-optimized/ascendc/`](ascend-optimized/ascendc/) (the
toolchain exploration that produced the measurement).

---

## 3. C++ op-coalesced forward

**What.** Replaced the per-op Python decode path (~960 `at::` dispatches/step) with a
single C++ call (`rwkv7_ascend_v3.cpp`, libtorch → CANN `aclnn` kernels) that runs the
whole RWKV-7 layer stack in one host→device trip.

**Result.** **3666 tok/s aggregate at B=64**; the foundation the NPUGraph path captures.

**Why it matters.** This is the fewer-launch-count base that, combined with NPUGraph,
closes the gap to optimized CUDA paths. It also fixed a **latent correctness bug**: the
original computed the recurrent state correctly for one step (cos=1.0) but never wrote
it back to the Python tensors, so multi-step generation collapsed — masked by single-
step tests. The fix adds three in-place writebacks.

**Where.** [`ascend-optimized/rwkv7_ascend_v3.cpp`](ascend-optimized/rwkv7_ascend_v3.cpp),
[`vllm-rwkv-ascend/serving/serve_engine.py`](vllm-rwkv-ascend/serving/serve_engine.py).

---

## 4. AscendC toolchain validated on the 910B3 (CANN 8.5.0 aarch64)

**What.** Worked the AscendC custom-op build pipeline end-to-end on the 910B3
(op-def JSON → `msopgen` → kernel → 910b compile → `.run` install → `aclnn` call) and
nailed down three non-obvious, undocumented fixes that the prior CANN-8.5.1/x86 docs
didn't cover: (1) SoC target is `ascend910b` (patched in three places), (2) the kernel
compiler is TVM-based and needs `numpy`+`scipy` so `python3` must resolve to the
3.11.14 build that has them, (3) a CANN-8.5.0 packaging quirk requiring a manual
`binary/config` mkdir. Captured in a reproducible one-command `build_op.sh`.

**Result.** A fused elementwise op (`RwkvWexp`, Sigmoid+Adds+Muls) builds, installs,
and runs **3.8× faster than eager**. Toolchain ready for future work.

**Why it matters.** Unblocks the *correct* use of AscendC — elementwise/recurrence
fusion (not GEMV, per contribution #2). The three fixes + script are reusable for any
future custom op on this stack.

**Where.** [`ascend-optimized/ascendc/`](ascend-optimized/ascendc/) (README with the
build recipe + 5 gotchas, kernels, test harness).

---

## 5. Production-grade serving framework (self-contained, no vLLM dependency)

**What.** A complete RWKV-7 continuous-batch serving engine for Ascend: OpenAI-style
`/v1/completions` (JSON + SSE streaming + stop strings), a sampler
(greedy/temperature/top-k/top-p), a `SlottedScheduler` for persistent batched recurrent
state (grow on join, swap-remove on leave, in-place evolution — no per-step `cat`), a
multi-worker front-end router, `/metrics`, a Docker image, and a 23-test pytest suite.

**Result.** Live serving produces coherent text; with `--graph-decode`, single-request
decode runs at A100 parity; batched aggregate 3666 tok/s. Continuous-batching correctness
verified bit-exact (mid-flight join, staggered completion, concurrent batch).

**Why it matters.** Full vLLM serving of RWKV-7 on Ascend is currently **blocked**
(`vllm-rwkv` tracks vllm v0.23; `vllm-ascend` tops out at 0.22.1 on CANN 8.5.0 — no
matching version). This framework serves RWKV-7 on NPU *now*, without waiting for the
upstream version convergence.

**Where.** [`vllm-rwkv-ascend/serving/`](vllm-rwkv-ascend/serving/) + [`vllm-rwkv-ascend/tests/`](vllm-rwkv-ascend/tests/).

---

## 6. Cross-platform validation — same-code NPU ≈ RTX 5070; positioned vs A100

**What.** Ran the **same** pure-PyTorch RWKV-7 code on both the 910B3 NPU and an RTX
5070 (CUDA) to isolate hardware from the software-gap confound, and assembled the
A100 comparison from the rwkv7-hf-adapter bench.

**Result.** Same-code: **910B3 NPU 63 ≈ RTX 5070 55 tok/s** (NPU ~1.15×) — the hardware
is comparable. The larger optimized-path gap vs CUDA is **software**, not a hardware
ceiling. With NPUGraph, 0.1B B=1 reaches A100 parity (353 vs 368).

**Why it matters.** Corrects the "the NPU is a 3× slower chip" reading — it isn't; the
gap is closable software (op-coalescing + graph), which we then closed for B=1.

**Where.** [`vllm-rwkv-ascend/BENCHMARK.md`](vllm-rwkv-ascend/BENCHMARK.md), the
`op_shim_*_bench.py` scripts.

---

## 7. SGLang port + Triton-ascend WKV

**What.** A second, independent serving path: RWKV-7 on SGLang for the 910B3, with a
fused **Triton-ascend WKV** recurrence kernel (2× over pure-torch) under SGLang's
built-in cuda graph.

**Result.** 1.5B B=1 ~66.6 tok/s.

**Why it matters.** The cross-repo comparison (this path's per-op Python model vs the
C++ op-coalesced forward) **isolated op-coalescing as the lever** — both paths use a
graph, but the coalesced one (114 tok/s) is 1.7× faster, pinpointing per-step op-count
as the difference. This is what pointed us at NPUGraph + op-coalescing.

**Where.** [`rwkv7-sglang-ascend/`](rwkv7-sglang-ascend/) (`ascend_port/model.py`,
`wkv_triton.py`, `BENCH.md`).

---

## Contribution map

| contribution | subdir | headline number |
|---|---|---|
| NPUGraph B=1 decode | `vllm-rwkv-ascend/` | 0.1B B=1: 60→353 tok/s (5.9×), A100 parity |
| AscendC-Cube disproof | `vllm-rwkv-ascend/` + `ascend-optimized/` | Cube kernel 33× slower than `at::linear` for B=1 GEMV |
| C++ op-coalesced forward | `ascend-optimized/` + `vllm-rwkv-ascend/` | 3666 tok/s @ B=64 |
| AscendC toolchain | `ascend-optimized/ascendc/` | fused op 3.8× faster; 3 reusable fixes |
| Serving framework | `vllm-rwkv-ascend/serving/` | live `/v1/completions`, 23 tests |
| Cross-platform validation | `vllm-rwkv-ascend/` | same-code NPU ≈ RTX 5070 |
| SGLang port + Triton WKV | `rwkv7-sglang-ascend/` | 1.5B B=1 ~66.6 tok/s; isolated op-coalescing |

Full data + reproduce: [`BENCHMARK.md`](BENCHMARK.md).
