# Benchmark

All numbers measured on **1 × Huawei Ascend 910B3** (20 cube + 40 vector cores, 62 GB
HBM), **CANN 8.5.0 aarch64**, `torch_npu` 2.9.0, Python 3.11.14 — unless labeled
otherwise. A100 figures are from the `rwkv7-hf-adapter` repo's `bench/results.jsonl`
(`rwkv7_forward_token` + `native_graph`, i.e. CUDA-graph decode). RTX 5070 figures are
local (sm_120, torch 2.11+cu128).

The decode benches use **random-init weights** (pure speed — the compute is identical to
real weights; correctness is verified separately, bit-exact). The 910B3 here is a
**shared box** (another tenant's sglang holds ~40 GB HBM), which is exactly why the
NPUGraph result matters — see the stability note.

---

## 1. Headline — NPUGraph vs eager decode

`torch.npu.NPUGraph` captures the ~960-op C++ decode step as one device-side graph and
replays it per token, removing the per-step host-dispatch overhead. (`perf/bench_npugraph.py`)

| model | B | eager ms/step | graph ms/step | eager tok/s | **graph tok/s** | speedup |
|---|---|---:|---:|---:|---:|---:|
| 0.1B | 1 | 16.7 | 2.8 | 60 | **353** | **5.9×** |
| 0.1B | 8 | 16.3 | 4.1 | 492 | 1969 | 4.0× |
| 0.1B | 64 | 16.6 | 6.7 | 3852 | **9508** | 2.5× |
| 1.5B | 1 | 35.3 | 8.9 | 28 | **113** | **3.7×** |
| 1.5B | 8 | 34.7 | 11.3 | 230 | 711 | 3.1× |
| 1.5B | 64 | 32.4 | 23.6 | 1978 | 2708 | 1.4× |

**Correctness:** bit-exact vs eager — single-step cosine 0.999995, maxabs 0; multi-step
greedy token sequences identical (`tests/test_npugraph_correctness.py`, 2/2 pass).

### Captured token embedding (2026-07-13)

The serving path previously built a device token tensor and ran the embedding lookup
outside NPUGraph on every token.  A fixed-address int64 token buffer now moves the
embedding lookup into the captured graph while preserving the scheduler-state copies
needed for safe B=1/B>1 transitions.

On an Ascend 910B3 with CANN 8.5.0 and torch_npu 2.9.0, the self-contained 0.1B-shape
probe (`L=12,H=12,N=64,vocab=65536`, 100 iterations) reports:

| path | ms/step | tok/s | vs legacy production |
|---|---:|---:|---:|
| pure graph replay | 2.580 | 387.6 | — |
| legacy production (external embedding + state copies) | 2.871 | 348.3 | 1.00x |
| captured embedding + state copies | **2.626** | **380.8** | **1.09x** |

Legacy and captured paths produce bit-exact logits and recurrent state over a
multi-token check.  Reproduce without a checkpoint or Transformers install:

```bash
python vllm-rwkv-ascend/perf/bench_graph_overhead.py --warmup 10 --iterations 100
```

This is a same-shape synthetic performance A/B, not a model-quality result.  It does
not replace the real-checkpoint rows above.

**Two things to notice in the table:**
- **Eager latency is B-independent** (0.1B: 16.7ms at B=1, 16.6ms at B=64; 1.5B: 35.3ms
  vs 32.4ms). Decode is **dispatch-bound**, not compute — the GEMV itself is ~free.
- **Graph latency grows with B** (0.1B: 2.8→6.7ms). Once dispatch is removed, the real
  compute (which scales with B) becomes visible. Speedup shrinks at large B because
  compute is then the larger share.

---

## 2. vs A100 (CUDA-graph decode)

Same workload, both under graph replay (NPUGraph on 910B3, `native_graph` on A100):

| model | B | 910B3 NPUGraph | A100 native_graph | ratio |
|---|---|---|---|---|
| 0.1B | 1 | 2.8ms / 353 tok/s | 2.71ms / 368 tok/s | **≈ parity (0.97×)** |
| 1.5B | 1 | 8.9ms / 113 tok/s | 6.08ms / 164.5 tok/s | 0.69× |
| 2.9B | 1 | — | 9.83ms / 101.8 tok/s | — |

At small compute (0.1B B=1) both are dispatch-bound and NPUGraph reaches **parity**. At
larger compute (1.5B+) A100's raw kernel speed leads — the remaining gap is per-kernel
execution, not dispatch. Closing it is an **op-coalescing/fusion** task (fewer/cheaper
kernels), explicitly **not** an AscendC-GEMV task (see §4).

---

## 3. Root cause — `at::linear` latency is B-independent

Direct measurement of a single fp16 projection (`F.linear`, the projection GEMV):

| shape | latency |
|---|---|
| B=1, 768×768 | **0.0236ms** |
| B=1, 2048×2048 | 0.0239ms |
| B=128, 768×768 | **0.0238ms** |

B=1 and B=128 are **identical** → the matmul is pure dispatch overhead (~24µs); the
GEMV compute is negligible. This is why collapsing launches (NPUGraph) is the lever, and
why a "faster GEMV" is irrelevant — there is no GEMV compute to save at B=1.

---

## 4. Negative result — AscendC Cube kernels are 33× *slower* for B=1 GEMV

The previously-pursued direction ("beat `F.linear` with a custom AscendC GEMV/Cube
kernel") is **counterproductive for B=1**. Built + ran the canonical AscendC Cube matmul
sample (`MatmulLeakyreluCustom`) on this 910B3, 1×768×768 fp16:

| path | B=1 GEMV latency | vs `at::linear` |
|---|---|---|
| `at::linear` (CANN) | 0.024ms | baseline |
| AscendC Cube matmul kernel | 0.79ms | **33× slower** (+ output wrong: cos=0.32, M=1 tiling artifact) |

The Cube unit is a dense-GEMM (compute) engine; a B=1 projection is a memory-BW-bound
GEMV that `at::linear` already optimally handles. **Do not pursue AscendC Cube/GEMV
kernels for the B=1 path.** (AscendC is still useful for *elementwise/recurrence*
fusion — see `ascend-optimized/ascendc/`.)

---

## 5. Same-code NPU ≈ RTX 5070 (hardware isolation)

The same pure-PyTorch RWKV-7 code (`rwkv7_*` op-shim) on both backends — only the device
differs — so this isolates **hardware** from the software-gap confound:

| backend | tok/s (0.1B, 16-token forward) |
|---|---|
| **910B3 NPU** | **63** |
| **RTX 5070 Laptop** (sm_120) | **55** |
| ratio | **NPU ~1.15× CUDA** |

The 910B3 is **not** a slow chip. The larger gap seen against heavily-optimized CUDA
paths (e.g. the HF adapter's `native_graph`) is **software** (op fusion + graph), not a
hardware ceiling — and NPUGraph + op-coalescing close it (§1, §2).

---

## 6. C++ op-coalesced forward — batched aggregate

The C++ forward (`rwkv7_ascend_v3.cpp`, ~960 ops packed into one call) at B=64,
**0.1B**: **3666 tok/s aggregate** (≈1.44ms/step). All six model sizes (0.1B–13.3B) lead
in batched aggregate on this forward; see `vllm-rwkv-ascend/serving/SERVING.md` for the
per-size table. With NPUGraph the 0.1B B=64 aggregate rises to **9508 tok/s** (§1).

---

## 7. Cross-repo: op-coalescing is the lever

Both serving paths use a graph; the difference is per-step op-count:

| path (1.5B fp16, B=1, graph decode) | tok/s | per-step ops |
|---|---:|---|
| `rwkv7-sglang-ascend` (per-op Python model + sglang cuda graph) | ~66.6 | ~15-20/layer (Python dispatch each) |
| `vllm-rwkv-ascend` (C++ op-coalesced + NPUGraph) | 114 | 1 call (coalesced) |
| A100 `native_graph` | 164.5 | fused + graph |

The 1.7× gap between the two NPU paths is **op-count**, not the graph (both have one) and
not the GEMV (§4). This is the observation that pointed us at NPUGraph + op-coalescing.

---

## Methodology notes

- **Random-init weights** for all decode speed benches (compute is weight-value-
  independent). Correctness verified separately on real checkpoints (greedy bit-exact vs
  HF-native; scheduler mid-flight-join/concurrent batch bit-exact).
- **Stability:** NPUGraph replay latency is stable across runs (±<1%); eager latency
  drifts ~10-20% with host load on this shared box. NPUGraph is immune because replay is
  device-side.
- **tok/s** is per-sequence for B=1 rows and **aggregate** (B × steps/s) for B>1 rows.
- **Warmup:** 3-5 untimed steps before timing; 50 timed iterations per data point.

---

## Reproduce

On a 910B3 box (CANN 8.5.0 + torch_npu, `rwkv7_hf` + the C++ forward reachable):

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# Headline: NPUGraph vs eager (§1)
python vllm-rwkv-ascend/perf/bench_npugraph.py

# Correctness (§1) — bit-exact vs eager
python -m pytest vllm-rwkv-ascend/tests/test_npugraph_correctness.py -v

# at::linear B-independence (§3): single fp16 projection, sweep B
python -c "
import torch, torch_npu, torch.nn.functional as F, time
dev='npu:0'; W=torch.randn(768,768,device=dev).half()
def bench(b):
    x=torch.randn(b,768,device=dev).half()
    for _ in range(10): F.linear(x,W)
    torch.npu.synchronize(); t=time.time()
    for _ in range(500): F.linear(x,W)
    torch.npu.synchronize(); return (time.time()-t)/500*1000
print('B=1:', round(bench(1),4), 'ms | B=128:', round(bench(128),4), 'ms')
"

# Same-code NPU vs 5070 (§5): npu_op_shim_bench.py on 910B3, op_shim_cuda_bench.py on 5070
```

Raw bench scripts: [`vllm-rwkv-ascend/perf/bench_npugraph.py`](vllm-rwkv-ascend/perf/bench_npugraph.py),
the `op_shim_*_bench.py` in `vllm-rwkv-ascend/`, and `ascend-optimized/bench_*.py`. A100
reference data: `rwkv7-hf-adapter` repo, `bench/results.jsonl`.
