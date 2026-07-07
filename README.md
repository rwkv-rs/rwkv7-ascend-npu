# rwkv7-ascend-npu

**RWKV-7 inference + serving on the Huawei Ascend 910B3 NPU — fast, correct, and at
A100 parity for single-stream decode.**

This repo consolidates the NPU work across three projects. The headline: a single
`torch.npu.NPUGraph` capture of the decode step brings **B=1 latency from ~60 tok/s to
353 tok/s (5.9×), matching an A100's CUDA-graph decode** — bit-exact, zero custom
kernels, and it corrects the prior view that this needed multi-month AscendC kernel
fusion (it doesn't; a Cube kernel is 33× *slower* for B=1).

## Headline results (910B3, CANN 8.5.0)

| | eager | **NPUGraph** | speedup | vs A100 |
|---|---|---|---|---|
| 0.1B B=1 | 60 tok/s | **353 tok/s** | **5.9×** | **parity** (A100 = 368) |
| 1.5B B=1 | 28 tok/s | **113 tok/s** | **3.7×** | 0.69× (A100 = 164.5) |
| 0.1B B=64 (agg) | 3852 | **9508** | 2.5× | — |

Bit-exact vs eager. Full tables + methodology: [`BENCHMARK.md`](BENCHMARK.md).

## Contributions

- **NPUGraph B=1 decode → A100 parity.** Capture the ~960-op decode step as one
  device-side graph; per-step host-dispatch overhead removed. *(headline)*
- **Disproved the AscendC-Cube direction.** A Cube matmul kernel measures **33× slower**
  than `at::linear` for a B=1 GEMV — the lever is op-coalescing + graph, not AscendC GEMV.
- **C++ op-coalesced forward.** ~960 ops packed into one C++ call → 3666 tok/s @ B=64;
  the base the NPUGraph path captures. Includes a latent state-writeback correctness fix.
- **AscendC toolchain on the 910B3**, with the 3 non-obvious build fixes + a reusable
  `build_op.sh` (a fused elementwise op runs 3.8× faster).
- **Production serving framework** (OpenAI `/v1/completions`, streaming, sampler,
  `SlottedScheduler`, 23 tests) — serves RWKV-7 on NPU now, while full vLLM serving is
  blocked on an upstream version mismatch.
- **Cross-platform validation:** same-code **910B3 ≈ RTX 5070** (NPU ~1.15×) — the
  hardware is comparable; the optimized-path gap is software, which we close for B=1.
- **SGLang port + Triton-ascend WKV** (2× over pure-torch) — a second serving path whose
  cross-repo comparison isolated op-coalescing as the lever.

Detail + the narrative: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

## Key finding

The B=1 bottleneck on the 910B3 is **host dispatch overhead** (~960 CANN kernel launches
per step, ~17µs each) — not compute (`at::linear` measures identically at B=1 and B=128).
The lever is **`torch.npu.NPUGraph`** (graph replay) + **op-coalescing** (fewer launches).
**Not** AscendC Cube kernels — those are 33× slower for B=1 (the Cube unit is a dense-GEMM
engine; a B=1 projection is a memory-BW-bound GEMV that `at::linear` already optimally
handles). See [`BENCHMARK.md`](BENCHMARK.md) §3-4.

## Repo layout

| subdir | what | entry points |
|---|---|---|
| [`vllm-rwkv-ascend/`](vllm-rwkv-ascend/) | serving framework + C++ forward + **NPUGraph** | [`serving/graph_decode.py`](vllm-rwkv-ascend/serving/graph_decode.py), [`BENCHMARK.md`](vllm-rwkv-ascend/BENCHMARK.md), [`serving/SERVING.md`](vllm-rwkv-ascend/serving/SERVING.md) |
| [`rwkv7-sglang-ascend/`](rwkv7-sglang-ascend/) | SGLang port + Triton-ascend WKV | [`ascend_port/model.py`](rwkv7-sglang-ascend/ascend_port/model.py), [`BENCH.md`](rwkv7-sglang-ascend/BENCH.md) |
| [`ascend-optimized/`](ascend-optimized/) | C++ forward + AscendC toolchain/kernels + benches | [`rwkv7_ascend_v3.cpp`](ascend-optimized/rwkv7_ascend_v3.cpp), [`ascendc/README.md`](ascend-optimized/ascendc/README.md) |

`vllm-rwkv-ascend/` and `rwkv7-sglang-ascend/` preserve each project's full git history
(imported via `git subtree`). `ascend-optimized/` holds the Ascend-specific work
extracted from an rwkv7-hf-adapter fork (the upstream `rwkv7_hf` package is referenced by
the C++ forward but not vendored).

## Quick start

On a 910B3 box (CANN 8.5.0 + `torch_npu`, `rwkv7_hf` + the C++ forward reachable):

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# reproduce the headline (NPUGraph vs eager, 5.9x at 0.1B B=1)
python vllm-rwkv-ascend/perf/bench_npugraph.py

# serve with the B=1 graph fast path
cd vllm-rwkv-ascend && python serving/serve_full.py \
  --model <model-dir> --H 32 --N 64 --L 24 --port 8001 --graph-decode
```

See each subdir's README/docs for model dims, env vars, and the full feature set.

## License

Apache 2.0. Each subdir derives from upstream RWKV-7 projects
(`rwkv-rs/vllm-rwkv`, `sgl-project/sglang`, `rwkv7-hf-adapter`) under their respective
licenses; see the LICENSE files / headers within.
