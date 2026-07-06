# rwkv7-ascend-npu

RWKV-7 inference + serving on **Huawei Ascend 910B3 NPU** (CANN 8.5.0) — consolidated
NPU work. Three independent projects, one repo:

| subdir | what | headline |
|---|---|---|
| [`vllm-rwkv-ascend/`](vllm-rwkv-ascend/) | self-contained RWKV7 continuous-batch serving framework (OpenAI `/v1/completions`, streaming, sampler, `SlottedScheduler`) + the C++ op-coalesced forward | **`torch.npu.NPUGraph` decode → B=1 latency at A100 parity** (0.1B 60→353 tok/s, **5.9×**; 1.5B 30→113 tok/s, **3.7×**) |
| [`rwkv7-sglang-ascend/`](rwkv7-sglang-ascend/) | RWKV-7 serving via SGLang (per-op Python model + Triton-ascend WKV + SGLang's built-in cuda graph) | 1.5B B=1 ~66.6 tok/s |
| [`ascend-optimized/`](ascend-optimized/) | Ascend optimization R&D: the C++ op-coalesced forward (`rwkv7_ascend_v3.cpp`), AscendC custom-kernel exploration (`ascendc/`), benches, tests, the AscendC toolchain | C++ forward 3666 tok/s @ B=64; AscendC toolchain validated (fused op 3.8× faster) |

## Key finding

The single-sequence (B=1) bottleneck on the 910B3 is **host dispatch overhead**
(~960 CANN kernel launches per decode step, ~17µs each) — not compute (`at::linear`
latency is identical at B=1 and B=128). The lever is **`torch.npu.NPUGraph`** (record
the step as one device-side graph, replay with a single host launch) + **op-coalescing**
(pack ops into fewer launches).

**Not** AscendC Cube kernels: a custom AscendC Cube matmul kernel measures **33×
slower** than `at::linear` for a B=1 GEMV (the Cube unit is a dense-GEMM engine; a B=1
projection is a memory-BW-bound GEMV). See
[`vllm-rwkv-ascend/BENCHMARK.md`](vllm-rwkv-ascend/BENCHMARK.md).

## vs A100

| model | B | 910B3 NPUGraph | A100 native_graph |
|---|---|---|---|
| 0.1B | 1 | 353 tok/s (2.8ms) | 368 tok/s (2.71ms) — **parity** |
| 1.5B | 1 | 113 tok/s (8.9ms) | 164.5 tok/s (6.08ms) |

At small compute (0.1B B=1) NPUGraph reaches A100 parity; at larger compute A100's raw
kernel speed leads (the next lever there is op-coalescing/fusion).

## Layout note

`vllm-rwkv-ascend/` and `rwkv7-sglang-ascend/` preserve each project's full git history
(imported via `git subtree`). `ascend-optimized/` holds the Ascend-specific work
extracted from an rwkv7-hf-adapter fork (the upstream `rwkv7_hf` package is referenced
by the C++ forward but not vendored here).

License: Apache 2.0 (each subdir inherits its upstream's license).
