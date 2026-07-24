# rwkv7-ascend-npu

## Current direct-fusion update (910B2C, 2026-07-13)

The opt-in Vector Core backend now fuses recurrence preparation with the fp32
rank-one state update. On the synthetic 0.1B-shaped B=1 decode row, two pinned
200-warmup/2000-iteration confirmations measure `0.650-0.662 ms/token`
(`1509.8-1537.3 tok/s`) when the recurrent cache remains resident in the captured
graph. The dynamic state-slot path measures `0.675-0.684 ms/token`
(`1462.5-1481.7 tok/s`). All 64 greedy tokens match, minimum logits cosine is
`0.999999344`, and maximum fp32 state difference is `0.00585938`.

The resident-cache row clears the provisional `1500 tok/s` target used for this
optimization pass, but that target is not a same-checkpoint, same-card Albatross
run. The dynamic cache-policy row remains below it, so this result is not a broad
Albatross parity claim. Reproduction details are in
[`vllm-rwkv-ascend/perf/ascendc/direct/README.md`](vllm-rwkv-ascend/perf/ascendc/direct/README.md).

For a reproducible three-way comparison against Qwen3.5 and the Albatross RWKV
engine, including single-card, multi-card, quality, memory, and result-table
requirements, see
[`BENCHMARK_QWEN35_ALBATROSS.md`](BENCHMARK_QWEN35_ALBATROSS.md).

**RWKV-7 inference + serving on the Huawei Ascend 910B3 NPU — fast, correct, and at
A100 parity for single-stream decode.**

This repository is the canonical monorepo for the Huawei work across Hugging
Face, vLLM, SGLang, shared quantization, kernels, and reproducible acceptance
evidence. The headline: a single
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

## Real 7.2B framework end-to-end acceptance

The three Huawei paths now also have a common **real 7.2B BF16 engine
benchmark**, separate from the small-model kernel and custom-serving numbers
above. Each measured row starts with one `Hello` token per request, greedily
generates 16 tokens, excludes one cold warm-up, and goes through the framework's
public engine API. Values are aggregate output tokens per second on one 910B3.

| framework API | B=1 | B=4 | B=8 | B4/B1 | status |
|---|---:|---:|---:|---:|---|
| Transformers `generate` | 13.15 | 47.58 | 99.22 | 3.62× | **pass** |
| vLLM V1 `LLM.generate` | 10.35 | 38.17 | 39.28 | 3.69× | **pass** |
| SGLang `Engine.generate` | 6.07 | 23.70 | 45.44 | 3.91× | **pass** |

Every measured request reproduces the shared greedy prefix
`[45, 308, 459]`; outputs within a batch are identical. The fail-closed gate
requires exact output, finite positive throughput, B4 aggregate scaling of at
least 1.25× over B1, and B8 throughput no lower than B4. These are in-process
framework-engine E2E measurements, not HTTP/network benchmarks. Reproduction
scripts and full logs live in each component's `evidence/rebuild` directory.
The vLLM and SGLang rows include the HF-derived batch-state execution shape:
vLLM's pure-decode path stays device-side instead of parsing each slot with
`Tensor.item()` in every layer, while SGLang advances the whole active batch in
one recurrence rather than looping over requests in Python.

## Contributions

- **NPUGraph B=1 decode → A100 parity.** Capture the ~960-op decode step as one
  device-side graph; per-step host-dispatch overhead removed. *(headline)*
- **Disproved the AscendC-Cube direction.** A Cube matmul kernel measures **33× slower**
  than `at::linear` for a B=1 GEMV — the lever is op-coalescing + graph, not AscendC GEMV.
- **C++ op-coalesced forward.** ~960 ops packed into one C++ call → 3666 tok/s @ B=64;
  the base the NPUGraph path captures. Includes a latent state-writeback correctness fix.
- **AscendC toolchain on the 910B3**, with the 3 non-obvious build fixes + a reusable
  `build_op.sh` (a fused elementwise op runs 3.8× faster).
- **Real framework engines.** Transformers, vLLM V1 and SGLang all load the
  7.2B checkpoint and pass the common B1/B4/B8 decode-throughput gate; vLLM's
  decode requests are now projected and recurrently updated as a true NPU batch
  instead of being serialized per request.
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
| [`rwkv7-hf-ascend/`](rwkv7-hf-ascend/) | complete HF package + Huawei runtime/oracle/evidence | [`README.md`](rwkv7-hf-ascend/README.md), [`HUAWEI_ASCEND.md`](rwkv7-hf-ascend/docs/hardware/HUAWEI_ASCEND.md) |
| [`vllm-rwkv-ascend/`](vllm-rwkv-ascend/) | vLLM V1 plugin + serving framework + C++ forward + **NPUGraph** | [`README.md`](vllm-rwkv-ascend/README.md), [`VLLM_ASCEND_PRODUCTION.md`](vllm-rwkv-ascend/VLLM_ASCEND_PRODUCTION.md) |
| [`rwkv7-sglang-ascend/`](rwkv7-sglang-ascend/) | external SGLang backend + recurrent state/cache scheduler | [`README.md`](rwkv7-sglang-ascend/README.md), [`ASCEND_FFN_QUANT.md`](rwkv7-sglang-ascend/ASCEND_FFN_QUANT.md) |
| [`ascend-optimized/`](ascend-optimized/) | C++ forward + AscendC toolchain/kernels + benches | [`rwkv7_ascend_v3.cpp`](ascend-optimized/rwkv7_ascend_v3.cpp), [`ascendc/README.md`](ascend-optimized/ascendc/README.md) |

Each component retains its source provenance and license in its own directory.
The complete `rwkv7_hf` package is vendored, so HF, vLLM, and SGLang development
and validation no longer depend on publishing changes to separate repositories.

## Unified repository validation

The consolidated tree was rebuilt on the validated Ascend 910B3 environment:

| component | focused tests | package build |
|---|---:|---:|
| `rwkv7-hf-ascend` | 18 passed | `rwkv7_hf_adapter-0.6.0` wheel |
| `vllm-rwkv-ascend` V1 plugin | 19 passed | `rwkv7_vllm_ascend-0.3.0` wheel |
| `rwkv7-sglang-ascend` | 16 passed | `sglang_rwkv7_ascend-0.2.0` wheel |
| shared W8/W4 quant layer | 19 passed, 4 skipped | import-safe shared module |

The real 7.2B HF, vLLM, and SGLang acceptance artifacts remain in their
component evidence directories. HF W8 now has a narrow exact-stack production
admission; shared vLLM/SGLang W8 and every W4 route remain fail-closed.
The serving evidence is also checked as one fail-closed contract by
[`benchmarks/verify_serving_acceptance.py`](benchmarks/verify_serving_acceptance.py);
see [`SERVING_ACCEPTANCE.md`](SERVING_ACCEPTANCE.md) for the admitted feature
matrix and explicit exclusions.

## Quick start

On a 910B3 box (CANN 8.5.0 + `torch_npu`, `rwkv7_hf` + the C++ forward reachable):

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# install the local Hugging Face runtime
python -m pip install -e ./rwkv7-hf-ascend

# reproduce the headline (NPUGraph vs eager, 5.9x at 0.1B B=1)
python vllm-rwkv-ascend/perf/bench_npugraph.py

# serve with the B=1 graph fast path
cd vllm-rwkv-ascend && python serving/serve_full.py \
  --model <model-dir> --H 32 --N 64 --L 24 --port 8001 --graph-decode
```

See each subdir's README/docs for model dims, env vars, and the full feature set.

## Ascend W8/W4 status

[`rwkv7_ascend_quant.py`](rwkv7_ascend_quant.py) and
[`rwkv7_ascend_model_quant.py`](rwkv7_ascend_model_quant.py) implement a shared,
quant-only RWKV-7 FFN checkpoint and loader path for HF, vLLM, and SGLang.
Packed payload is about 50% of FP16 for W8 and 26.6% for group-128 W4, with no
hidden dense FP16 weight copy.

The public HF W8 path now passes a real 7.2B NPUGraph backend gate on the exact
FP16 910B3 stack at B1/B4/B8: isolated active HBM is 71.48% of FP16 and median
paired speed is 1.020x-1.026x FP16. Its five production prompts have identical
greedy output, while a retained synthetic stress row discloses one rank-2
near-tied flip. A separate HF W4 value-only candidate reduces active HBM to
79.53% and runs 1.032x-1.045x FP16, but fails strict generation/logit/loss
quality gates and remains explicit/fail-closed. The shared quant-only loader
used by vLLM/SGLang also remains fail-closed. See
[`ASCEND_QUANT_ACCEPTANCE.md`](ASCEND_QUANT_ACCEPTANCE.md).

## License

Apache 2.0. Each subdir derives from upstream RWKV-7 projects
(`rwkv-rs/vllm-rwkv`, `sgl-project/sglang`, `rwkv7-hf-adapter`) under their respective
licenses; see the LICENSE files / headers within.
