# Ascend 910B3 vLLM/SGLang 30-minute serving soak

Both real 7.2B serving engines passed the default fail-closed soak in
`benchmarks/run_serving_soak.py`.

## Environment

- 1x Ascend 910B3 64 GiB; CANN 8.5.0
- PyTorch 2.9.0+cpu; torch_npu 2.9.0
- vLLM 0.18.0; vllm-ascend 0.18.0; plugin 0.3.0
- SGLang commit `d0b9689805232d8ab37789121cbc3b766b5c723e`;
  plugin 0.2.0; Transformers 5.12.1
- real `fla-hub/rwkv7-7.2B-g0a` checkpoint, BF16

The measured interval starts only after engine load and four allocator/state
warm-up cycles. Each engine repeatedly runs B1/B4/B8/B4 through its public
in-process generation API and emits eight greedy tokens per request.

## Results

| gate | vLLM V1 | SGLang |
|---|---:|---:|
| measured duration | 1801.73 s | 1801.78 s |
| cycles | 1015 | 978 |
| requests | 4314 | 4153 |
| generated tokens | 34512 | 33224 |
| canonical output | exact on every request | exact on every request |
| HBM head median | 17689 MiB | 18610 MiB |
| HBM tail median | 17689 MiB | 18610 MiB |
| HBM tail growth | **0 MiB** | **0 MiB** |
| HBM linear slope | 0.084 MiB/hour | -0.058 MiB/hour |
| observed HBM range | 17689–17691 MiB | 18609–18610 MiB |
| post-shutdown HBM | 3452 MiB | 3492 MiB |
| overall | **PASS** | **PASS** |

Every request produced the same full sequence:

```text
[45, 308, 459, 332, 22168, 32355, 4706, 22590]
```

The hard memory gates are tail growth no greater than 256 MiB and least-squares
slope no greater than 128 MiB/hour. Post-shutdown HBM must return within
256 MiB of the pre-engine idle sample.

### Throughput-tail stability

Trace collection deliberately stays enabled during this test because the soak
must prove physical recurrent-state reuse. It adds device-to-host scheduler
instrumentation, so these numbers are stability ratios, **not** replacements
for the uninstrumented E2E throughput table.

| backend | batch | head median tok/s | tail median tok/s | tail/head |
|---|---:|---:|---:|---:|
| vLLM | 1 | 8.55 | 8.72 | 1.020x |
| vLLM | 4 | 27.85 | 28.45 | 1.021x |
| vLLM | 8 | 29.57 | 30.02 | 1.015x |
| SGLang | 1 | 5.99 | 5.86 | 0.978x |
| SGLang | 4 | 22.91 | 22.86 | 0.998x |
| SGLang | 8 | 42.83 | 42.23 | 0.986x |

The admission floor is 0.80x for every batch.

### Recurrent-state lifecycle

- vLLM emitted 4331 fresh-state zero events. In 4324 events the recycled slot
  contained nonzero prior state before clearing; there were zero nonzero states
  after clearing. Physical slots 1–7 were reused.
- SGLang emitted 4170 fresh slot assignments. All 16 physical Mamba-pool slots
  were assigned fresh in multiple completed request cycles. Exact output across
  every reuse is required.

## Reproduction

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh

cd vllm-rwkv-ascend
PYTHONPATH=.:$PYTHONPATH /data/venvs/vllm/bin/python \
  ../benchmarks/run_serving_soak.py \
  --backend vllm \
  --output ../benchmarks/results/serving_soak_20260724/vllm.json

cd ../rwkv7-sglang-ascend
PYTHONPATH=.:/data/work/sglang-upstream/python:$PYTHONPATH \
  /data/venvs/sglang/bin/python ../benchmarks/run_serving_soak.py \
  --backend sglang \
  --output ../benchmarks/results/serving_soak_20260724/sglang.json
```

`vllm.log` and `sglang.log` are complete stdout/stderr captures.
`*.trace.jsonl` are the full scheduler/state traces. `script.sha256` pins the
runner used by both captures and `SHA256SUMS` authenticates every committed
artifact.

The generic vLLM Triton import warning and SGLang optional-model/kernel import
warnings in the logs are non-fatal; both selected RWKV engines complete and
shut down successfully.
