# RWKV-7 Hugging Face for Huawei Ascend

This directory is the canonical Hugging Face component of the
`rwkv7-ascend-npu` monorepo. It vendors the complete `rwkv7_hf` Python package
and the Huawei-specific runtime, tests, tools, and evidence needed to run
RWKV-7 through Transformers on Ascend without CUDA, Triton, or FLA at runtime.

## Validated stack

- 1 × Huawei Ascend 910B3, 64 GiB
- CANN 8.5.0
- PyTorch 2.9.0 + torch_npu 2.9.0
- real `fla-hub/rwkv7-7.2B-g0a` checkpoint

Unknown devices and software stacks fail closed rather than inheriting the
910B3 performance policy.

## Implemented

- `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and generation
- native recurrent cache with batch select/reorder/clone operations
- ragged batch execution and chunked prefill
- eager/native-JIT Ascend runtime selection
- independent CPU reference oracle and tensor-by-tensor NPU comparison
- W8/W4 experimental weight-only helpers and W4 channel equalization
- save/reload and a small training forward/backward smoke path

## Install and test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e .

python -m pytest -q \
  tests/test_ascend_quant.py \
  tests/test_ascend_reference_oracle.py \
  tests/test_ascend_runtime.py \
  tests/test_ascend_w4_cle.py

python tests/test_huawei_ascend_smoke.py \
  --model <native-hf-model-dir> \
  --device npu:0 --dtype bf16 --backend eager
```

## Real 7.2B acceptance

| check | result |
|---|---:|
| focused CPU tests | 18 passed |
| real BF16 load/forward/generate | passed |
| CPU oracle vs NPU tensors | 208 passed |
| minimum logits cosine | 0.99996638 |
| maximum logits NRMSE | 0.007904 |
| minimum recurrent-state cosine | 0.99984443 |
| maximum recurrent-state NRMSE | 0.0177149 |
| B=2 ragged/cache/chunk path | passed |
| allocated / peak HBM | 14,433,095,680 / 14,485,243,392 bytes |

## Real 7.2B engine throughput

`bench/run_e2e_performance.py` measures the public Transformers
`model.generate` path after one cold warm-up. On the validated 910B3, BF16,
one-token prompts and 16 generated tokens per request:

| batch | aggregate output tok/s | per-request tok/s | peak allocated HBM |
|---:|---:|---:|---:|
| 1 | 13.15 | 13.15 | 14.68 GB |
| 4 | 47.58 | 11.90 | 14.82 GB |
| 8 | 99.22 | 12.40 | 15.10 GB |

B4/B1 aggregate scaling is 3.62× and B8/B1 is 7.55×. Every request reproduced
the shared greedy prefix `[45, 308, 459]`; the fail-closed JSON reports
`status=PASS`. Reproduce with:

```bash
python bench/run_e2e_performance.py \
  --model /path/to/fla-hub-rwkv7-7.2B-g0a \
  --output bench/ascend_910b3_20260724/rebuild/e2e_performance.json
```

The clean-rebuild JSON, logs, hashes, commands, and environment metadata are in
[`bench/ascend_910b3_20260724/rebuild/`](bench/ascend_910b3_20260724/rebuild/).
The broader adapter documentation imported with the source is retained as
[`UPSTREAM_README.md`](UPSTREAM_README.md).

W8/W4 execution remains experimental. The monorepo-level production policy and
real-model latency evidence are documented in
[`../ASCEND_QUANT_ACCEPTANCE.md`](../ASCEND_QUANT_ACCEPTANCE.md).

## Provenance and license

See [`SOURCE.md`](SOURCE.md) for the exact imported revision. This component
retains the source project's MIT license in [`LICENSE`](LICENSE).
