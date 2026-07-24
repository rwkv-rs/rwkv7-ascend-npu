# RWKV-7 for SGLang on Ascend

An installable, out-of-tree RWKV-7 recurrent backend for SGLang on Huawei
Ascend. The production path no longer copies an overlay into `site-packages` or
rewrites seven upstream files. It uses SGLang's public external-model and
linear-attention registries, plus two auditable upstream patches: all-linear
KV-pool sizing and an optional lightweight hybrid-wrapper registration hook.

## Scope and truthful status

Implemented and CPU/unit gated:

- standard HF `config.json`, tokenizer and safetensors consumed by SGLang;
- SGLang `MambaPool` recurrent state: two token shifts plus fp32 WKV state;
- continuous/dynamic batching through physical request slots;
- packed variable-length prefill and arbitrary chunk continuation;
- cancellation/slot clearing, copy/reorder, CPU offload and restore adapter;
- eager decode/extend metadata without an `sgl_kernel_npu` dependency;
- fp32 recurrent math with bf16/fp16 model activations;
- independent recurrence oracle and chunk/slot lifecycle tests.

Not claimed by this branch without a dated hardware artifact:

- W8/W4 end-to-end model speed or memory acceptance;
- tensor/pipeline parallel correctness;
- radix-prefix state snapshots;
- speculative decoding;
- CUDA/NPU graph capture (the production wrapper passes `--disable-cuda-graph`);
- long-running concurrency throughput.

`versions.env` is the only supported source/ABI matrix. Current pins:
SGLang `d0b9689`, optional sgl-kernel-npu `d661747`, torch/torch_npu 2.9.0,
Transformers 5.12.1, CANN 8.5.0.

## Layout

```text
sglang_rwkv7_ascend/
  configuration_rwkv7.py  HF config + exact MambaPool state shape
  backend.py               Ascend recurrent backend + all-linear no-op backend
  models/rwkv7.py          SGLang model and HF safetensors loader
  kernels/wkv.py           correctness-first torch recurrence
  state_cache.py           copy/reorder/offload/restore state contract
patches/
  sglang-all-linear-pool.patch
  sglang-external-linear-pure-torch-npu.patch
scripts/
  install_production.sh
  verify_install.py
  serve_production.sh
```

The old `ascend_port/sglang_overlay` and `deploy_wiring.py` remain only as
historical evidence. Do not use them for new deployments.

## Reproducible install

The server must already contain these pinned checkouts:

```text
/data/work/sglang-upstream
/data/work/sglang-rwkv-ascend
```

Then:

```bash
cd /data/work/sglang-rwkv-ascend/rwkv7-sglang-ascend
bash scripts/install_production.sh
```

The installer rejects commit skew, applies both SGLang patches idempotently,
and installs into
`/data/venvs/sglang`. It does not mutate the upstream SGLang pyproject or copy
plugin files into the SGLang tree. `sgl-kernel-npu` is not required on Atlas
A2/910B; set `BUILD_SGL_KERNEL_NPU=1` only on a supported toolchain.

## Tests

No NPU reservation:

```bash
/data/venvs/sglang/bin/python scripts/verify_install.py
/data/venvs/sglang/bin/python -m pytest -q \
  tests/test_wkv_unit.py tests/test_wkv_correctness.py \
  tests/production/test_chunked_dynamic_state.py
```

Hardware smoke (only after unit gates):

```bash
/data/venvs/sglang/bin/python -m pytest -q tests/test_wkv_triton_vs_torch.py
```

Real 7.2B engine acceptance (one command, emits JSON plus the server log):

```bash
bash scripts/run_engine_acceptance.sh \
  /data/models/fla-hub-rwkv7-7.2B-g0a /data/work/sglang-rwkv-acceptance.json
```

It submits two different prompts concurrently, makes the long prompt cross
multiple prefill chunks, then repeats the short prompt after slot release to
detect stale recurrent state on slot reuse. It also gates the shared `Hello`
three-token oracle against the vLLM dense token IDs and writes a `.sha256`
manifest for the final JSON, server log, and backend trace.

The clean-rebuild Atlas 910B3 acceptance artifact is committed under
`evidence/rebuild/` (the older independent capture remains under
`evidence/910b3/`). Its JSON is fail-closed (`passed=true` only when every
dynamic-batch, mixed-mode, chunk-continuation, physical-slot-reuse, and shared
dense-token-oracle gate succeeds). The clean rebuild observed 48 backend
forwards, a real batch of two, a mixed decode+prefill forward, continuation of
one physical slot across prefix lengths 0/64/.../481, and deterministic reuse
of released slot 4. `SHA256SUMS` authenticates the JSON, server log, backend
trace, and worker response.

## Real 7.2B end-to-end throughput

`scripts/run_e2e_performance.py` measures the actual SGLang
`Engine.generate` path after one cold warm-up. The pure-torch kernel borrows
HF's batch-state execution shape: time remains sequential, but all independent
requests advance together instead of a Python loop over batch entries.

| batch | aggregate output tok/s | per-request tok/s | B1 scaling |
|---:|---:|---:|---:|
| 1 | 6.07 | 6.07 | 1.00× |
| 4 | 23.70 | 5.93 | 3.91× |
| 8 | 45.44 | 5.68 | 7.49× |

The real 7.2B BF16 run used 16 greedy output tokens per one-token request.
Every row reproduced `[45, 308, 459]`, outputs within each batch were
identical, B4 aggregate scaling exceeded the 1.25× gate, B8 exceeded B4, and
the fail-closed JSON reports `status=PASS`.

```bash
python scripts/run_e2e_performance.py \
  --model /path/to/fla-hub-rwkv7-7.2B-g0a \
  --output evidence/rebuild/e2e_performance.json
```

This is an in-process SGLang engine benchmark. Long-running concurrency and
HTTP/network throughput remain separate release gates.

## Serve

```bash
bash scripts/serve_production.sh /data/models/rwkv7-hf \
  --max-running-requests 32 --chunked-prefill-size 512
```

The wrapper registers `Rwkv7Config`, `RWKV7ForCausalLM`, the recurrent backend,
and the process-local `ascend` no-op full-attention backend before SGLang parses
arguments. Radix prefix caching is
deliberately off; active-request state caching and chunk continuation remain on.

## Ascend W8/W4 FFN seam (production disabled)

The default-off, fail-closed loader and `npu_weight_quant_batchmatmul` dispatch
contract is documented in [ASCEND_FFN_QUANT.md](ASCEND_FFN_QUANT.md). It is a
raw-kernel-candidate integration seam only; no quantized SGLang E2E acceptance
is claimed yet.
