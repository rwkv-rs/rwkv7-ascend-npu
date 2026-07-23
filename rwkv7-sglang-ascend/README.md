# RWKV-7 for SGLang on Ascend

An installable, out-of-tree RWKV-7 recurrent backend for SGLang on Huawei
Ascend. The production path no longer copies an overlay into `site-packages` or
rewrites seven upstream files. It uses SGLang's public external-model and
linear-attention registries, plus one auditable upstream patch for an all-linear
KV-pool division-by-zero.

## Scope and truthful status

Implemented and CPU/unit gated:

- standard HF `config.json`, tokenizer and safetensors consumed by SGLang;
- SGLang `MambaPool` recurrent state: two token shifts plus fp32 WKV state;
- continuous/dynamic batching through physical request slots;
- packed variable-length prefill and arbitrary chunk continuation;
- cancellation/slot clearing, copy/reorder, CPU offload and restore adapter;
- decode graph-safe fixed-shape metadata through `AscendMambaAttnBackendBase`;
- fp32 recurrent math with bf16/fp16 model activations;
- independent recurrence oracle and chunk/slot lifecycle tests.

Not claimed by this branch without a dated hardware artifact:

- W8/W4 end-to-end model speed or memory acceptance;
- tensor/pipeline parallel correctness;
- radix-prefix state snapshots;
- speculative decoding;
- long-running concurrency throughput.

`versions.env` is the only supported source/ABI matrix. Current pins:
SGLang `d0b9689`, sgl-kernel-npu `d661747`, torch/torch_npu 2.9.0,
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
/data/work/sglang-upstream/third_party/sgl-kernel-npu
/data/work/sglang-rwkv-ascend
```

Then:

```bash
cd /data/work/sglang-rwkv-ascend/rwkv7-sglang-ascend
bash scripts/install_production.sh
```

The installer rejects commit skew, applies the small all-linear patch
idempotently, builds `sgl-kernel-npu`, and installs into
`/data/venvs/sglang`. It does not mutate the upstream SGLang pyproject or copy
plugin files into the SGLang tree.

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

## Serve

```bash
bash scripts/serve_production.sh /data/models/rwkv7-hf \
  --max-running-requests 32 --chunked-prefill-size 512
```

The wrapper registers `Rwkv7Config`, `RWKV7ForCausalLM`, the recurrent backend,
and `rwkv7_ascend` before SGLang parses arguments. Radix prefix caching is
deliberately off; active-request state caching and chunk continuation remain on.
