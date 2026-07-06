# vllm-rwkv-ascend

NPU (Huawei Ascend 910B) adaptation of [`rwkv-rs/vllm-rwkv`](https://github.com/rwkv-rs/vllm-rwkv)
— the Albatross faster3a RWKV-7 engine ported into vLLM.

> **Update (post-v0.1.0): vLLM serving on Ascend is blocked** by upstream version lag
> (vllm-rwkv base = vllm **v0.23** vs vllm-ascend newest **0.22.1** which itself needs
> CANN 9.0.0; we're on CANN 8.5.0). Rather than wait, we built a **self-contained
> RWKV7 continuous-batch serving engine** — live OpenAI-compatible `/v1/completions`,
> dynamic batching, top-k/top-p/temperature. **Same-code (pure PyTorch) the 910B3 NPU ≈ an RTX 5070** (NPU ~1.15×); the optimized-path gap vs CUDA is software, not hardware.
> See **[`BENCHMARK.md`](BENCHMARK.md)** + [`serving/SERVING.md`](serving/SERVING.md).

## Docs

- **[`BENCHMARK.md`](BENCHMARK.md)** — performance: NPU vs CUDA (same-code NPU ≈ RTX 5070; optimized-path gap is software)
- **[`serving/SERVING.md`](serving/SERVING.md)** — the serving framework (what it is, how to run, current state)
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — internals deep-dive (request lifecycle, scheduler, the C++ forward)
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — dev setup, running tests, conventions
- **[`serving/QUANT.md`](serving/QUANT.md)** — quantization investigation (W8A16 measured not faster)
- **[`RELEASE_NOTES_v0.2.0.md`](RELEASE_NOTES_v0.2.0.md)** — serving framework release (latest); [`v0.1.0`](RELEASE_NOTES_v0.1.0.md) (op-shim)
- **[`LICENSE`](LICENSE)** — Apache 2.0

## Design: additive layer, zero upstream edits

We track `rwkv-rs/vllm-rwkv` as the `upstream` remote and **edit none of its
files**. All NPU work is runtime-overlaid:

```
vllm-rwkv-ascend/
├── harness/rwkv7_fast_v3a.py   # vendored standalone (Albatross faster3a) for Phase-1 testing
├── rwkv7_npu_ops.py            # ~40 rwkv7_* ops re-implemented in pure PyTorch (the shim)
├── device_patch.py             # runtime monkeypatch: first_device/zero_state/sync -> npu
├── bootstrap.py                # import this -> install shim + patch device + no-op load_extensions
└── run_phase1.py               # correctness: upstream+shim on NPU vs HF-native, cosine
```

`git fetch upstream && git merge upstream/main` stays fast-forward / conflict-free
because our changes live in separate files. If upstream renames `first_device` /
`zero_state` / an op namespace, only `device_patch.py` / `rwkv7_npu_ops.py` need a
one-line update.

## Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | RWKV-7 model produces correct logits on NPU (shim vs HF-native, cos) | ✅ done |
| 2 | Perf: C++ op-coalesced hot path (`rwkv7_ascend_v3.cpp`, 323 tok/s) + continuous batching | ✅ done (state-writeback bug fixed) |
| 3 | Full vLLM serving (OpenAI API) on NPU via `vllm-project/vllm-ascend` | ❌ **blocked** — vllm-ascend version lag; own-framework server in [`serving/`](serving/) instead (live, >2× Albatross) |

## Phase 1 run (on 910B3)

```bash
PYTHONPATH=/root/rwkv7-ascend:. python run_phase1.py <0.1b.pth> <0.1b-hf-dir>
# expect: PHASE1_RESULT cos_shim_vs_hfnative>0.99 verdict=PASS
```

## Op-shim math provenance

- layout + call sites: `harness/rwkv7_fast_v3a.py` (Albatross faster3a)
- WKV recurrence + dithering: `faster3a_2605/cuda/rwkv7_wkv_fp16_v2.cu`
- per-token TMix/CMix equations: `rwkv7_hf/native.py` (verified cos=1.0 vs official `rwkv`)

The fp16 decay `exp2(A/(1+exp2(B*w)))` and `exp(-0.606531*sigmoid(w))` are the same
function (verified equal at w=0 -> 0.7385, w=1 -> 0.6418); the shim uses the exact
Albatross form + rotator dithering to track the CUDA ground truth.
