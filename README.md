# vllm-rwkv-ascend

RWKV-7 inference + serving on **Huawei Ascend 910B NPU**. Two things live here:

1. **A self-contained RWKV7 continuous-batch serving framework** (`serving/`) — the
   active, usable deliverable. Live OpenAI `/v1/completions`, streaming + stop
   strings, sampler, multi-worker router, error isolation, Prometheus metrics, a
   Docker image, and a 21-test pytest suite. Built because full vLLM serving is
   blocked on Ascend (see [why](#why-a-separate-serving-framework-the-vllm-blocker)).
2. **A dormant op-shim** (root `.py` + `harness/`) — the Phase-1 `vllm-rwkv` port
   (pure-PyTorch `rwkv7_*` ops). Kept for the day `vllm-ascend` catches up to
   `vllm-rwkv`'s base.

> **Same-code (pure PyTorch) the 910B3 NPU ≈ an RTX 5070** (NPU ~1.15×); the
> optimized-path gap vs CUDA is **software, not hardware**. See
> [`BENCHMARK.md`](BENCHMARK.md).

## Docs

- **[BENCHMARK.md](BENCHMARK.md)** — performance: NPU vs CUDA (same-code NPU ≈ RTX 5070; optimized-path gap is software)
- **[serving/SERVING.md](serving/SERVING.md)** — the serving framework (what it is, how to run, current state, ~70% production-ready)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — internals deep-dive (request lifecycle, SlottedScheduler, the C++ forward, scaling)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — dev setup, running tests, conventions
- **[serving/QUANT.md](serving/QUANT.md)** — quantization investigation (W8A16 measured *not faster* on this stack)
- **[perf/README.md](perf/README.md)** — the C++ op-coalesced forward (the state-writeback correctness fix)
- **[RELEASE_NOTES_v0.2.0.md](RELEASE_NOTES_v0.2.0.md)** — serving framework release; [v0.1.0](RELEASE_NOTES_v0.1.0.md) (op-shim)
- **[LICENSE](LICENSE)** — Apache 2.0

## Repo layout

```
serving/        the serving framework (ACTIVE): RWKV7Engine, SlottedScheduler,
                AsyncServer, serve_router, sampler, Dockerfile, run_cluster.sh
perf/           the C++ op-coalesced forward (rwkv7_ascend_v3.cpp) — shared by serving
tests/          pytest suite (11 sampler unit + 10 NPU integration — 21/21 pass)
harness/        vendored Albatross standalone (used by the op-shim + the benches)
.github/        CI workflow (sampler unit tests; NPU tests auto-skip)
rwkv7_npu_ops.py, device_patch.py, bootstrap.py, run_phase1.py
                the DORMANT vllm-rwkv op-shim path (Phase 1)
op_shim_cuda_bench.py, npu_op_shim_bench.py
                same-code NPU-vs-CUDA benches (see BENCHMARK.md)
contrib/        PROPOSAL_to_vllm_rwkv.md (decided against — see its STATUS header)
```

## Why a separate serving framework (the vLLM blocker)

Full vLLM serving of RWKV7 on Ascend is currently **not possible** — upstream
version lag:

```
vllm-rwkv (rwkv-rs)   base = vllm v0.23.1rc0   (tracks vllm main)
vllm-ascend (Huawei)  newest = 0.22.1rc1       (needs CANN 9.0.0 + torch_npu 2.10)
our 910B3             CANN 8.5.0 + torch_npu 2.9.0
```

No `vllm-ascend` release matches `vllm-rwkv`'s v0.23 base. Rather than wait,
`serving/` is a self-contained engine that serves RWKV7 on NPU **now**. The
op-shim (root) stays for when `vllm-ascend` catches up.

## The dormant op-shim (Phase 1)

The root `.py` files re-implement `vllm-rwkv`'s ~40 CUDA `rwkv7_*` ops in **pure
PyTorch** (`rwkv7_npu_ops.py`), plus runtime device patches (`device_patch.py`) and
a bootstrap loader (`bootstrap.py`) — an additive layer over `vllm-rwkv` with zero
edits to its files. Phase 1 verified: op-shim vs HF-native **cos=0.99998**,
argmax 100% match. This path is dormant (full vLLM serving needs `vllm-ascend`,
blocked above). The fp16 decay `exp2(A/(1+exp2(B*w)))` + rotator dithering track
the Albatross faster3a CUDA ground truth.

## License

Apache 2.0 ([LICENSE](LICENSE)). Derives from [`rwkv-rs/vllm-rwkv`](https://github.com/rwkv-rs/vllm-rwkv) (Apache-2.0).
