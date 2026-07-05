# Contributing

Thanks for helping. This covers dev setup, the repo layout, running tests, where to
add things, and conventions.

## Dev environment

You need an **Ascend NPU box** (the integration tests won't run without one). The
reference setup (the 910B3 this was built on):

- CANN 8.5.0, `torch_npu==2.9.0`, `torch==2.9.0+cpu`, Python 3.11
- `pip install fastapi uvicorn httpx pytest`
- the HF adapter `rwkv7_hf/` (from `rwkv7-hf-adapter-ascend`) on `PYTHONPATH`
- the RWKV vocab `rwkv_vocab_v20230424.txt`
- a model checkpoint (e.g. `rwkv7-g1d-0.1b-hf`)

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
# serve_engine.py / serve_full.py use env vars for paths (defaults = 910B3 layout):
export RWKV7_HF_PATH=/root/rwkv7-ascend      # where rwkv7_hf/ lives
export RWKV7_CPP_PATH=/root/rwkv7_ascend_v3.cpp   # the C++ forward source
```

## Repo layout

- `serving/` — the serving framework (the active project). See
  [`serving/SERVING.md`](serving/SERVING.md) + [`ARCHITECTURE.md`](ARCHITECTURE.md).
- `perf/` — the C++ forward (`rwkv7_ascend_v3.cpp`), shared with the serving engine.
- `tests/` — pytest suite (11 sampler unit tests run anywhere; 10 NPU integration
  tests need an NPU).
- `rwkv7_npu_ops.py`, `device_patch.py`, `bootstrap.py` — the Phase-1 op-shim (the
  vLLM port; dormant, awaiting vllm-ascend ↔ vllm-rwkv version alignment).
- `.github/workflows/ci.yml` — CI (sampler tests; NPU tests auto-skip).

## Running tests

```bash
# on an NPU box — all 21 tests (10 integration + 11 sampler)
source /usr/local/Ascend/cann-8.5.0/set_env.sh
python -m pytest tests/ -v

# without an NPU — only the 11 sampler unit tests run; integration auto-skips
PYTHONPATH=serving python -m pytest tests/test_sampler.py -v
```

CI (`.github/workflows/ci.yml`) runs the sampler suite on every push/PR.

## Where things go

| Change | File | Test |
|---|---|---|
| sampler logic (temp/top_k/top_p) | `serving/sampler.py` | `tests/test_sampler.py` (no NPU) |
| scheduler / batching / stop | `serving/serve_full.py` (`SlottedScheduler`) | `tests/test_integration.py` (NPU) |
| HTTP API / streaming / errors | `serving/serve_full.py` (`AsyncServer`, endpoints) | `tests/test_integration.py` (NPU) |
| routing / metrics | `serving/serve_router.py` | manual (needs a cluster) |
| the forward math | `perf/rwkv7_ascend_v3.cpp` | `tests/test_integration.py::test_greedy_bitexact` |
| C++ rebuild | automatic — `torch.utils.cpp_extension.load` recompiles on source change | |

When changing the C++ forward, delete the cached build (`rm -rf ~/.cache/torch_extensions/rwkv7_ascend_v3*`) to force a recompile.

## Conventions

- **Branch + PR**: branch from `master`, open a PR to `master`. The current active
  dev branch is `npu-serving-engine` (PR #1) until it merges.
- **Test before commit**: the NPU integration tests must pass on an NPU box; the
  sampler tests must pass everywhere.
- **Commits**: conventional one-line subject + body explaining why. Sign with your
  own identity (no bot co-author trailers).
- **Honest limitations**: if something is untested (e.g. the Docker image, multi-NPU),
  say so in code comment / doc — don't claim it works.
- **Two paths, one cpp**: the op-shim (Phase 1) and the serving engine are
  independent paths; they share only `perf/rwkv7_ascend_v3.cpp`. Keep changes to one
  path from breaking the other.
