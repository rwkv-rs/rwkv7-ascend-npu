# v0.2.0 — RWKV7-Ascend serving framework

The first usable **serving** release. v0.1.0 shipped the op-shim + perf module
(Phase 1). v0.2.0 ships a **self-contained RWKV7 continuous-batch serving engine
for Ascend NPU** — because full vLLM serving stays blocked on `vllm-ascend`/`vllm-rwkv`
version lag (vllm v0.23 vs vllm-ascend max 0.22.1-on-CANN-9.0).

## What's new

- **Serving framework (`serving/`)** — a live OpenAI-compatible `/v1/completions`
  API on Ascend NPU, no vLLM dependency:
  - `serve_engine.py` — RWKV7 engine (loads weights, compiles the C++ forward).
  - `sampler.py` — greedy / temperature / top_k / top_p (CI-testable, standalone).
  - `serve_full.py` — a worker: `SlottedScheduler` (persistent batched recurrent
    state, no per-step cat) + `AsyncServer` (single-threaded loop) + FastAPI +
    **streaming (SSE) + stop strings + error isolation + request timeouts +
    startup warmup**.
  - `serve_router.py` — front-end router: fans out to N workers (least-in-flight),
    forwards JSON + SSE, `/health`, `/metrics` (Prometheus).
  - `run_cluster.sh` — launches N workers (one per NPU via `ASCEND_RT_VISIBLE_DEVICES`)
    + the router.
- **Correctness fix** in `perf/rwkv7_ascend_v3.cpp`: the recurrent state wasn't
  written back across steps (single-step cos=1.0 masked it; multi-step collapsed to
  a fixed cycle). Added in-place writebacks for `state_all` / `xpa_all` / `xpf_all`.
  Greedy is now bit-exact vs HF-native across steps.
- **Tests + CI**: 21 pytest tests (10 NPU integration + 11 sampler unit) + GitHub
  Actions (NPU tests auto-skip in CI).
- **Dockerfile** (CANN 8.5.0 base; NPU devices mounted at run — see header).
- **Quantization investigation + PoC** (`QUANT.md`): W8A16 via
  `npu_weight_quant_batchmatmul` is correct but **measured slower** (0.6–0.89×) on
  this stack — fp16 matmul is already memory-BW-bound + tuned. Not pursued.

## Verified (910B3, CANN 8.5.0, torch_npu 2.9.0)

- Greedy bit-exact vs HF-native; scheduler (mid-flight join, staggered completion,
  concurrent batch) bit-exact vs standalone.
- 64 concurrent × 32 new tokens = **3666 aggregate tok/s** (>2× Albatross ≈ 3000);
  all 6 model sizes lead in batched aggregate.
- Live API: 3 concurrent `/v1/completions` on 1.5B → coherent text.
- 21/21 tests pass in ~29s.

## Production readiness: ~70%

Core engine, batching, API, reliability, tests, packaging all in place. Hard-blocked
gaps (need other infra/scope): multi-NPU verification (needs a multi-NPU box), Docker
build/test (needs a Docker+NPU host), B=1 latency (AscendC GEMV-Cube, multi-month),
quantization (investigated, doesn't help here). See `serving/SERVING.md`.

## What this is

vLLM's *serving ideas* (continuous batching, OpenAI API, router) applied to RWKV7 on
NPU — **not a vLLM clone**. RWKV7 is RNN (state-based), so vLLM's PagedAttention +
KV-cache machinery doesn't apply; we wrote a minimal state-based scheduler instead.
~7 files, single-card-per-worker, RWKV7+NPU-specific.
