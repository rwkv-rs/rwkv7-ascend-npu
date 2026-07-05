# RWKV7-Ascend serving framework

> A self-contained RWKV7 continuous-batch serving engine for Ascend NPU. Live
> OpenAI-compatible `/v1/completions` API, dynamic batching, streaming + stop
> strings, sampler (temperature/top-k/top-p), multi-worker router, Prometheus
> metrics, a Docker image, and a 21-test pytest suite — at **>2× Albatross**
> aggregate throughput on a 910B3. Our own code, no vLLM dependency.

---

## What this is (and isn't)

It's **vLLM's *serving ideas* applied to RWKV7 on NPU** — not a vLLM clone.

- **Borrowed from vLLM (the right ideas):** continuous batching, OpenAI-compatible
  API, a front-end router over multiple workers, a `/metrics` endpoint, SSE
  streaming + stop strings.
- **Our own (because RWKV7 ≠ a vLLm attention model):** a **state-based** scheduler
  (`SlottedScheduler`) that batches *recurrent state* `(sa, xp, xf, vf)` — not
  vLLM's PagedAttention + KV-cache blocks (RWKV7 is an RNN; it has no KV-cache).
  vLLM's signature machinery doesn't apply, so we wrote the minimal engine that does.
- **Scale:** ~7 files, single-card-per-worker. ~5–10% of vLLM's surface —
  deliberately minimal, RWKV7+NPU-specific.
- **Not a vLLm port:** vLLM isn't even reachable here (`vllm-ascend` can't pair
  with `vllm-rwkv` on CANN 8.5.0 — see "Why" below). We bypass it entirely.

Even upstream `vllm-rwkv` had to rewrite vLLM's v1 scheduler for RWKV7
(`rwkv_decode_wave`) — so RWKV7 was always going to need its own engine path.

---

## Why this exists (the vLLM blocker)

Serving RWKV7 on Ascend via vLLM is currently **not possible** — upstream version lag:

```
vllm-rwkv (rwkv-rs)   base = vllm v0.23.1rc0   (tracks vllm main)
vllm-ascend (Huawei)  newest = 0.22.1rc1       (needs CANN 9.0.0 + torch_npu 2.10)
                      huawei mirror tops out at 0.18.0
our 910B3             CANN 8.5.0 + torch_npu 2.9.0
```

No vllm-ascend release matches vllm-rwkv's v0.23 base. Overlaying vllm-rwkv's
`vllm/` onto stock vllm 0.13.0 + vllm-ascend 0.13.0 deadlocks (lazy
`current_platform` import cycle). So we serve RWKV7 ourselves.

---

## Architecture

```
                 HTTP (FastAPI / uvicorn)
                          |
        ┌─────────────────┴─────────────────┐
        |  serve_router.py (front-end)       |   least-in-flight routing,
        |  /v1/completions  /health  /metrics|   forwards JSON + SSE
        └─────────────────┬─────────────────┘
                          | (HTTP, one per worker)
          ┌───────────────┴───────────────┐
          ▼                               ▼
  worker 0 (NPU 0)               worker 1 (NPU 1) ...
   serve_full.py                  serve_full.py
   AsyncServer (single-threaded    (torch_npu is main-thread-only;
   asyncio loop)                   one SlottedScheduler per worker)
          |
   SlottedScheduler  — persistent batched recurrent state
   (grow on join, swap-remove on leave, NO per-step cat)
          |
   RWKV7Engine._step — one batched decode step
          |
   rwkv7_ascend_v3.cpp (C++ forward, libtorch/aclnn) — the NPU analog of
          |                       Albatross's hand-tuned CUDA kernel
   CANN 8.5.0 / 910B
```

RWKV7 is **state-based (RNN)**, so continuous batching is *simpler* than vLLM:
no PagedAttention, no KV-cache blocks. Each sequence carries a recurrent state;
the scheduler packs all active sequences' states into one batched tensor, runs one
C++ forward, samples per-sequence, and splits the evolved states back out.

Single-threaded per worker: `torch_npu` is **main-thread-only** (a worker thread
hangs), so each worker runs one asyncio loop with the decode step on it. Scale-out
is **multi-worker** (one per NPU), not multithreading.

---

## Components (`serving/` + `tests/`)

| file | role |
|---|---|
| `serve_engine.py` | `RWKV7Engine`: loads model + weights, compiles the C++ forward, batched decode. |
| `sampler.py` | `SamplerCfg` + `sample_rows` (greedy batched fast-path / temperature / top_k / top_p). Separated so it's CI-testable without NPU/rwkv7_hf. |
| `serve_full.py` | A worker: `SlottedScheduler` (persistent state, no per-step cat) + `AsyncServer` (single-threaded loop) + FastAPI `/v1/completions` (JSON + SSE) + stop strings + error isolation + timeouts + warmup. |
| `serve_router.py` | Front-end router: fans `/v1/completions` to N workers (least-in-flight), forwards JSON + SSE, `/health`, `/metrics`. |
| `run_cluster.sh` | Launcher: N workers (one per NPU via `ASCEND_RT_VISIBLE_DEVICES`) + the router. |
| `serve_scheduler.py` | Earlier cat-per-step scheduler (superseded by `SlottedScheduler`; kept as reference). |
| `engine_probe.py` | Correctness probe: C++ forward persists recurrent state across steps, bit-exact vs HF-native. |
| `Dockerfile` | CANN 8.5.0 base + deps + framework + rwkv7_hf + vocab (NPU devices mounted at run). |
| `QUANT.md` | Quantization investigation + PoC (W8A16 measured **not faster** on this stack). |
| `perf/rwkv7_ascend_v3.cpp` | The C++ forward, shared with the Phase-1 perf module. |
| `tests/` | 21 pytest tests (see below). |

---

## The C++ state-writeback fix (a latent correctness bug)

The original `rwkv7_ascend_v3.cpp` computed the recurrent state correctly for a
**single** step (cos=1.0 vs HF-native) but **never wrote it back** — the macro
reassigned a local C++ variable instead of the Python-passed tensor, so state
didn't persist and multi-step generation collapsed to a fixed cycle. (Single-step
cos=1.0 masked it; `bench_batch` re-zeroed state each call.) The fix adds three
in-place writebacks: `state_all[li].copy_(state)`, `xpa_all[li].copy_(h)`,
`xpf_all[li].copy_(h2)`. (`v_first` needs none — overwritten at layer 0 each step.)
After the fix, greedy `[0..15]→[16,17,18,21,18,21,18,21]` is bit-exact vs HF-native.

---

## Verified results (910B3, CANN 8.5.0, torch_npu 2.9.0)

- **Correctness**: greedy bit-exact vs HF-native; scheduler (mid-flight join,
  staggered completion, concurrent batch) bit-exact vs standalone — `ALL_MATCH`.
- **Throughput**: 64 concurrent × 32 new tokens = **3666 aggregate tok/s**
  (>2× Albatross ≈ 3000); all 6 model sizes (0.1B–13.3B) lead in batched aggregate.
- **Live API**: 3 concurrent `/v1/completions` on 1.5B → coherent text (greedy
  story + sampled poem).
- **Quantization PoC**: W8A16 via `npu_weight_quant_batchmatmul` is correct
  (cos=1.0) but **measured 0.6–0.89× (slower)** — fp16 matmul is already
  memory-BW-bound + heavily tuned. Not worth wiring in (see `QUANT.md`).
- **Test suite**: 21/21 pass (10 NPU integration + 11 sampler unit) in ~29s.

---

## Production readiness (~70%)

| Dimension | Status |
|---|---|
| Core engine correctness | 🟢 strong (bit-exact, 6 sizes) |
| Continuous batching | 🟢 strong (dynamic, bit-exact) |
| Serving API | 🟢 `/v1/completions` + streaming + stop + sampler (no chat/tools/structured) |
| Aggregate throughput | 🟢 >2× Albatross, all sizes |
| Reliability | 🟡 error isolation + timeouts + warmup (no rate-limit/auth/graceful) |
| Tests + CI | 🟡 21 pytests + GitHub Actions (no load tests; CI has no NPU) |
| Observability | 🟡 `/health` + `/metrics` (no logging/tracing) |
| Packaging | 🟡 Dockerfile written (untested — no Docker on 910B3) |
| Scalability | 🔴 multi-worker architecture ready, but multi-NPU unverified (910B3 has 1 NPU) |
| B=1 latency | 🔴 launch-overhead-bound (needs AscendC GEMV-Cube, multi-month) |
| Quantization | 🔴 investigated, doesn't help on this stack |

**Hard-blocked (need other infra/scope, not "incomplete"):** multi-NPU verification
(needs a multi-NPU box), Docker build/test (needs a Docker+NPU host), AscendC
GEMV-Cube (multi-month research).

---

## Quick start

**Single worker (dev):**
```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /root/rwkv7-ascend   # so rwkv7_hf/ + the cpp are reachable
RWKV7_REQ_TIMEOUT=120 python serve_full.py \
  --model <model-dir> --H 32 --N 64 --L 24 --port 8001
curl localhost:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":50,"temperature":0.7,"top_k":40}'
```

**Cluster (router + N workers, one per NPU):**
```bash
RWKV7_WORKERS_N=1 RWKV7_MODEL=<model-dir> RWKV7_H=32 RWKV7_L=24 bash run_cluster.sh
# -> router on :8000 forwarding to worker(s) on :8001+
curl localhost:8000/v1/completions ...
curl localhost:8000/metrics
```

**Docker:** build + run with NPU devices + a model mounted — see `Dockerfile`
header for the `docker run` command.

> **Paths** are configurable via env (`RWKV7_HF_PATH`, `RWKV7_CPP_PATH`,
> `RWKV7_DEVICE`, `RWKV7_VOCAB`, `RWKV7_REQ_TIMEOUT`, `RWKV7_MAX_TOKENS_CAP`).
> Model dims `--H/--N/--L` must match the checkpoint (0.1B 12/64/12, 0.4B 16/64/24,
> 1.5B 32/64/24, 2.9B 40/64/32, 7.2B 64/64/32, 13.3B 64/64/61).

---

## Tests

```bash
# on 910B3 (runs all 21: 10 NPU integration + 11 sampler unit)
python -m pytest tests/ -v
# in CI (GitHub Actions, no NPU): 11 sampler unit tests run, 10 integration auto-skip
```

---

## Relationship to the rest of this repo

- `rwkv7_npu_ops.py` / `device_patch.py` / `bootstrap.py` — the **op-shim** (Phase 1,
  the vLLM port). Independent of this serving framework; kept for the day
  vllm-ascend catches up to vllm-rwkv's base.
- `perf/rwkv7_ascend_v3.cpp` — the C++ forward, **shared** by the Phase-1 perf
  module and this serving framework (the state-writeback fix benefits both).
- `serving/` (this directory) — the self-contained serving engine, the pragmatic
  path to a live RWKV7 NPU service today.
