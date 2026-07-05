# RWKV7-Ascend serving framework

> **TL;DR** — Since `vllm-ascend` can't pair with `vllm-rwkv` on CANN 8.5.0 (version
> lag, see below), this directory is a **self-contained RWKV7 continuous-batch
> serving engine for Ascend NPU**, built on our verified C++ forward. It serves an
> OpenAI-compatible `/v1/completions` API, does dynamic continuous batching, supports
> top-k / top-p / temperature sampling, and hits **>2× Albatross** aggregate throughput
> on a 910B3 — all our own code, no vLLM dependency.

---

## Why this exists (the vLLM blocker)

Serving RWKV7 on Ascend via vLLM is currently **not possible**, and the reason is
upstream version lag — not anything fixable on our side:

```
vllm-rwkv (rwkv-rs)   base = vllm v0.23.1rc0   (tracks vllm main, rebased weekly)
vllm-ascend (Huawei)  newest = 0.22.1rc1       (needs CANN 9.0.0 + torch_npu 2.10)
                      huawei mirror tops out at 0.18.0
our 910B3             CANN 8.5.0 + torch_npu 2.9.0
```

There is **no vllm-ascend release that matches vllm-rwkv's v0.23 base**. Overlaying
vllm-rwkv's `vllm/` onto stock vllm 0.13.0 + vllm-ascend 0.13.0 fails: vllm-rwkv's
`platforms/__init__.py` uses a lazy `current_platform` (`__getattr__`) that
deadlocks vllm-ascend 0.13.0's import chain. Unblocking would require a major
CANN 8.5→9.0 + torch_npu 2.9→2.10 upgrade *and* still leaves a 0.23↔0.22 mismatch.

So instead of waiting on the upstreams, we serve RWKV7 ourselves.

---

## Architecture

```
                       HTTP (FastAPI / uvicorn)
                              |
                       /v1/completions  (async)
                              |
                         AsyncServer        ← single-threaded asyncio loop
                              |                (torch_npu is main-thread-only;
                              |                 _run task created on uvicorn's
                              |                 running loop via startup event)
                       SlottedScheduler      ← persistent batched recurrent state
                              |                 (grow on join, swap-remove on leave,
                              |                 NO per-step cat)  → 3666 tok/s
                              |
                    RWKV7Engine._step        ← one batched decode step
                              |
                  rwkv7_ascend_v3.cpp        ← C++ forward (libtorch/aclnn),
                  (perf/rwkv7_ascend_v3.cpp)    the NPU analog of Albatross's CUDA kernel
                              |
                          CANN 8.5.0 / 910B3
```

RWKV7 is **state-based (RNN), not attention-based** — so continuous batching is
simpler than vLLM: there's no PagedAttention, no KV-cache block management. Each
sequence carries a recurrent state `(sa, xp, xf, vf)`; the scheduler packs all
active sequences' states into one batched tensor, runs **one** C++ forward, samples
per-sequence, and splits the evolved states back out.

---

## Components (`serving/`)

| file | role |
|---|---|
| `serve_engine.py` | `RWKV7Engine`: loads model + weights, compiles the C++ forward, exposes `generate()` (prefill per-seq → stack states → batched decode). |
| `serve_full.py` | **The server.** `SlottedScheduler` (persistent state, no per-step cat) + `SamplerCfg`/`sample_rows` (greedy/top-k/top-p/temperature) + `AsyncServer` (single-threaded loop) + FastAPI `/v1/completions` + `/health`. Run with `--serve`. |
| `serve_scheduler.py` | Earlier cat-per-step scheduler (superseded by `SlottedScheduler` in `serve_full.py`; kept for reference — shows the design evolution). |
| `engine_probe.py` | Correctness probe: verifies the C++ forward **persists recurrent state across steps** and generates bit-exact vs HF-native. |

The C++ forward lives at `perf/rwkv7_ascend_v3.cpp` (shared with the Phase-1 perf module).

---

## The C++ state-writeback fix (correctness bug found + fixed)

The original `rwkv7_ascend_v3.cpp` `RWKV7_BODY` macro computed the recurrent state
correctly for a **single** step (cos=1.0 vs HF-native) — but **never wrote it back**:

```cpp
auto state = state_all[li];          // local C++ variable (a view)
...
state = state * w_exp + state @ ab + vk;   // REASSIGNS the local — Python's tensor never updates
```

So multi-step generation collapsed: state stayed at its initial value every step and
the model fell into a fixed cycle (`33,2973,144,...`). `bench_batch` missed this
because it re-zeroed state every call.

The fix adds three in-place writebacks **after** the state is consumed:

```cpp
state = state * w_exp.view(...) + at::matmul(state, ab...) + vk...;
state_all[li].copy_(state);     // persist recurrent state
...
xpa_all[li].copy_(h);           // persist attn x_prev (after x_prev is read)
...
xpf_all[li].copy_(h2);          // persist ffn x_prev (after xpf is read)
```

(`v_first` needs no persistence — it's overwritten at layer 0 every step.)

After the fix, greedy `[0..15] -> [16,17,18,21,18,21,18,21]` is **bit-exact** vs
HF-native. This unblocks correct multi-step generation.

---

## Verified results (910B3, CANN 8.5.0, torch_npu 2.9.0)

- **Correctness**: greedy generation bit-exact vs HF-native; slotted scheduler
  (mid-flight join + staggered completion) bit-exact vs standalone (`ALL_MATCH`).
- **Throughput** (`serve_full.py` `SlottedScheduler`, 0.1B, 64 concurrent x 32 new
  tokens): **3666 aggregate tok/s** (> 2x Albatross ~ 3000). The earlier
  cat-per-step scheduler measured 3284; the slotted buffer (no per-step cat) +
  batched-greedy fast-path recovered the overhead.
- **All 6 model sizes** (0.1B-13.3B) benchmarked via the C++ forward; leading
  Albatross in batched aggregate at every size (lead grows with model size).
- **Live API**: 3 concurrent `/v1/completions` requests on 1.5B -> coherent text
  (greedy story + sampled poem).

---

## Quick start (on a 910B box)

Pre-reqs: CANN env sourced; the HF adapter (`rwkv7_hf/`, from
`rwkv7-hf-adapter-ascend`) + the RWKV vocab (`rwkv_vocab_v20230424.txt`) + the model
checkpoint on disk; `fastapi`+`uvicorn` installed.

```bash
# 1. one-time: make the C++ forward reachable at the path serve_engine expects
ln -sf /path/to/vllm-rwkv-ascend/perf/rwkv7_ascend_v3.cpp /root/rwkv7_ascend_v3.cpp

# 2. start the server (1.5B example)
setsid bash -c '
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  cd /path/to/rwkv7-hf-adapter-ascend     # so rwkv7_hf/ is importable
  PYTHONPATH=/path/to/vllm-rwkv-ascend/serving:/path/to/rwkv7-hf-adapter-ascend \
  python serve_full.py --model <1.5b-model-dir> --H 32 --N 64 --L 24 --serve
' </dev/null >/tmp/sf.log 2>&1 &

# 3. use it
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":50,"temperature":0.7,"top_k":40}'
```

> **Note on paths**: `serve_engine.py` currently hardcodes the 910B3 layout
> (`sys.path.insert(0,"/root/rwkv7-ascend")`, cpp source `/root/rwkv7_ascend_v3.cpp`).
> Adjust to your environment (or export `PYTHONPATH` + symlink the cpp). The model
> dims `--H/--N/--L` must match the checkpoint (0.1B: 12/64/12, 0.4B: 16/64/24,
> 1.5B: 32/64/24, 2.9B: 40/64/32, 7.2B: 64/64/32, 13.3B: 64/64/61).

---

## Known limitations (production polish, not core functionality)

- The single-threaded loop blocks ~15ms/step during decode (torch_npu is
  main-thread-only — a worker thread hangs). Throughput under high concurrency would
  need a **multi-process** design, not multi-thread.
- ~4299 tok/s theoretical (raw `bench_batch`) vs 3666 achieved here — the gap is
  per-step Python overhead (token-tensor build, `argmax().tolist()`); a slotted
  "next-token" tensor buffer would close it.
- No EOS / stop-string handling (generation runs to `max_tokens`).
- One model per worker; model is loaded once at startup.
- Greedy + temperature/top-k/top-p only (no beam, no structured output).

---

## Relationship to the rest of this repo

- `rwkv7_npu_ops.py` / `device_patch.py` / `bootstrap.py` — the **op-shim** that lets
  upstream vllm-rwkv CUDA ops run on NPU (Phase 1, the vllm port). Independent of this
  serving framework — kept for the day vllm-ascend catches up to vllm-rwkv's base.
- `perf/rwkv7_ascend_v3.cpp` — the C++ forward, **shared** by the Phase-1 perf module
  and this serving framework (the state-writeback fix benefits both).
- `serving/` (this directory) — the self-contained serving engine, the pragmatic path
  to a live RWKV7 NPU service today.
