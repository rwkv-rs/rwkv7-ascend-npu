# Architecture

A deep-dive for maintainers. For the user-facing overview, see
[`serving/SERVING.md`](serving/SERVING.md). This doc explains *how* the framework
works internally: the request lifecycle, the state-based scheduler, the
single-threaded model, the C++ forward, and how it scales.

## Layer cake

```
┌─────────────────────────────────────────────────────────────┐
│  Client (curl / OpenAI SDK)                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP /v1/completions (JSON or SSE)
┌──────────────────────────▼──────────────────────────────────┐
│  serve_router.py  (FastAPI, port 8000)                       │
│  • least-in-flight worker selection                          │
│  • forwards JSON + SSE transparently                         │
│  • /health (pings workers)  /metrics (Prometheus)            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (one conn per request, to a worker)
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐   ┌──────────────┐
│ worker 0     │  │ worker 1     │ … │ worker N-1   │   (one process per NPU)
│ serve_full.py│  │ serve_full.py│   │ serve_full.py│
│ NPU 0        │  │ NPU 1        │   │ NPU N-1      │   (ASCEND_RT_VISIBLE_DEVICES)
│ ┌──────────┐ │  │ ┌──────────┐ │   │ ┌──────────┐ │
│ │AsyncServer│ │  │ │AsyncServer│ │   │ │AsyncServer│ │  single asyncio loop each
│ │ (1 thread)│ │  │ │ (1 thread)│ │   │ │ (1 thread)│ │  (torch_npu is main-thread-only)
│ └────┬─────┘ │  │ └────┬─────┘ │   │ └────┬─────┘ │
│      │       │  │      │       │   │      │       │
│ ┌────▼─────┐ │  │ ┌────▼─────┐ │   │ ┌────▼─────┐ │
│ │Slotted   │ │  │ │Slotted   │ │   │ │Slotted   │ │  continuous-batch scheduler
│ │Scheduler │ │  │ │Scheduler │ │   │ │Scheduler │ │  (persistent batched recurrent state)
│ └────┬─────┘ │  │ └────┬─────┘ │   │ └────┬─────┘ │
│ ┌────▼─────┐ │  │ ┌────▼─────┐ │   │ ┌────▼─────┐ │
│ │RWKV7Engine│ │  │ │RWKV7Engine│ │   │ │RWKV7Engine│ │  prefill + 1 batched decode step
│ └────┬─────┘ │  │ └────┬─────┘ │   │ └────┬─────┘ │
│ ┌────▼──────────────────────────────────▼─────┐ │
│ │ rwkv7_ascend_v3.cpp  (C++ forward)           │ │  libtorch/aclnn ops → CANN kernels
│ └──────────────────────┬──────────────────────┘ │
└────────────────────────┼────────────────────────┘
                         ▼
                    CANN 8.5.0 / Ascend NPU
```

## Why two tiers (router + workers)

`torch_npu` is **main-thread-only**: a decode step in a background thread hangs the
process. So each worker is a single asyncio loop (decode on the loop thread), and
you scale by adding **worker processes** (one per NPU), not threads. The router is a
thin HTTP fan-out so clients hit one endpoint.

## Request lifecycle

**Non-streaming:**
```
client ──POST /v1/completions──▶ router ──pick least-in-flight worker──▶ worker
worker:  complete(prompt) appends a Seq (future) to AsyncServer.pending
         _run loop (on uvicorn's loop):
           • drain pending → SlottedScheduler.add(seq)  [prefill: B=1, builds state]
           • each tick: step()  [one batched forward for ALL active seqs]
           • when seq.gen reaches max_new (or stop) → seq.fut.set_result(final_text)
worker ──JSON──▶ router ──JSON──▶ client
```

**Streaming (SSE):**
```
client ──POST (stream:true)──▶ router ──▶ worker
worker:  submit_stream() → Seq with an asyncio.Queue (is_stream=True)
         _run loop: each step pushes the decoded delta into the seq's queue;
                    on completion pushes a None sentinel
worker ──SSE chunks (data: {delta:...})──▶ router ──SSE──▶ client
         (router forwards bytes; client sees token-by-token)
```

## SlottedScheduler — state-based continuous batching

RWKV7 is an RNN: each sequence carries a recurrent state `(sa, xp, xf, vf)`, not a
KV-cache. So batching = **pack active sequences' states into one batched tensor**,
run one forward, split the evolved states back. No PagedAttention, no block manager.

```
state layout (persistent, B = active count):
   sa : [L, B, H, N, N]  fp32   ← recurrent WKV state
   xp : [L, B, hidden]   fp16   ← attn "previous x" (for shift-mix)
   xf : [L, B, hidden]   fp16   ← ffn "previous x"
   vf : [B, hidden]      fp16   ← layer-0 v (rebuilt each step, no persistence)
```

- **join**: a new request prefills (B=1) to build its state, then `torch.cat`s its
  state onto the persistent batched tensor (`B += 1`).
- **per step**: `forward(active_tokens, sa, xp, xf, vf)` → logits `[B, vocab]` +
  the C++ forward **writes the evolved state back in place** (`state_all[li].copy_(…)`),
  so the persistent tensor evolves across steps with **no per-step cat**.
- **leave**: when a seq hits `max_new` or a stop string, swap-remove its slot
  (copy last slot into the freed slot, `B -= 1`).
- **stop strings**: each step, decode the seq's tokens; hold back the last
  `max_stop_len` chars (so a multi-char stop isn't emitted prematurely); on a
  match, emit only up to the stop and terminate.

The persistent-tensor trick is the perf win vs a naive "re-cat every step" design
(that measured 3284 tok/s; this one 3666).

## Single-threaded async model

`AsyncServer._run` is an asyncio task created on **uvicorn's running loop** (via
`@app.on_event("startup")` — not a separate loop, which would never run). It
alternates: drain pending → `step()` (blocks ~15ms, all torch on this thread) →
`await asyncio.sleep(0)` (yield to handle new HTTP requests). Errors in `step()`
trigger `fail_all` (fail every in-flight seq's future + reset the batch), so one bad
batch can't kill the worker. Per-request `asyncio.wait_for` enforces `RWKV7_REQ_TIMEOUT`.

## C++ forward (`perf/rwkv7_ascend_v3.cpp`)

One C++ call runs all `L` layers of TMix (attention) + CMix (FFN) for a batch,
using `at::linear` / `at::matmul` / `at::layer_norm` / `at::group_norm` → CANN's
fused kernels. This collapses ~960 Python op-launches into one call (the launch
overhead, not the math, dominates RWKV7 on NPU).

**State writeback (the correctness-critical part):** the macro must write the new
recurrent state back into the Python-passed tensors, *after* they've been read.
Three in-place copies per layer:
```cpp
state_all[li].copy_(state);   // recurrent WKV state
xpa_all[li].copy_(h);         // attn input  (the next step's x_prev)
xpf_all[li].copy_(h2);        // ffn input   (the next step's x_prev)
```
(`v_first` needs none — it's overwritten at layer 0 each step.) Without these, the
macro's local reassignment of `state` is lost and multi-step generation collapses to
a fixed cycle (single-step cos=1.0 still holds, which masks it).

## Scaling

- **One worker per NPU**: `run_cluster.sh` pins worker `i` to NPU `i` via
  `ASCEND_RT_VISIBLE_DEVICES=$i`; each worker sees only its NPU (as `npu:0` in its
  own view). The router load-balances (least-in-flight).
- **No TP/PP** within a worker — each worker serves the whole model on one NPU.
  (Tensor/pipeline parallelism is a vLLM-scale feature we deliberately don't have.)
- Verified at 1 worker on the single-NPU 910B3; multi-NPU scaling needs a
  multi-NPU box to validate.
