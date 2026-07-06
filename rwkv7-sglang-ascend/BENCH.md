# RWKV-7 sglang on Ascend 910B3 — benchmarks

Hardware: 1 × Huawei Ascend 910B3 (64 GB HBM), CANN 8.5.0.
Stack: sglang 0.5.14 + torch 2.8.0 + torch_npu 2.8.0.post2 + sgl-kernel-npu
2026.6.1 + triton-ascend; RWKV-7 recurrence via `ascend_port/wkv.py` (pure torch,
M1a/M1c token-exact). Decode cuda graph ON (captured for bs 1..64 in ~26 s).

## fla-hub/rwkv7-1.5b-world — fp16, Triton-ascend WKV (operator-perf P3)

Operator optimization (loop bbbe7e93): swapped the pure-torch WKV recurrence for
the fused **Triton kernel via triton-ascend** (`ascend_port/wkv_triton.py`,
Hakureirm's kernel; compiles on the 910B3 with core tl ops, matches pure-torch
to ~1e-6). Decode (bs=1, cuda graph):

| WKV path | 1.5B fp16 bs=1 decode |
|---|---:|
| pure-torch (ascend_port/wkv.py) | ~33 tok/s |
| **Triton-ascend fused kernel** | **~66.6 tok/s (2×)** |

Output verified identical ("Once upon a time → …little girl named Lily…").
Remaining gap to A100/3090-class is **op-count, not the projection GEMV** — see the
correction + cross-repo comparison below (the GEMV is NOT the lever; an AscendC
Cube GEMV kernel measures 33× *slower* than `F.linear` for B=1).

### GEMV exploration (done) — F.linear is the best available; AscendC GEMV is NOT the lever

Tested 3 M=1 GEMV paths on the 910B3 (fp16, 2048²) vs torch `F.linear` (~0.027ms, ~26% HBM bandwidth):
- **triton-ascend GEMV** (reduction kernel): 0.092ms — **3.4× slower** than F.linear.
- **bgmv_expand reuse** (sgl_kernel_npu LoRA gemv, single-group): 0.119ms — **3.4× slower AND wrong** (LoRA residual semantics, not a clean drop-in).
- **torch F.linear** (torch_npu matmul): the baseline — the fastest available.

So the projection GEMV has **no quick win** via triton-ascend or op-reuse. The earlier
conclusion here — "beating F.linear needs a custom AscendC GEMV kernel" — is **wrong**:
the sibling repo `vllm-rwkv-ascend` built + ran the canonical AscendC **Cube** matmul
sample (`MatmulLeakyreluCustom`) on this same 910B3, and it measures **0.79ms vs
`F.linear`'s 0.024ms for a B=1 GEMV — 33× *slower***. The Cube unit is a dense-GEMM
(compute) engine; a B=1 projection is a memory-BW-bound GEMV that `F.linear` already
optimally handles. **Do not pursue an AscendC GEMV/Cube kernel for the B=1 path.** The
WKV fusion (triton-ascend, 2×) was the win achievable via the quick paths.

### Cross-repo comparison — op-coalescing is the real lever (not the GEMV)

This repo already runs decode under SGLang's built-in cuda graph (bs 1..64), so the
~960 host dispatches are already collapsed — and 1.5B B=1 still lands at **~66.6 tok/s**.
`sibling vllm-rwkv-ascend` (C++ op-coalesced forward — ~960 ops packed into one C++
call — + its own `NPUGraph`) reaches **114 tok/s** for the same 1.5B B=1. Both use a
graph; the gap is **op-count per step** (this repo's per-op Python `ascend_port/model.py`,
~15-20 ops/layer, vs a single coalesced C++ call). A100 reference: 1.5B B=1
`native_graph` = 164.5 tok/s.

| path (1.5B fp16, B=1, graph decode) | tok/s | bottleneck |
|---|---:|---|
| this repo (per-op Python + sglang cuda graph) | ~66.6 | op-count (Python dispatch per op, even under graph) |
| vllm-rwkv-ascend (C++ op-coalesced + NPUGraph) | 114 | op-count (fewer, but still kernel exec floor) |
| A100 `native_graph` | 164.5 | kernel exec (CUDA graph + fused ops) |

**Roadmap (the next lever):** port the op-coalesced-forward approach into this repo's
attention backend — pack the per-layer projection/recurrence ops into fewer launches
(C++ extension or a Triton fused layer kernel). That, on top of the existing cuda graph,
is what closes the gap to vllm-rwkv-ascend / A100. The AscendC GEMV scaffold
(`ascend_port/ascendc/gemv_m1_kernel.cpp`) is kept for elementwise/recurrence fusion,
not GEMV.

## fla-hub/rwkv7-0.4b-world — bf16 (torch_npu 2.9.0, venv-29)

bf16 serving WORKS end-to-end on the 910B3 (was blocked on torch_npu 2.8.0.post2
aclnn norm failure; torch_npu 2.9.0 -- still CANN-8.5.0-compatible -- fixes
aclnnLayerNorm/GroupNorm, and the `wkv_recurrent` output is cast back to the
input dtype so fp32 doesn't leak into bf16 LayerNorm). Model mem 0.91 GB (half
of fp32's 1.87 GB), decode cuda graph captured, greedy output identical to fp32
("Eiffel Tower... -> Paris, France..."). Launch via venv-29 + `--dtype bfloat16`.

## fla-hub/rwkv7-0.4b-world (fp32, 24 layers, hidden 1024)

| Phase | Throughput | Notes |
|---|---:|---|
| Prefill | ~15 tok/s | eager (chunked, 128-token prompt) |
| Decode (bs=1, cuda graph) | **~102 tok/s** | steady state; first batch 5.5 (warmup) |

Greedy output verified coherent, e.g. prompt "The Eiffel Tower is located in the
city of" -> " Paris, France. It is a symbol of the city and is one of the most
recognizable structures in the world. The".

Reproduce: `scripts/serve_ascend.sh` then `POST /generate`.

## fla-hub/rwkv7-1.5b-world (fp32, 24 layers, hidden 2048) — scaling

Serves on the 910B3 (same integration path as 0.4B). Load 6.01 GB, Mamba state
pool 19.08 GB (1627 slots), decode cuda graph captured (bs 1..64, ~32 s), server
"fired up and ready". Same `--dtype float32` production path. (bf16 blocked by a
CANN 8.5.0 aclnn norm-op limitation — see git log.)

## Notes / next perf levers
- Decode is cuda-graph captured (`npu graph: True`); the layout-agnostic
  `token_shift` (NPU conv state is `[size+1, 1, hidden]`) fixed the earlier
  capture-mode shape mismatch.
- Pure-torch WKV recurrence (sequential time loop) is the main decode cost;
  a triton-ascend or AscendC WKV kernel (P3) would lift decode throughput
  further toward the bandwidth bound.
- fp32 today; bf16/fp16 projections would ~2x throughput + halve memory.
