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
Remaining gap to A100/3090-class (~230 tok/s): the projection GEMV (r/k/v/o +
ffn still use torch's default F.linear for M=1) — the next optimization lever.

### GEMV exploration (done) — F.linear is the best available; AscendC needed to beat it

Tested 3 M=1 GEMV paths on the 910B3 (fp16, 2048²) vs torch `F.linear` (~0.027ms, ~26% HBM bandwidth):
- **triton-ascend GEMV** (reduction kernel): 0.092ms — **3.4× slower** than F.linear.
- **bgmv_expand reuse** (sgl_kernel_npu LoRA gemv, single-group): 0.119ms — **3.4× slower AND wrong** (LoRA residual semantics, not a clean drop-in).
- **torch F.linear** (torch_npu matmul): the baseline — the fastest available.

So the projection GEMV has **no quick win** via triton-ascend or op-reuse. Beating F.linear (toward ~80% bandwidth, ~0.007ms) needs a **custom AscendC (CANN C++) GEMV kernel** — a multi-hour C++ effort (template fully mapped: sgl-kernel-npu `bgmv_expand` op_kernel/op_host + ACLRT_LAUNCH_KERNEL + cmake + `torch.ops.npu.<op>` binding; `ascend_port/ascendc/gemv_m1_kernel.cpp` scaffold committed). Even a perfect GEMV only reaches ~90 tok/s (projections are ~3-4ms of the 15ms/token); the remaining gap to 150+ needs fusing glue/lora/gate_corr too (more AscendC kernels). The WKV fusion (triton-ascend, 2×) was the win achievable via the quick paths; the rest is the AscendC fast-path marathon.

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
