# RWKV-7 sglang on Ascend 910B3 — benchmarks

Hardware: 1 × Huawei Ascend 910B3 (64 GB HBM), CANN 8.5.0.
Stack: sglang 0.5.14 + torch 2.8.0 + torch_npu 2.8.0.post2 + sgl-kernel-npu
2026.6.1 + triton-ascend; RWKV-7 recurrence via `ascend_port/wkv.py` (pure torch,
M1a/M1c token-exact). Decode cuda graph ON (captured for bs 1..64 in ~26 s).

## fla-hub/rwkv7-0.4b-world (fp32, 24 layers, hidden 1024)

| Phase | Throughput | Notes |
|---|---:|---|
| Prefill | ~15 tok/s | eager (chunked, 128-token prompt) |
| Decode (bs=1, cuda graph) | **~102 tok/s** | steady state; first batch 5.5 (warmup) |

Greedy output verified coherent, e.g. prompt "The Eiffel Tower is located in the
city of" -> " Paris, France. It is a symbol of the city and is one of the most
recognizable structures in the world. The".

Reproduce: `scripts/serve_ascend.sh` then `POST /generate`.

## Notes / next perf levers
- Decode is cuda-graph captured (`npu graph: True`); the layout-agnostic
  `token_shift` (NPU conv state is `[size+1, 1, hidden]`) fixed the earlier
  capture-mode shape mismatch.
- Pure-torch WKV recurrence (sequential time loop) is the main decode cost;
  a triton-ascend or AscendC WKV kernel (P3) would lift decode throughput
  further toward the bandwidth bound.
- fp32 today; bf16/fp16 projections would ~2x throughput + halve memory.
