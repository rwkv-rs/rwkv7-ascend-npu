# Quantization investigation (int8 / int4 for RWKV7 on Ascend)

**Status: investigated — feasible but a ~2-week effort, NOT a quick PoC. Deferred.**
(out of the 路-A production loop's completable scope, same category as AscendC GEMV-Cube.)

## What we checked (910B3, torch_npu 2.9.0, CANN 8.5.0)

- `torch.ao.quantization` imports — but it has **no Ascend int8 backend**; the
  quantized ops don't dispatch to a fast NPU kernel.
- `torch.npu` exposes **zero** quant/int8 attributes (no `aclnnMatmul` int8 surface).
- int8 matmul via `a.float() @ b.float()` works — but that's a fake (casts back to
  fp32, no throughput win).

So **there is no high-level "quantize-and-go" path** on this stack.

## The real path (W8A16 — int8 weights, fp16 activations)

RWKV7 is memory-bound (small per-token compute, lots of weight traffic), so
**weight-only int8** is the win — same conclusion as on Blackwell
(`mm8`/`mm4` affine quant sped memory-bound layers 1.5–2× there).

To get it on Ascend:
1. **Offline-quantize** the fp16 projection weights (`at::linear` weights: r/k/v/o,
   ffn key/value, the LoRA w/a/v) to int8 + per-channel scales.
2. Replace each `at::linear(x, w)` in `rwkv7_ascend_v3.cpp` with an **aclnn int8
   matmul** (weight int8, activation fp16, fused dequant) — the aclnn C API, not a
   Python knob.
3. Verify accuracy (cos vs fp16, expect ≥0.99) and measure the speedup on the
   memory-bound sizes.

This is real low-level CANN engineering: ~30 linears in the cpp, aclnn int8
signature research, scale handling, accuracy + perf validation. **~2 weeks.**

## Alternative: atb (Ascend Transformer Boost)

`vllm-ascend` serves quantized models via **atb** (libatb.so, ships with the CANN
image). Wiring our cpp forward into atb's quantized linear is another route — but
it's a bigger architectural change (atb manages the layer, not us).

## Conclusion

- Feasible: yes. Ascend does int8 (aclnn W8A16 / atb).
- Quick PoC in the loop: **no** — torch_npu 2.9.0 gives no high-level handle; the
  first useful step is an aclnn int8 matmul in the cpp, which is the ~2-week item.
- Deferred from the production loop. Revisit when (a) we accept the 2-week scope,
  or (b) torch_npu exposes a Python-level W8A16 linear (then it's a small change).
