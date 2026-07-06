# Quantization investigation + PoC (int8 / int4 for RWKV7 on Ascend)

**Status: PoC'd — W8A16 is NOT faster on this stack. Quant is not worth pursuing here.**

> Correction to the first draft: torch.ops.npu **does** have quant ops (292 ops,
> including `npu_weight_quant_batchmatmul`, `npu_anti_quant`, `npu_dynamic_quant`,
> `npu_group_quant`). The earlier "zero high-level quant ops" claim was wrong — I'd
> checked `torch.npu` attrs, not `torch.ops.npu`.

## The op + correct layout

`torch.ops.npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale)` is the
W8A16 path (fp16 activation × int8 weight, fused dequant). Correct convention:

- `x`:        fp16 `[B, in]`
- `weight`:    **int8 `[in, out]`** (transposed vs PyTorch linear's `[out, in]`)
- `antiquant_scale`: fp16 `[out]` (per-output-channel)

Per-output-channel int8 weight = `round(w / scale)` where `scale[o] = max|w[o,:]|/127`.
Reconstruction `cos = 1.0` vs fp16 linear — the op is correct.

## Measured speedup (910B3, CANN 8.5.0, torch_npu 2.9.0) — fp16 linear vs W8A16

```
H=768  (0.1B-like):  B=1 0.88x | B=8 0.92x | B=64 0.92x        (launch-overhead-bound)
H=4096 (7B-like, 32MB->16MB weight):
  B=1   fp16 0.024ms  W8A16 0.028ms  0.85x
  B=8   fp16 0.025ms  W8A16 0.028ms  0.89x
  B=64  fp16 0.025ms  W8A16 0.041ms  0.60x
  B=256 fp16 0.047ms  W8A16 0.065ms  0.73x
```

W8A16 is **slower at every size**. The fp16 matmul is already memory-bandwidth-bound
(0.024ms to read 32MB ≈ 1.3 TB/s ≈ HBM peak) and the CANN fp16 kernel is heavily
tuned; the W8A16 kernel's overhead + lower optimization exceed the half-traffic
benefit. (Contrast with Blackwell, where `mm8`/`mm4` *did* speed memory-bound
layers 1.5-2× — different kernel maturity.)

## Conclusion

- W8A16 via `npu_weight_quant_batchmatmul`: correct, but **does not help throughput**
  on CANN 8.5.0. Spending the ~2 weeks to wire it into the cpp forward would not pay.
- Remaining (uncertain, not quick): W4A16 (int4 via group-quant), W8A8
  (`npu_dynamic_quant` + matmul), or atb quantized layers (the vllm-ascend path).
  Given W8A16 already loses to fp16 here, these are unlikely to win without a better
  kernel; not worth the effort now.
- **Defer until** CANN ships a faster W8A16/W4A16 kernel, or until we adopt atb
  (which has its own tuned quant path). Re-measure then.

Memory benefit note: int8 weights would still halve **memory footprint** (helpful
for fitting 13.3B on smaller cards) even with no throughput win — so it's not
useless, just not a speed win on this stack.
