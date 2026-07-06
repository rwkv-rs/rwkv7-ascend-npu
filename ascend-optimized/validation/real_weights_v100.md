# Real-weight verification — Ascend 910B3 vs V100 CUDA (all model sizes)

> **Conclusion: all 6 RWKV-7 models (0.1B–13.3B) on Ascend 910B3 NPU match V100
> CUDA within fp16 precision** — cos 0.99997–1.0, argmax 100% identical. This
> resolves the earlier "random weights, output quality unverified" caveat in
> ASCEND_RESULTS.md.

## Method

For each model size, the **same real weights** (official RWKV-7 checkpoint,
HF-converted) are loaded on both sides and run through the **same HF-native
fla-free forward** (`NativeRWKV7ForCausalLM`) on a fixed 16-token prompt
(`[0..15]`):

- **NPU side**: Ascend 910B3, torch_npu, fp16. Forward → `npu_logits` `[16, V]`.
- **V100 side** (reference): Tesla V100-32GB, CUDA, fp16, same forward → `v100_logits`.
- Compare: cosine similarity over all logits, argmax-match rate, max abs diff.

The fla-free native forward is itself verified cos=1.0 vs FLA and vs the official
`rwkv` package on the HF-adapter side, so this gates NPU-vs-CUDA numerical
equivalence of the whole stack.

## Results (2026-07-05, 910B3)

| model | params | cos vs V100 | argmax match | max_abs | greedy next-8 | verdict |
|---|---|---|---|---|---|---|
| rwkv7-g1d-0.1b | 0.1B | 0.99997 | 1.000 | 0.1875 | 16,17,18,21,18,21,18,21 | ✅ PASS |
| rwkv7-g1d-0.4b | 0.4B | 0.99997 | 1.000 | 0.1562 | 16,17,18,19,20,21,22,23 | ✅ PASS |
| rwkv7-g1g-1.5b | 1.5B | 1.00000 | 1.000 | 0.1875 | 16,17,18,19,20,21,22,23 | ✅ PASS |
| rwkv7-g1g-2.9b | 2.9B | 0.99998 | 1.000 | 0.0938 | 16,17,18,19,20,21,22,23 | ✅ PASS |
| rwkv7-g1g-7.2b | 7.2B | 0.99998 | 1.000 | 0.0938 | 16,17,18,19,20,21,22,23 | ✅ PASS |
| rwkv7-g1g-13.3b | 13.3B | 0.99999 | 1.000 | 0.1250 | 16,17,18,19,20,21,22,23 | ✅ PASS |

All greedy continuations continue the counting prompt `[0..15] → 16,17,...`
(the tiny 0.1B diverges after a few tokens, as expected, but still matches V100
token-for-token). Raw JSON in `results.json`.

## Reproduce

```bash
# NPU side (910B3): forward + save logits
PYTHONPATH=. python validation/verify_model.py <hf-model-dir>   # writes npu_<model>_logits.pt

# V100 side (reference): same forward, save v100_<model>_logits.pt
# (gen_v100_refs.py on the V100 box), then compare → cos/argmax/verdict
```

`verify_model.py` loads via `NativeRWKV7ForCausalLM.from_pretrained` (NOT
`AutoModelForCausalLM`+trust_remote_code — the transferred dirs carry only
config.json + safetensors, no adapter .py), forwards the fixed prompt on `npu:0`,
compares to `refs/v100_<model>_logits.pt` if present, writes a per-model JSON.

## Environment

- NPU: Ascend 910B3, torch_npu (CANN 8.5.0), python 3.11.14
- Reference: Tesla V100-PCIE-32GB, torch 2.5.1+cu124
- Weights: official RWKV-7 `g1d`/`g1g` checkpoints, HF-converted, fp16
