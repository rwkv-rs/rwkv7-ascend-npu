# Ascend 910B3 clean-rebuild quant dispatch evidence

Captured on 2026-07-24 with one Ascend 910B3, CANN 8.5.0, PyTorch 2.9.0,
and torch_npu 2.9.0. The environment row records driver/CANN file hashes,
the operator schema hash, benchmark hash, quant module hash, iteration counts,
and exact device name.

- `ascend910b3_quant_dispatch.raw.log` is the unmodified combined output. Its
  first two lines are a harmless `git rev-parse` diagnostic from the gitless
  remote staging directory.
- `ascend910b3_quant_dispatch.jsonl` is the same capture with only JSON records.
- `ascend910b3_quant_acceptance.json` is generated from that JSONL by:

```bash
python benchmarks/analyze_ascend_quant_acceptance.py \
  benchmarks/results/rebuild/ascend910b3_quant_dispatch.jsonl \
  --output benchmarks/results/rebuild/ascend910b3_quant_acceptance.json
```

The result accepts only exact raw-operator candidate rows. It deliberately
reports `production_gate_passed: false`: W4 module dispatch regresses on the
expansion projection, and neither bit width has passed a model/backend
end-to-end latency and quality gate.

## Real 7.2B model diagnostic

`real_model_w8_long.json` and `real_model_w4_cle_long.json` are synchronized
HF decode diagnostics on the real `fla-hub/rwkv7-7.2B-g0a` checkpoint. They
replace all 64 FFN key/value projections, retain no dense copy of those
weights, and compare against an independently loaded FP16 model. W4 uses
group-128 quantization plus the function-preserving weight-CLE transform.

| format | active tensor footprint | active allocator HBM | paired decode speed vs FP16 | min logits cosine | max NRMSE | all gates |
|---|---:|---:|---:|---:|---:|---:|
| W8A16 | 70.179% | 70.437% | 0.9800x | 0.999976 | 0.009482 | no |
| W4A16 + weight-CLE | 56.188% | 57.489% | 0.9756x | 0.995965 | 0.133690 | no |

The speed column is the median of alternating, paired dense/quant runs, not a
ratio selected from unrelated medians. W8 uses 9 pairs of 96 decode tokens;
W4 uses 7 pairs of 64. Both formats reduce active HBM substantially, but both
miss the no-slower latency gate and change one near-tied greedy choice on the
fixed dense token path. Production admission therefore remains empty.

Reproduce with the script whose SHA-256 is recorded in
`real_model_long_probe_script.sha256`:

```bash
python benchmarks/quick_w8_real_model_probe.py \
  --model <native-hf-model-dir> \
  --output benchmarks/results/rebuild/real_model_w8_long.json \
  --bit 8 --timed-steps 96 --rounds 9

python benchmarks/quick_w8_real_model_probe.py \
  --model <native-hf-model-dir> \
  --output benchmarks/results/rebuild/real_model_w4_cle_long.json \
  --bit 4 --equalization weight-cle --timed-steps 64 --rounds 7
```
