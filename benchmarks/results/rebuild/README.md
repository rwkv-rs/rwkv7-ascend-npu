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
