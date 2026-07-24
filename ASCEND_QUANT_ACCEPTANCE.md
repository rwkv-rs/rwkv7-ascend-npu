# RWKV-7 Ascend weight-only quantization acceptance

## Scope and status

This gate is deliberately narrow and reproducible. It covers the two dominant
FFN projections of the 7.2B RWKV-7 configuration (`4096 -> 16384` and
`16384 -> 4096`) on the **single measured stack**:

- Ascend 910B3 64 GiB
- CANN 8.5.0
- PyTorch 2.9.0 + torch_npu 2.9.0
- FP16 activations

Unknown Huawei devices, software stacks, dtypes, layer sizes, and batch sizes
fail closed to the serving implementation's 16-bit path. Operator availability
alone is never treated as a performance result.

## Raw-kernel candidates (not production acceptance)

The clean-rebuild dispatch profile uses 30 warmups followed by seven rounds of
200 calls and compares median synchronized wall time against the exact
same-shape FP16 matmul. The raw-operator candidate floor is **1.02x**, leaving
margin above mere parity. Only the exact rows rechecked in this profile are
listed below; no interpolation between rows is permitted.

| profile | exact logical rows `M` | raw-op floor on both FFN shapes | packed/FP16 payload |
|---|---:|---:|---:|
| W4A16, symmetric group 128 | 1, 8 | >=1.02x | 26.5625% |
| W8A16, symmetric per-output-channel | 17, 28 | >=1.02x | about 50.02% |

Run:

```bash
python benchmarks/analyze_ascend_quant_acceptance.py \
  benchmarks/results/rebuild/ascend910b3_quant_dispatch.jsonl \
  --output benchmarks/results/rebuild/ascend910b3_quant_acceptance.json
```

These rows populate only `RAW_KERNEL_CANDIDATE_BATCHES`.
`PRODUCTION_VERIFIED_BATCHES` is intentionally empty, and
`should_quantize(...)` therefore returns `False` for every tuple.

`rwkv7_ascend_quant.py` provides packed state-dict persistence, CPU oracles,
NPU execution, and this fail-closed policy. W4 checkpoints occupy 26.5625%
of the FP16 payload for these shapes (packed weights plus scale and zero tensors);
W8 occupies about 50.0%.

## Integration rule (important)

The raw-kernel gate is necessary but not sufficient for a serving claim.
Python dispatch, per-layer policy checks, graph breaks, packing, recurrent-state
movement, sampling, and scheduler behavior must be included in HF/vLLM/SGLang
end-to-end gates. The serving scheduler should evaluate policy once per batch
and call a cached raw operator fast path; repeated eager Python policy checks
can erase the small expansion-kernel gain. A backend is marked accepted only
when its own end-to-end artifact also passes.

W4 operator cosine is not a language-quality claim. Model-level perplexity or
fixed-corpus logits/generation acceptance is required before W4 is enabled by
default. W8 has much smaller local error but is subject to the same rule. Prior
end-to-end probes did not pass all latency and quality gates, so neither W8 nor
W4 is currently production-enabled.

The same clean rebuild also measured the actual Python module path. W8 passed
the 1.02x microbenchmark floor at rows 17 and 28 on both projections, while W4
fell below FP16 on the `4096 -> 16384` projection (0.922x at row 1 and 0.913x
at row 8). This is why raw-op candidates and production acceptance are separate
tables, and why production remains disabled for both formats.
