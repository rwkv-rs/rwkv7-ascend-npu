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

## HF W8 backend production admission

The public `rwkv7_hf.quantize_ascend_w8a16(..., policy="speed")` route now has
an end-to-end production row for the exact stack above, the real 7.2B
checkpoint, FP16, both FFN shapes, and logical rows B1/B4/B8. It replaces all
64 FFN key/value projections, removes their floating weights, and captures the
canonical HF decode path with NPUGraph.

| Batch | FP16 tok/s | HF W8 tok/s | Median paired speedup |
|---:|---:|---:|---:|
| 1 | 25.8451 | 26.4352 | 1.0241x |
| 4 | 94.2471 | 96.0139 | 1.0205x |
| 8 | 173.7094 | 178.3984 | 1.0259x |

Model tensor payload and isolated active HBM are respectively 70.18% and
71.48% of FP16. All five production quality prompts produce identical greedy
tokens; the global 48-step comparison has minimum cosine 0.99994028, maximum
NRMSE 0.01338704, minimum top-20 overlap 0.95, and maximum corpus loss delta
0.01193333.

The committed diagnostic also discloses one synthetic near-tied argmax change
(the W8 choice is FP16 rank 2 with a 0.02734375 top-1 margin). It remains
included in the global numeric gates but is not counted as a production corpus
generation/loss row.

The policy rejects other rows, BF16, other projection shapes, every other
device/runtime stack, and even an explicit unvalidated-stack override. Evidence
and reproduction commands are under
`rwkv7-hf-ascend/bench/ascend_910b3_w8_graph_20260724/`.

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

These rows populate only the shared module's `RAW_KERNEL_CANDIDATE_BATCHES`.
`PRODUCTION_VERIFIED_BATCHES` is intentionally empty, and
`should_quantize(...)` therefore returns `False` for every shared vLLM/SGLang
tuple. The HF W8 policy above is separately backend-gated.

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
default. The HF W8 row above is the first backend to pass that rule; W4 and the
vLLM/SGLang W8 paths are not production-enabled.

The committed real-checkpoint diagnostic now quantizes all 64 FFN key/value
projections of `fla-hub/rwkv7-7.2B-g0a`, removes the replaced FP16 weights, and
measures alternating paired decode runs:

| format | model tensor ratio | active HBM ratio | paired decode vs FP16 | min cosine | max NRMSE |
|---|---:|---:|---:|---:|---:|
| W8A16 | 0.70179 | 0.70437 | 0.9800x | 0.999976 | 0.009482 |
| W4A16 group-128 + weight-CLE | 0.56188 | 0.57489 | 0.9756x | 0.995965 | 0.133690 |

Both earlier eager experiments satisfy the memory and numeric-threshold checks, but miss the
no-slower decode gate and change one near-tied greedy choice on the fixed dense
path. The exact JSON, logs, script hash, and commands are under
`benchmarks/results/rebuild/`.

The same clean rebuild also measured the actual Python module path. W8 passed
the 1.02x microbenchmark floor at rows 17 and 28 on both projections, while W4
fell below FP16 on the `4096 -> 16384` projection (0.922x at row 1 and 0.913x
at row 8). This is why raw-op candidates and backend production acceptance are
separate tables. The shared serving policy and W4 remain disabled even though
the separate HF W8 NPUGraph route is narrowly admitted.
