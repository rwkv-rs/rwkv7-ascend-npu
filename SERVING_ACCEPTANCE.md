# RWKV-7 Ascend serving acceptance

## Production-admitted scope

The committed clean-rebuild evidence admits the dense BF16 serving paths below
on one **Ascend 910B3 64 GiB**. Admission is limited to the exact measured
framework stacks and real `fla-hub/rwkv7-7.2B-g0a` checkpoint.

| Gate | vLLM V1 | SGLang |
|---|---:|---:|
| real public engine API | pass | pass |
| continuous/dynamic batching | pass | pass |
| chunked prefill with recurrent continuation | pass | pass |
| mixed decode + prefill | pass | pass |
| physical recurrent-state slot reuse | pass | pass |
| deterministic output after slot reuse | pass | pass |
| shared `Hello` greedy prefix | `[45, 308, 459]` | `[45, 308, 459]` |
| B1/B4/B8 throughput gate | pass | pass |

The framework throughput rows are:

| backend | B1 output tok/s | B4 output tok/s | B8 output tok/s | B4/B1 |
|---|---:|---:|---:|---:|
| vLLM V1 | 10.35 | 38.17 | 39.28 | 3.69x |
| SGLang | 6.07 | 23.70 | 45.44 | 3.91x |

These values are in-process engine measurements, not HTTP/network throughput.
Full commands, logs, traces, JSON and SHA256 manifests are in:

- `vllm-rwkv-ascend/evidence/rebuild/`
- `rwkv7-sglang-ascend/evidence/rebuild/`

## Machine-verifiable gate

Run from the repository root:

```bash
python benchmarks/verify_serving_acceptance.py
pytest -q tests/test_serving_acceptance.py
```

The verifier authenticates every required evidence file through its committed
`SHA256SUMS`, recomputes throughput and scaling from token/time totals, and
checks scheduler-level invariants rather than trusting top-level `PASS` fields.
It rejects missing/corrupt artifacts, model or hardware drift, output mismatch,
prefill-budget violations, absent continuation/mixed-batch events, and
inconsistent physical-slot reuse.

## Fail-closed boundaries

This admission does **not** cover:

- vLLM or SGLang W8/W4 end-to-end serving;
- prefix/radix reuse for recurrent RWKV state;
- speculative decoding;
- tensor or pipeline parallel execution;
- NPUGraph/CUDA-graph serving in the SGLang path;
- devices other than the measured 910B3 stack.

Both serving quant loaders remain default-off. A manifest that claims
`production_accepted=true` is rejected; the explicit acknowledgement path is
still labelled `raw-kernel-candidate-only`. HF W8 has a separate, narrow
backend admission documented in `ASCEND_QUANT_ACCEPTANCE.md`; it does not
implicitly admit the vLLM or SGLang execution paths.
