# Ascend 910B3 clean-rebuild SGLang evidence

This directory is the output of:

```bash
bash scripts/run_engine_acceptance.sh \
  /data/models/fla-hub-rwkv7-7.2B-g0a evidence/rebuild/acceptance.json
```

Environment: one Ascend 910B3 64 GiB, CANN 8.5.0, PyTorch/torch_npu 2.9.0,
Transformers 5.12.1, and pinned SGLang commit
`d0b9689805232d8ab37789121cbc3b766b5c723e`.

`acceptance.json` passed every fail-closed gate: two live requests, 64-token
chunked prefill, mixed decode+prefill, recurrent state continuation, physical
Mamba slot reuse after release, deterministic repeated output, radix cache
disabled, and the shared `Hello -> [45, 308, 459]` dense-token oracle.

The measured 63.28 seconds is an acceptance-workload wall time including cold
first inference, not a throughput claim.

## Real 7.2B E2E throughput gate

The separate warmed `scripts/run_e2e_performance.py` run goes through the real
SGLang `Engine.generate` API:

| batch | aggregate output tok/s | per-request tok/s | B1 scaling |
|---:|---:|---:|---:|
| 1 | 5.76 | 5.76 | 1.00× |
| 4 | 18.60 | 4.65 | 3.23× |
| 8 | 30.33 | 3.79 | 5.27× |

All requests generated 16 tokens, matched `[45, 308, 459]`, and passed the
dynamic-scaling gates. `e2e_performance.json` reports `status=PASS`; the
adjacent log records the complete run.
