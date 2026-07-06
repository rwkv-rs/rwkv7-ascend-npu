# RWKV-7 Chain Speculative Decoding on Ascend — status

Reference: Hakureirm/rwkv-sglang `sglang_overlay/sglang/srt/speculative/rwkv_chain_worker.py`
(ADR-0006, 595 lines, device-agnostic Python — snapshot/rollback is pure
`torch.Tensor.clone()`/`.copy_()` on the MambaPool `conv[0]`/`conv[1]`/`temporal`
slots; no CUDA/Triton kernels of its own).

## What's done on the 910B3

- **Worker deployed** to the sglang tree (`sglang/srt/speculative/rwkv_chain_worker.py`,
  fetched verbatim from Hakureirm main).
- **Registered** in `spec_info.py` (`scripts/deploy_spec_wiring.py`, idempotent):
  `RWKV_CHAIN` enum value + `is_rwkv_chain()` predicate + a `create_worker`
  branch returning `RwkvChainWorker`.
- **Backend contract confirmed**: the Ascend `Rwkv7AttnBackend.recurrence` extend
  branch commits `final_state` back into `temporal[safe_idx]` (line 225) — exactly
  the contract the worker's snapshot/restore + J==K commit-free / J<K restore+rerun
  relies on. No backend change needed.
- **Verified launching**: target fla-hub/rwkv7-1.5b-world (6.01 GB) + draft
  fla-hub/rwkv7-0.4b-world (1.86 GB) both load; the worker constructs and reports
  `RWKV_CHAIN spec worker up: draft=...0.4b K=4 (eager increment (i))`.

## Remaining (v0.5.14 interface gap)

After the worker comes up, the v0.5.14 scheduler calls
`model_worker.alloc_memory_pool(...)` (a `BaseSpecWorker` method). Hakureirm's
`RwkvChainWorker` deliberately does NOT subclass `BaseSpecWorker` (it mirrors
`StandaloneWorker` directly against Hakureirm's sglang base), so it lacks
`alloc_memory_pool` (and likely a few more `BaseSpecWorker` methods the v0.5.14
scheduler pokes). Closing this = either (a) re-parent `RwkvChainWorker` on
`BaseSpecWorker` and override only what differs, or (b) add the handful of
expected methods (`alloc_memory_pool`, `init_attention_backends`, etc.). This is a
version-skew adaptation, not an Ascend issue.

Launch flags (once the gap is closed):
```
--speculative-algorithm RWKV_CHAIN \
--speculative-draft-model-path <fla 0.4B> \
--speculative-num-draft-tokens 4 \
--max-running-requests 48          # spec-decode asserts this is set
```

## Speed reality (important)

Per Hakureirm's own F0031, chain spec-decode is "increment (i) FUNCTIONAL": 9/10
prompts token-identical to plain greedy (1 fp16 reduction-order flip on a
near-tie), but **eager 0.67× the cuda-graphed plain baseline** at 1.5B — i.e. it
is NOT yet a speedup. The speedup is gated behind increment (ii): draft-decode
cuda graph + fixed-shape K-token verify graph. So even after the v0.5.14 gap is
closed, expect spec-decode to be slower than the plain 102 tok/s fp32 path until
the spec forwards are graphed (future work, matches Hakureirm's roadmap).
