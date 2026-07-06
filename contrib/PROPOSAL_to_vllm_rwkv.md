# Proposal: contribute the pure-PyTorch op fallback upstream to `rwkv-rs/vllm-rwkv`

> **STATUS: Decided AGAINST (2026-07).** After examining `rwkv-rs/vllm-rwkv` (a new
> repo — 2 weeks old, 2 stars, issues disabled, CUDA-perf-focused, maintained by
> vLLM contributors), this won't merge: vLLM handles non-CUDA hardware via
> **hardware plugins** (vllm-ascend / vllm-rocm / ...), NOT model-level Python op
> fallbacks — so the op-shim is architecturally misaligned with vLLM. Plus
> vllm-rwkv is faster3a-CUDA-focused (a slow Python fallback is off-scope). The
> op-shim's value stays in THIS repo as a public reference. The merge-likely
> contribution (6-model weight verification) went to `rwkv7-hf-adapter-ascend`
> (PR #3) instead. Kept for the analysis below.

This is a ready-to-use draft for a PR (or initial issue) to
[`rwkv-rs/vllm-rwkv`](https://github.com/rwkv-rs/vllm-rwkv). The artifact is our
`rwkv7_npu_ops.py` — which despite the name is **100% pure PyTorch** (only
`import torch` + `torch.nn.functional`; no NPU/CUDA-specific code) and therefore a
**device-agnostic fallback** for the `rwkv7_*` ops.

## PR title

> Add pure-PyTorch fallback for `rwkv7_*` ops (CPU / non-CUDA backends + CUDA reference)

## PR body (paste, then adjust)

### What

Pure-PyTorch implementations of the ~40 `rwkv7_*` ops across the four namespaces
(`rwkv7_v3a_ops`, `rwkv7_fast_ops_fp16`, `rwkv7_wkv_fp16_v2`, `rwkv7_wkv_fp32_v2`),
registered via `torch.library`. **Additive** — sits alongside the CUDA kernels; used
only when the CUDA kernels aren't available.

Source: [`rwkv7_npu_ops.py`](../rwkv7_npu_ops.py) (~480 lines, pure torch). For this
PR, copy it to e.g. `vllm/rwkv7_pytorch_fallback.py` and generalize the docstring
(it was named "npu" for historical reasons — the code is device-agnostic).

### Why

Today RWKV7 in vllm-rwkv **requires CUDA** — the `rwkv7_*` ops are registered for
the CUDA dispatch key only, so the model can't run without a GPU. This fallback
unblocks:

- **CPU inference** (run RWKV7 with no GPU).
- **CI / unit tests without a GPU** (the sampler, scheduler, model logic can be
  tested on CPU runners).
- **A reference implementation** to verify the CUDA kernels (cosine vs the
  pure-PyTorch path).
- **A base for non-CUDA backends** (Ascend NPU, etc.) — they register the same ops
  and get a correct, if not optimal, path immediately.

### How (integration)

The fallback registers the ops for the CPU / Composite dispatch key, so it activates
automatically when CUDA kernels are absent. Minimal wiring in vllm-rwkv:

```python
# at model import time, register the fallback if CUDA kernels aren't built/available
import torch
if not torch.cuda.is_available():
    from vllm.rwkv7_pytorch_fallback import install
    install()   # registers pure-PyTorch impls for the rwkv7_* namespaces
```

(Or make it opt-in via an env var, e.g. `RWKV7_FORCE_FALLBACK=1`, for the
"reference vs CUDA" testing use-case.)

### Evidence

The implementations are mathematically verified: against the HF-native fla-free
forward (same WKV7 recurrence) on a fixed prompt, **cos = 0.99998**, argmax 100%
identical (tested on Ascend NPU; the math is device-independent so it holds on CPU
too). Detail in this repo's Phase-1 result.

> Note: this verifies the fallback against the **reference math** (HF-native). A
> direct **fallback-vs-CUDA-kernel** cosine test on a GPU box is the one test this
> PR should add (the artifact's author lacks a CUDA box with vllm-rwkv's kernels to
> run it). The expected result is cos ≈ 0.9999 (the CUDA kernels and the reference
> are both implementations of the same WKV7 equations).

### Caveats

- **Slow**: pure-Python op dispatch (~15 Python calls/layer) — a fallback/reference,
  not a performance path. The CUDA kernels remain the production path on GPU.
- **fp16/fp32 numerics**: the recurrence accumulates in fp32 (matching the CUDA
  kernel's `wkv_fp32` path); small fp16-vs-CUDA differences are expected and bounded.

## Before submitting

1. **Rename + reframe**: copy `rwkv7_npu_ops.py` → `rwkv7_pytorch_fallback.py`,
   change the docstring from "NPU" to "CPU / non-CUDA / reference".
2. **Add the direct CUDA-vs-fallback cosine test** (the one gap noted above) — run
   on a GPU box.
3. **Check the op signatures match current `vllm-rwkv`** (`csrc/libtorch_stable/rwkv7/`)
   — re-sync if upstream changed any op since this was written.
4. Follow vllm-rwkv's contribution rules (style, tests, CI).

## Not for this PR (separate)

- The **C++ op-coalesced forward** (`perf/rwkv7_ascend_v3.cpp`) is NPU-deployed
  (libtorch + CANN) — not generic, not for vllm-rwkv.
- The **serving framework** (`serving/`) is an independent engine, not vllm-rwkv code.
- An **optimized NPU path** (aclnn/AscendC kernels) belongs in `vllm-ascend`, not
  here.
