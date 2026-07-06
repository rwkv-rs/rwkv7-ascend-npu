# RWKV-7 SGLang — Ascend NPU port

Production-grade serving of **RWKV-7** (Goose) on [SGLang](https://github.com/sgl-project/sglang)
for **Huawei Ascend NPU** (Ascend 910B3, CANN 8.5.0). Correctness-gated, cuda-graph
accelerated, fp32 + bf16, reproducible end-to-end.

## Status — production serving works on the 910B3

| | result |
|---|---|
| **Correctness gates** | WKV recurrence vs independent numpy oracle: worst **1.8e-6**; full RWKV-7 model greedy vs the numpy oracle: **20/20 token-exact** (0.1B BlinkDL `.pth`) |
| **fp32 serving** | fla-hub/rwkv7-0.4b-world, decode cuda graph, **~102 tok/s** bs=1, correct greedy ("Eiffel Tower… → Paris, France…") |
| **bf16 serving** | same model, **0.91 GB** (half of fp32), cuda graph, greedy identical to fp32 |
| **Scaling** | fla-hub/rwkv7-1.5b-world loads + serves (6 GB model + 19 GB state pool, cuda graph) |
| **Spec-decode** | chain worker deployed + registered + constructs + target verify graph capture (K=4) runs — see [SPEC_DECODE.md](SPEC_DECODE.md) |

Greedy output is verified coherent and matches the numpy reference, e.g.
`"The Eiffel Tower is located in the city of"` →
`" Paris, France. It is a symbol of the city and is one of the most recognizable structures in the world. The"`.

## Reproduce (on a 910B3 box, CANN 8.5.0 + ATB already installed)

```bash
# 1. Ascend env (fresh py3.11 venv): torch + torch_npu matched to CANN 8.5.0.
bash scripts/bootstrap_ascend_env.sh         # /data/rwkv7-sglang-venv  (torch 2.7.1, for the correctness gates)

# 2. Build + install SGLang (NPU build), then pin the NPU-matched deps.
bash scripts/install_sglang_ascend.sh         # sgl-kernel-npu from source + sglang[all_npu]
bash scripts/resume_sglang_build.sh           # (if the build's python/deps need the documented fixes)

# 3. Deploy the RWKV-7 integration into the sglang tree + serve (fp32, cuda graph).
bash scripts/deploy_ascend.sh                 # overlay + wiring + cann stub + dep-pin
bash scripts/serve_ascend.sh /data/rwkv7-models/rwkv7-0.4b-world-fla   # POST /generate
```

For **bf16** (half memory): use the torch_npu-2.9.0 venv instead —
`scripts/test_torch_npu_29_bf16.sh` (venv-29), `rebuild_sgl_kernel_npu_29.sh`,
`install_sglang_venv29.sh`, then `serve_ascend.sh … --dtype bfloat16` with the
venv-29 python. (torch_npu 2.9.0 is still CANN-8.5.0-compatible and fixes the
aclnn bf16 norm failure seen on 2.8.0.post2.)

See [BENCH.md](BENCH.md) for the numbers and [SPEC_DECODE.md](SPEC_DECODE.md)
for the speculative-decoding status.

## How it works

RWKV-7 is wired into SGLang as an **all-linear Mamba-family model**
(`linear_layer_ids = all layers`, `full_attention_layer_ids = []`), reusing
SGLang's hybrid-linear state plumbing:

- `ascend_port/wkv.py` — the DPLR delta-rule WKV recurrence in **pure torch**
  (no Triton/CUDA), greedy-exact vs the numpy oracle. Decode (T=1), multistep,
  and varlen-packed (`cu_seqlens`) modes; fp32 state, output cast to the input
  dtype so bf16 serving doesn't leak fp32 into downstream norms.
- `ascend_port/model.py` — full plain-torch RWKV-7 (faithful port of the numpy
  oracle; the M1c gate).
- `ascend_port/sglang_overlay/` — the SGLang integration: `Rwkv7Config`,
  `Rwkv7AttnBackend` (subclasses `AscendMambaAttnBackendBase`; `recurrence()`
  calls `wkv_recurrent` via gather/compute/scatter on the MambaPool), the model,
  and the wiring. `scripts/deploy_wiring.py` applies the 7-file sglang edits
  idempotently; `scripts/deploy_ascend.sh` is the one-shot deploy.
- A `triton.language.extra.cann` stub unblocks imports (PyPI triton-ascend 3.2.0
  is stripped; RWKV-7's pure-torch path never calls those kernels).

## Target hardware / stack

- NPU: Ascend 910B3 (64 GB HBM), CANN 8.5.0, ATB, aarch64 / openEuler
- fp32: torch 2.8.0 + torch_npu 2.8.0.post2 + sgl-kernel-npu + sglang 0.5.14
- bf16: torch 2.9.0 + torch_npu 2.9.0 (same CANN 8.5.0)

## Known limitations / roadmap

- **int8 quantization**: not yet — Hakureirm's quant paths are all CUDA
  (hand-written int4/int8 GEMV, cutlass w8a8 sm80-90 only). Ascend needs a
  from-scratch path (pure-torch w8a16 dequant-GEMV, or a CANN int8 matmul).
- **spec-decode**: the chain worker runs (target verify graph K=4) but the full
  launch hits v0.5.14 `BaseSpecWorker` interface skew (the worker targets
  Hakureirm's sglang base `bd08540`); pin the sglang checkout to `bd08540` to
  finish. Note: even Hakureirm's chain spec is eager 0.67× until the per-round
  forwards are cuda-graphed.
- **HCCL tensor parallel**: untested — this port was on a single NPU.

## Origin & credit

The integration design — RWKV-7 as an all-linear Mamba-family model in SGLang,
the custom linear-attention backend, the force-disabled token radix cache, the
DPLR delta-rule WKV recurrence, and the chain speculative decoding — is ported
from **[Hakureirm/rwkv-sglang](https://github.com/Hakureirm/rwkv-sglang)**
(Apache-2.0). That project targets NVIDIA CUDA + Triton (+ Apple MLX); this repo
reuses its NVIDIA-agnostic integration shape and rewrites the kernel path for
Ascend / CANN. Reference snapshots live under `reference/hakureirm/`.

## License

Apache-2.0, matching the reference project. See [LICENSE](LICENSE).
