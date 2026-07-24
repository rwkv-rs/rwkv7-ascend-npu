# Ascend wheel release 2026.07.24

This release packages the three Python integration layers validated by this
repository for the Huawei Ascend 910B3 stack. The wheels are pure Python; they
do **not** bundle CANN, `torch_npu`, vLLM, SGLang, or an RWKV checkpoint.

## Artifacts

| component | version | wheel | bytes | SHA-256 |
|---|---:|---|---:|---|
| Hugging Face adapter | 0.6.0 | `rwkv7_hf_adapter-0.6.0-py3-none-any.whl` | 2355116 | `e4f39fbcb3677d463de4a0a847102f4cded14b3eb7653a56f47b2bc717e16ea8` |
| vLLM Ascend plugin | 0.3.0 | `rwkv7_vllm_ascend-0.3.0-py3-none-any.whl` | 88627 | `4e5fb76894ee706f38a6b02fc324bd84b0d84345f09fc711223a1c7897d3272a` |
| SGLang Ascend plugin | 0.2.0 | `sglang_rwkv7_ascend-0.2.0-py3-none-any.whl` | 152102 | `587ed56aac60e36b565cb5ec33d82f85018f344bb2c85075ec81cd381edfe9bb` |

`release_manifest.json` records the package metadata, source-tree digest,
entry points, archive inspection, and isolated install smoke for each wheel.
[`ascend_install_smoke.json`](ascend_install_smoke.json) additionally records a
clean `--target --no-deps` install of the exact artifacts on an Ascend 910B3,
including real vLLM and SGLang entry-point registration and NPU discovery.
Verify every artifact before installing:

```bash
cd release/2026.07.24
sha256sum --check SHA256SUMS
```

## Installation

First install the runtime appropriate to the component. The hardware-validated
environment uses CANN 8.5.0 and `torch_npu`; the vLLM integration requires the
validated vLLM/vLLM-Ascend 0.18 environment, and the SGLang integration requires
the pinned SGLang source revision recorded in its evidence.

```bash
# Hugging Face adapter (declared Python dependencies may be resolved normally)
python -m pip install \
  release/2026.07.24/rwkv7_hf_adapter-0.6.0-py3-none-any.whl

# Install these into already prepared, version-pinned engine environments.
python -m pip install --no-deps \
  release/2026.07.24/rwkv7_vllm_ascend-0.3.0-py3-none-any.whl
python -m pip install --no-deps \
  release/2026.07.24/sglang_rwkv7_ascend-0.2.0-py3-none-any.whl
```

Source the CANN environment before starting an engine:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
```

## Reproducible build

From a clean checkout with Python 3.11:

```bash
python -m pip install \
  build==1.3.0 setuptools==79.0.1 wheel==0.46.3
python tools/build_release_wheels.py --output /tmp/rwkv7-ascend-release
cmp /tmp/rwkv7-ascend-release/SHA256SUMS \
    release/2026.07.24/SHA256SUMS
```

The builder fixes `SOURCE_DATE_EPOCH`, canonicalizes each wheel into a sorted
store-only ZIP so zlib versions cannot change its bytes, verifies safe archive
paths, rejects compiled payloads and symlinks, checks wheel metadata and entry
points, verifies every `RECORD` digest and size, and imports each package from
an isolated wheel-only target directory.

## Admission boundary

The release preserves the repository's fail-closed feature matrix:

- BF16/FP16 Hugging Face, vLLM, and SGLang engine paths are admitted on the
  evidenced 910B3 stack.
- Hugging Face W8A16 has a narrow, exact-stack production admission.
- Hugging Face W4A16 remains a measured candidate because it fails strict
  quality gates.
- Quantized vLLM and SGLang serving remain disabled until engine-level hardware
  evidence passes.

This is not a claim of validation on every Ascend SKU or a multi-card release.
See [`../../ASCEND_QUANT_ACCEPTANCE.md`](../../ASCEND_QUANT_ACCEPTANCE.md) and
[`../../SERVING_ACCEPTANCE.md`](../../SERVING_ACCEPTANCE.md) for the complete
evidence contract.
