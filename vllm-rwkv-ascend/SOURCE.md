# Source provenance

The production-shaped vLLM V1 plugin in this directory is synchronized from
the validated plugin tree at commit:

`57cc98c0d9558dd58d9a366f4192922f9bf4550b`

That revision was built and tested against:

- vLLM 0.18.0
- vllm-ascend 0.18.0
- PyTorch 2.9.0 + torch_npu 2.9.0
- CANN 8.5.0
- Huawei Ascend 910B3

The clean real-engine acceptance artifacts are under `evidence/rebuild/`.
Huawei development is canonical in this monorepo; no external vLLM-RWKV fork
is required to install this plugin.
