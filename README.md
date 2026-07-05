# RWKV-7 SGLang — Ascend NPU port

Production-grade serving of **RWKV-7** (Goose) on [SGLang](https://github.com/sgl-project/sglang) for **Huawei Ascend NPU** (Ascend 910B3, CANN 8.5.0).

## Status

🚧 Early / work-in-progress.

## Origin & credit

The integration design — declaring RWKV-7 as an all-linear Mamba-family model
inside SGLang, the custom linear-attention backend, the force-disabled token
radix cache, and the DPLR delta-rule WKV recurrence — is ported from
**[Hakureirm/rwkv-sglang](https://github.com/Hakureirm/rwkv-sglang)**
(Apache-2.0). That project targets NVIDIA CUDA + Triton; this repo reuses its
NVIDIA-agnostic integration shape and rewrites the kernel path for
Ascend / CANN.

## Target hardware

- NPU: Ascend 910B3 (64 GB HBM)
- Stack: CANN 8.5.0, ATB, torch + torch_npu, aarch64 / openEuler

## Plan

- **P0** environment + this repo
- **P1** correctness on Ascend (plain-torch WKV, token-exact vs a numpy oracle)
- **P2** serving (SGLang server up on the Ascend backend, dynamic batching, chunked prefill)
- **P3** performance (Ascend WKV kernel)
- **P4** production (TP over HCCL, CANN int8, chain speculative decoding)

## License

Apache-2.0, matching the reference project. See [LICENSE](LICENSE).
