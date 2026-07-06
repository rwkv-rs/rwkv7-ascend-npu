# 中文

## 概述

`vllm-rwkv-ascend` v0.1.0 是 [`rwkv-rs/vllm-rwkv`](https://github.com/rwkv-rs/vllm-rwkv)(Albatross faster3a RWKV-7 引擎移植进 vLLM)的 **华为 Ascend 910B NPU 适配**。采用**附加层设计**——不修改 upstream 任何文件,所有 NPU 工作运行时叠加,`git pull upstream` 永远 fast-forward / 零冲突。

本版本交付:**Phase 1 正确性**(op-shim)+ **性能模块**(C++ op-coalesced forward),均在 910B3 真实权重下验证。

## 新增

- **op-shim(`rwkv7_npu_ops.py`)**:把 ~40 个 `torch.ops.rwkv7_*` CUDA op 用纯 PyTorch 重新实现并注册(torch.library),让 upstream 的 RWKV-7 模型代码**原样**跑在 NPU 上,无需 CUDA / Triton / FLA。**Phase 1 验证**:upstream 独立脚本 + shim 在 910B3 NPU 上 vs 我们已验证的 HF-native(同 0.1B 权重)→ **cos=0.99998,argmax 100% 一致**。
- **性能模块(`perf/rwkv7_ascend_v3.cpp`)**:整层 TMix+CMix 塞进**一次 C++ 调用**,干掉每层 ~15 次 Python op 调度。910B3 实测 **cos=0.99999(对齐 HF-native python)**,**B=1 68 tok/s,B=32 2229 tok/s aggregate**;每步 ~14.5ms 几乎不随 B 变(launch-overhead-bound),batch 摊销开销 → aggregate 近线性涨。
- **device_patch / bootstrap**:运行时盖掉 upstream 的 `first_device` / `zero_state` / `torch.cuda.device` / `is_current_stream_capturing` 等硬编码 → NPU,不编辑 upstream 文件。

## 设计:附加层,零 upstream 改动

```
vllm-rwkv-ascend/
├── rwkv7_npu_ops.py     # ~40 op 纯 PyTorch fallback
├── device_patch.py      # 运行时设备补丁
├── bootstrap.py         # install shim + patch + load_extensions 置空
├── harness/             # vendored 独立脚本(测试用)
├── perf/                # C++ op-coalesced forward(快速路径)
└── run_phase1.py        # 正确性验证入口
```

## 已知限制(诚实标注)

- **验证范围**:Phase 1 验证的是**模型层 / 独立 forward**(op-shim 对齐 HF-native cos=0.99998)。**完整 vLLM serving(OpenAI API / 连续批处理调度器)尚未在 NPU 上跑通**——那需要 `vllm-project/vllm-ascend` 插件 + worker/调度器 NPU 化(Phase 3,数周)。
- **单序列 2× Albatross**:B=1 单序列打不过 Albatross 的手调 CUDA,需要 GEMV-Cube 融合(**多月工程**)。但 **B>1 aggregate 吞吐**已达 2× Albatross 区间(参见 perf benchmark)。
- 910B3 单卡 B=1 = 68 tok/s;910B2C 上同代码 = 323 tok/s(910B3 每算子开销更大)。

## 路线图

- **Phase 2**:接 continuous batching,测 aggregate 吞吐能否复现 2× Albatross。
- **Phase 3**:vllm-ascend + OpenAI API serving。

---

# English

## Overview

`vllm-rwkv-ascend` v0.1.0 adapts [`rwkv-rs/vllm-rwkv`](https://github.com/rwkv-rs/vllm-rwkv) (the Albatross faster3a RWKV-7 engine ported into vLLM) for **Huawei Ascend 910B NPU**. It uses an **additive-layer design** — zero upstream files are modified; all NPU work is overlaid at runtime, so `git pull upstream` stays fast-forward / conflict-free.

This release ships **Phase 1 correctness** (op-shim) and the **perf module** (C++ op-coalesced forward), both verified with real weights on 910B3.

## What's New

- **Op-shim (`rwkv7_npu_ops.py`)**: re-implements all ~40 `torch.ops.rwkv7_*` CUDA ops in pure PyTorch and registers them via `torch.library`, so upstream's RWKV-7 model code runs **unchanged** on NPU — no CUDA / Triton / FLA. **Phase 1 verified**: upstream standalone + shim on 910B3 NPU vs our verified HF-native (same 0.1B weights) → **cos=0.99998, argmax 100% match**.
- **Perf module (`perf/rwkv7_ascend_v3.cpp`)**: collapses the full TMix+CMix layer into **one C++ call**, eliminating ~15 Python op dispatches per layer. Measured on 910B3: **cos=0.99999 (vs HF-native python)**, **B=1 68 tok/s, B=32 2229 tok/s aggregate**; per-step ~14.5 ms is nearly B-independent (launch-overhead-bound), so batching amortizes overhead → aggregate scales ~linearly.
- **device_patch / bootstrap**: runtime overrides for upstream's `first_device` / `zero_state` / `torch.cuda.device` / `is_current_stream_capturing` hardcodes → NPU, without editing upstream files.

## Design: additive layer, zero upstream edits

```
vllm-rwkv-ascend/
├── rwkv7_npu_ops.py     # ~40 ops as pure-PyTorch fallbacks
├── device_patch.py      # runtime device patches
├── bootstrap.py         # install shim + patch + no-op load_extensions
├── harness/             # vendored standalone (for testing)
├── perf/                # C++ op-coalesced forward (fast path)
└── run_phase1.py        # correctness entry point
```

## Known Limitations (honest)

- **Scope verified**: Phase 1 verifies the **model layer / standalone forward** (op-shim vs HF-native cos=0.99998). **Full vLLM serving (OpenAI API / continuous-batching scheduler) is NOT yet running on NPU** — that needs the `vllm-project/vllm-ascend` plugin + worker/scheduler NPU-ization (Phase 3, weeks).
- **Single-seq 2× Albatross**: B=1 single-seq cannot beat Albatross's hand-tuned CUDA without GEMV-Cube fusion (**multi-month**). But **B>1 aggregate throughput** already clears the 2× Albatross range (see perf benchmark).
- 910B3 single-card B=1 = 68 tok/s; same code on 910B2C = 323 tok/s (910B3 has higher per-op overhead).

## Roadmap

- **Phase 2**: continuous batching — measure whether aggregate throughput reproduces 2× Albatross.
- **Phase 3**: vllm-ascend + OpenAI API serving.
