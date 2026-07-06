# RWKV-7 HF Adapter — Ascend NPU 移植版

[![Ascend 910B](https://img.shields.io/badge/NPU-Ascend%20910B2C-red)](https://e.huawei.com)
[![torch_npu](https://img.shields.io/badge/torch__npu-2.9.0rc1-blue)](https://gitee.com/ascend/pytorch)
[![Status](https://img.shields.io/badge/status-forward-green)]()

基于 [rwkv7-hf-adapter](https://github.com/dsadsasdaddas/rwkv7-hf-adapter) 的**华为昇腾 NPU 全套移植**。使用 fla-free native 后端（纯 PyTorch），通过 `torch_npu` 在 Ascend 910B 上运行 RWKV-7 的 HF 全生态（load / generate / save / Cache / PEFT / Trainer），无需 CUDA / Triton / FLA。

## 环境验证（Ascend 910B2C, 64GB HBM）

| 项目 | 版本 |
|---|---|
| NPU | Ascend 910B2C |
| CANN | 8.5.1 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0rc1 |
| transformers | 4.57.6 |

## 测试结果

| 测试 | 结果 | 说明 |
|---|---|---|
| 环境 probe | ✅ PASS | npu-smi 26.0.rc1, 1× 910B2C 可见 |
| Native forward (tiny) | ✅ PASS | NativeRWKV7ForCausalLM on npu:0 |
| Native generate (tiny) | ✅ PASS | HF generate + recurrent cache |
| HF API contract (tiny) | ✅ PASS | batch=2, NativeRWKV7Cache |
| **C++ op-coalesced forward** | ✅ **323 tok/s (B=1)** | cos=1.0,1.83× Python eager |
| **batch decode 2× Albatross** | ✅ **见下表** | aggregate 吞吐,cos=1.0 验证 |

## ✅ 核心结果:batch decode 实现 2× Albatross(aggregate 吞吐)

launch 开销被 batch 摊薄 + 投影 GEMM 化 → forward 时间 1→128 batch 只涨 ~3×。**全规模 aggregate tok/s**(随机权重,cos=1.0 验证):

| model | B=1 | B=16 | B=64 | B=128 |
|---|---|---|---|---|
| 0.1B | 323 | 3433 | 9446 | **13504** |
| 1.5B | 87 | 953 | 2121 | — |
| 2.9B | 69 | 680 | 1748 | — |
| 7B | 33 | 313 | 793 | — |
| 13B | 25 | 235 | 585 | — |

单序列延迟(B=1)未达 2×(需 GEMV-Cube 融合,多月工程);详见 `ASCEND_RESULTS.md`。

## 复现 / 使用(在 Ascend 910B 服务器上)

```bash
# 0. 环境(CANN 8.5.1, torch_npu 2.9.0rc1)
source /usr/local/Ascend/cann-8.5.1/set_env.sh

# 1. RWKV-7 HF adapter 代码在 /root/rwkv7-adapter(depth-1 clone)
cd /root/rwkv7-adapter
export PYTHONPATH=/root/rwkv7-adapter

# 2. C++ op-coalesced forward + batch decode 吞吐(0.1B,B=1..16)
#    会自动用 torch.utils.cpp_extension.load 编译 rwkv7_ascend_v3.cpp
python3 bench_batch.py

# 3. 多模型 batch 吞吐(0.1B→13B,改脚本里 configs=[...] 选规模)
python3 bench_models.py

# 4. 正确性验证(B=4,每条序列 cos vs Python model)
python3 verify_batch.py

# 5. 单序列快速 generate(裸 decode loop,绕过 HF GenerationMixin)
python3 fast_generate_npu.py
```

> 注:`bench_batch.py` / `bench_models.py` / `verify_batch.py` / `rwkv7_ascend_v3.cpp`
> 都在本仓库;测速用随机权重(服务器被墙下不到真实模型),速度数字有效、输出质量未验。


## 用法

```python
import ascend_defaults  # 自动设置 NPU + fla-free native 后端
import torch_npu

from transformers import AutoModelForCausalLM, AutoTokenizer

dev = ascend_defaults.get_npu_device()  # "npu:0"
tok = AutoTokenizer.from_pretrained("path/to/rwkv7-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "path/to/rwkv7-hf", trust_remote_code=True,
    torch_dtype=torch.float16, device_map=dev
).eval()

ids = tok("Hello", return_tensors="pt").input_ids.to(dev)
print(tok.decode(model.generate(ids, max_new_tokens=20)[0]))
```

或使用 fast_generate（3× 更快）:

```python
from fast_generate_npu import fast_generate_npu
out = fast_generate_npu(model, ids, max_new_tokens=32)
```

## 优化探索记录

| 优化路径 | 结果 | 说明 |
|---|---|---|
| **raw decode loop** | ✅ **3.19×** (51→164 tok/s) | 绕过 HF GenerationMixin |
| `torch.compile(aot_eager)` | ❌ 0.37×（更慢） | NPU 编译开销 > 收益 |
| `torch.compile(inductor)` | ❌ 失败 | NPU driver 未配置 |
| `torch.npu.CUDAGraph` | ❌ 不存在 | torch_npu 2.9.0rc1 无此 API |
| triton-npu | ❌ 未安装 | 标准 triton 只支持 CUDA |
| bf16 | ❌ 无差异 | 910B 上 bf16 ≈ fp16 |
| batched shift-mix | 🚧 待验证 | microbench 4×（44→11μs），模型级验证中 |

## 架构

```
rwkv7-hf-adapter-ascend/
├── rwkv7_hf/              # 完整 HF adapter（从主仓库迁移）
│   ├── native_model.py    # fla-free NativeRWKV7ForCausalLM（纯 PyTorch）
│   ├── native.py          # per-token math（TMix/CMix port）
│   ├── modeling_rwkv7.py  # HF wrapper（from_pretrained / generate / save）
│   ├── configuration_rwkv7.py
│   └── ...                # 25 个模块
├── tests/                 # 全套测试（含 Ascend smoke）
├── scripts/               # 转换 + 验证脚本
├── fast_generate_npu.py   # 裸 decode 循环（3.19× over HF generate）
├── ascend_defaults.py     # NPU 默认环境设置
└── README.md
```

## 与主仓库的关系

主仓库 [rwkv7-hf-adapter](https://github.com/dsadsasdaddas/rwkv7-hf-adapter) 支持 CUDA/Triton/FLA（V100/A100/4090/5070 等）。本仓库是其 **Ascend NPU 分支**：

- 共享同一套 HF adapter 代码（`rwkv7_hf/`）
- 默认 fla-free native 后端（`RWKV7_NATIVE_MODEL=1`）
- 禁用所有 Triton/CUDA kernel 路径（`RWKV7_FAST_FORWARD=0`）
- 额外提供 `fast_generate_npu.py`（绕过 HF generate 开销）

## License

Apache-2.0
