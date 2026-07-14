# RWKV-7 Ascend vs Qwen3.5 / Albatross Benchmark

本文固定两条互不替代的比较线：

- **Qwen3.5** 是部署性能和模型质量对手。性能必须同卡、同卡数、同精度、同 batch、同 prompt/decode 长度；质量必须用同一评测器单独测试。
- **Albatross** 是 RWKV-7 高性能推理引擎参考。只有同一 RWKV checkpoint、同一硬件、同一 dtype、同一 B/T 和 cache 策略的结果，才允许计算引擎倍率。

速度领先不能证明模型质量领先，跨 GPU 的 Albatross 数字也不能写成同卡引擎胜负。

## 固定模型矩阵

当前只验收用户指定的五个 Qwen3.5 Dense 尺寸：

| Tier | RWKV-7 | Qwen3.5 | 主工作负载 |
| --- | ---: | ---: | --- |
| dense-0 | 0.4B | 0.8B | B1/B4，P512/P2048，D128 |
| dense-1 | 1.5B | 2B | B1/B4，P512/P2048，D128 |
| dense-2 | 2.9B | 4B | B1/B4，P512/P2048，D128 |
| dense-3 | 7.2B | 9B | B1/B4，P512/P2048，D128 |
| dense-4 | 13.3B | 27B | B1/B4，P512/P2048，D128；fp16 至少 2 卡 |

机器可读定义位于 `vllm-rwkv-ascend/perf/qwen35_dense_matrix.json`。任何缺失、blocked、OOM 或正确性失败的行都不能计为通过。

单行通过条件：

1. RWKV prefill tok/s 严格大于 Qwen；
2. RWKV decode tok/s 严格大于 Qwen；
3. RWKV 同口径峰值显存不高于 Qwen；
4. RWKV greedy match 通过且 logits cosine 不低于 `0.9999`；
5. 两侧设备、卡数、dtype、B/P/D 和显存统计范围一致。

## 2026-07-14 单卡结果

环境：单张 `Ascend910B2C 64GB`、CANN `8.5.1`、PyTorch `2.9.0+cpu`、torch-npu `2.9.0rc1`、fp16 权重。Qwen 使用官方 `vLLM 0.18.0 / vLLM-Ascend 0.18.0` strict eager 文本路径；当前 CANN/torchair 组合不能稳定启用官方图模式，因此结果明确标记为 eager。prefill 使用完整形状预热后取 5 次中位数；decode 固定 D128。显存范围都是所选设备上的全部 NPU 进程。

原始远端文件名、精确浮点结果和机器信息已整理到：

- `vllm-rwkv-ascend/perf/results/qwen35_dense_910b2c_20260714.json`
- `vllm-rwkv-ascend/perf/results/qwen35_dense_910b2c_20260714_report.md`

### dense-0：RWKV-7 0.4B vs Qwen3.5-0.8B

| B | Prompt | RWKV prefill | Qwen prefill | 比率 | RWKV decode | Qwen decode | 比率 | RWKV/Qwen 峰值显存 | 状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 512 | 14,446.2 | 8,858.3 | 1.631x | 146.2 | 64.9 | 2.251x | 1,414 / 30,099 MiB | pass |
| 4 | 512 | 31,221.0 | 18,237.7 | 1.712x | 470.1 | 247.3 | 1.901x | 1,770 / 30,159 MiB | pass |
| 1 | 2048 | 28,667.4 | 26,697.2 | 1.074x | 149.8 | 65.4 | 2.290x | 1,700 / 30,179 MiB | pass |
| 4 | 2048 | 43,134.0 | 42,359.4 | 1.018x | 472.8 | 246.2 | 1.920x | 2,938 / 30,559 MiB | pass |

结论：这一档当前为 **4/4 工作负载通过**。B4/P2048 的 prefill 余量只有 `1.8%`，后续改动必须持续回归，不能把该行当作宽松领先。

### dense-1：RWKV-7 1.5B vs Qwen3.5-2B

| B | Prompt | RWKV prefill | Qwen prefill | 比率 | RWKV decode | Qwen decode | 比率 | RWKV/Qwen 峰值显存 | 状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 512 | 10,838.1 | 8,773.4 | 1.235x | 110.0 | 65.7 | 1.674x | 4,404 / 29,717 MiB | pass |
| 4 | 512 | 19,187.3 | 16,610.2 | 1.155x | 378.2 | 247.6 | 1.527x | 5,118 / 29,777 MiB | pass |
| 1 | 2048 | 18,871.0 | 22,707.5 | 0.831x | 107.5 | 65.3 | 1.647x | 4,990 / 29,817 MiB | **fail: prefill** |
| 4 | 2048 | 21,930.1 | 34,065.9 | 0.644x | 379.4 | 247.7 | 1.532x | 7,634 / 30,237 MiB | **fail: prefill** |

结论：短 prompt 和全部 decode/显存轴已领先；P2048 prefill 仍未达标。B1 还差约 `20.3%` 吞吐，B4 还差约 `35.6%`。因此不能宣称 2B 档全面超过。

B1/P2048 已做完整 2048-token reference 校验：greedy match 为 true，logits cosine `0.999997318`。B4/P2048 已通过短序列 recurrence smoke、decode eager 对齐，以及 NPU Graph 与 eager 的逐元素一致性；完整 B4/P2048 reference 仍应在最终推广前补跑。

### dense-2 / dense-3 / dense-4

当前主机没有对应的 RWKV-7 2.9B/7.2B/13.3B 与 Qwen3.5 4B/9B/27B 完整权重，所以这些行状态为 **missing**，不是 pass。当前只暴露 1 张 NPU；27B fp16 条目要求至少 2 张，因此多卡状态为 **blocked**。

多卡预检命令：

```bash
cd vllm-rwkv-ascend
PYTHONPATH=. python perf/run_qwen35_dense_matrix.py \
  --output results/qwen35-dense-preflight.json \
  --require-ready
```

脚本会读取 `ASCEND_RT_VISIBLE_DEVICES` 或 torch-npu 的可见卡数。卡数不足时返回非零并记录 `blocked`，不会生成伪 TP/PP 性能结果。正式多卡行必须让 Qwen TP 与 RWKV 支持的 PP/服务吞吐路径使用相同卡数，并同时报告 speedup、efficiency、每卡峰值显存和正确性。

## 本轮优化证据

以下改动全部保持 opt-in benchmark 路径，不改变默认服务行为：

- AscendC tiled prefill shift-mix；
- 原生 BF16 gate-factor 输出，去掉 FP32 中间落盘后再转换；
- 原生 BF16 unit-lower inverse 输入/输出；
- BF16 causal-matrix mask；
- 32×32 对角块 inverse 加 Cube 分层重建；
- work-efficient dense affine tree prefix；
- `tree_root` 使用 FP32 additive root 直接生成最终 cache，B1/P2048 从约 `113.8 ms` 降到 `108.5 ms`，并通过完整 2048-token greedy gate；
- batch-aware resident NPU Graph 把 embedding、整步 decode、argmax 和 next-token 写回一起捕获。

NPU Graph 解决了长 prompt 后的异常 decode 退化：

| 模型/工作负载 | eager decode | graph decode | Qwen decode | graph 正确性 |
| --- | ---: | ---: | ---: | --- |
| RWKV 0.4B, B4/P2048 | 195.5 tok/s | 472.8 tok/s | 246.2 tok/s | logits/state max abs = 0 |
| RWKV 1.5B, B4/P2048 | 193.1 tok/s | 379.4 tok/s | 247.7 tok/s | logits/state max abs = 0 |

1.5B/B4/P2048 的 stage probe 表明剩余问题不是 wrapper：总 prefill 约 `373.6 ms`，其中 recurrent scan `237.4 ms`。主要组成是 gate/factor preparation `60.8 ms`、blocked inverse `59.8 ms`、dense summary `48.0 ms`、dense prefix `17.7 ms` 和 factor apply `14.6 ms`。下一步必须继续做 compact-WY summary/prefix/apply-output 原生融合，减少 dense `[N,N]` 中间量与多次全局内存往返；继续做 Python wrapper 微调不足以补齐 B4 的差距。

## 机器分析

```bash
cd vllm-rwkv-ascend
PYTHONPATH=. python perf/analyze_qwen35_matrix.py \
  perf/results/qwen35_dense_910b2c_20260714.json \
  --json-output perf/results/qwen35_dense_910b2c_20260714_report.json \
  --markdown-output perf/results/qwen35_dense_910b2c_20260714_report.md
```

当前 analyzer 的全局状态是 **FAIL**：dense-0 全部 pass；dense-1 有两个 P2048 prefill fail；其余三档 missing。`--strict` 会按预期返回非零。

## 复现 RWKV 最佳长上下文行

构建 direct kernels 后，1.5B/B1/P2048 的关键参数如下：

```bash
RWKV7_ASCENDC_DIRECT_BUILD_DIR=/tmp/rwkv7_direct \
python perf/bench_rwkv7_pth_prefill.py \
  --model-pth /path/to/rwkv7-g1h-1.5b.pth \
  --device npu:0 --batch-size 1 --prompt-length 2048 \
  --decode-length 128 --correctness-length 2048 \
  --warmup 1 --iterations 5 --decode-warmup 8 \
  --chunk-scan --chunk-size 128 --chunk-compute-dtype bf16 \
  --fused-shift-mix --fused-shift-mix-rows-per-block 16 \
  --head-layer-norm --fused-add-layer-norm \
  --dense-chunk-prefix --dense-prefix-algorithm tree_root \
  --chunk-inverse-backend native_blocked \
  --chunk-inverse-base-size 32 --decode-npu-graph \
  --output results/rwkv7-1.5b-b1-p2048-d128.json
```

Qwen 侧必须使用相同 B/P/D、设备、卡数、精度和内存统计范围。当前环境的 vLLM-Ascend 命令需显式使用 `--enforce-eager`；如果未来兼容的官方图模式可用，必须重跑 Qwen 主表，而不是继续沿用 eager 作为永久基线。

## Albatross 比较

当前没有 Albatross 在 Ascend 上的同 checkpoint 后端，也没有本项目在同一张 NVIDIA 卡上与 Albatross 同条件复测的数据。因此 Albatross 状态仍是 **pending**，本轮不计算倍率。

可引用的 Albatross 公开数字只能作为外部背景，不能进入上面的 Qwen 同卡主表。正式对比至少固定：

- 相同 RWKV `.pth` 与 checksum；
- 相同 NVIDIA GPU 和卡数；
- 相同 fp16/bf16、B/T、prefill/decode 定义；
- 相同 cache 是否驻留、embedding 位置和输出头范围；
- 相同 warmup/iterations，并报告 p50/p90、tok/s 和峰值显存。

示例流程：

```bash
git clone https://github.com/BlinkDL/Albatross.git
cd Albatross/faster3a_2605
python rwkv7_fast_v3a.py \
  --model /path/to/the-same-rwkv7.pth \
  --wkv fp16 --emb gpu --warmup 20 --iters 200 \
  --cases "1x1,1x512,1x2048,4x1,4x512,4x2048"
```

随后在同卡运行本项目等价 workload。跨 RTX 4090/5090 与 Ascend 910B 的结果只能标为 platform comparison。

## 模型质量门槛

Qwen3.5 的模型质量比较与本性能表分开。要声明“全模型超过”，必须针对 0.8B/2B/4B/9B/27B 对应档分别固定 tokenizer、chat template、thinking 模式、采样参数、数据集 revision 和 evaluator commit，至少覆盖 MMLU-Pro、C-Eval、IFEval、GPQA、MATH-500、代码、多语种和长上下文。没有这些复测，本文只报告系统性能，不作质量领先结论。

## 达标规则

只有当 analyzer 对五档共 20 个工作负载全部返回 `pass`，多卡必需行完成真实验证，并且 Albatross/质量结论各自有独立同条件证据时，才允许写“全面超过”。当前准确结论是：

- 0.8B 对应档性能 4/4 通过；
- 2B 对应档短上下文和全部 decode/显存轴通过，长上下文 prefill 未通过；
- 4B/9B/27B 权重矩阵和真实多卡仍未执行；
- Albatross 同条件对比与模型质量对比仍未完成。
