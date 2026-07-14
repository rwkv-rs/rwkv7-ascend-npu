# RWKV-7 Ascend vs Qwen3.5 与 Albatross Benchmark 规范

本文定义一套可复现的三方 benchmark，用于回答两个不同的问题：

1. 本项目的 RWKV-7 Ascend 后端与 Albatross RWKV 后端相比有多快；
2. RWKV-7 与 Qwen3.5 在性能、显存/内存和模型质量上各有什么取舍。

这两个问题必须分别报告。Albatross 是 RWKV-7 推理引擎参考，适合用同一份
RWKV checkpoint 比较后端；Qwen3.5 是不同模型，速度结果不能代替质量结果。
任何缺少同 checkpoint、同精度、同批量和同输入/输出长度的公开数字，只能列为
外部背景，不进入倍率或达标结论。

## 1. 对比范围

### 1.1 三条独立赛道

| 赛道 | 对比对象 | 控制变量 | 能回答的问题 |
| --- | --- | --- | --- |
| A：RWKV 引擎 | 本项目 vs Albatross | 同 RWKV checkpoint、dtype、B/T、cache 策略 | 两个后端在各自平台上的延迟和吞吐 |
| B：模型性能 | RWKV-7 vs Qwen3.5 | 同设备、同卡数、同原始文本、同并发、同输出预算 | 部署同档模型的性能和资源成本 |
| C：模型质量 | RWKV-7 vs Qwen3.5 | 同数据集、评测器、提示模板、采样参数 | 模型能力差异 |

跨平台的 Ascend/RTX 结果可以报告，但必须标为 **platform comparison**。只有在
同一张 NVIDIA 卡上复测相同 RWKV checkpoint 和工作负载，才能把本项目的 CUDA/HF
兼容路径与 Albatross 称为纯引擎对比。不同卡的结果不得使用“引擎已追平”措辞。

### 1.2 建议模型梯度

| RWKV-7 | Qwen3.5 | 用途 | 限制 |
| --- | --- | --- | --- |
| 0.4B（L24/D1024） | 0.8B（L24/D1024） | 首选结构对照 | 隐藏维度和层数接近，但参数量、词表和算子不同 |
| 1.5B | 2B | 小模型部署档 | 不是等参数配对 |
| 2.9B | 4B | 中等模型部署档 | 必须记录实测参数量与权重字节数 |
| 7.2B | 9B | 大模型部署档 | 单卡能否容纳也是结果的一部分 |

RWKV 官方的张量对照给出 L24/D1024 下 RWKV-7 约 `450.768M` 实际参数、
Qwen3.5 约 `752.393M` 参数；对应状态规模分别约为 `1.622M`，以及
`5.050M + 6.144M × (T/1000)`。这能解释二者的状态/上下文内存差异，但不能证明
模型质量相等。来源：[RWKV-7 / Qwen3.5 tensor comparison][rwkv-qwen-tensors]。

Qwen3.5-0.8B 官方模型卡说明其语言模型为 24 层、hidden size 1024，采用
`6 × (3 Gated DeltaNet + 1 Gated Attention)` 的混合布局，原生上下文为
262,144 tokens。本文只比较文字语言模型路径；启动时使用
`--language-model-only`，避免视觉编码器和多模态 profiling 干扰结果。

## 2. 当前已有证据

下表只说明已有数据的位置和量级，**不能横向计算倍率**。

| 状态 | 模型/形状 | 后端与设备 | 工作负载 | 已有结果 | 可用于正式三方倍率 |
| --- | --- | --- | --- | ---: | --- |
| 本仓库实测 | synthetic 0.1B shape，L12/H12/N64/V65536 | AscendC direct，Ascend 910B2C | B1T1 decode，resident cache | `1509.8–1537.3 tok/s` | 否：不是实际 checkpoint |
| 本仓库实测 | 同上 | AscendC direct，Ascend 910B2C | B1T1 decode，dynamic state slot | `1462.5–1481.7 tok/s` | 否：不是实际 checkpoint |
| Albatross 官方外部数据 | RWKV-7 7.2B fp16 | Albatross，单 RTX 5090 | B1T1 decode | `144.04 tok/s`（v3a 示例） | 否：模型、卡和后端条件不同 |
| Albatross 官方外部数据 | RWKV-7 7.2B fp16 | Albatross，单 RTX 5090 | B1T1024 prefill | `17000+ tok/s`（README 声明） | 否：模型、卡和口径不同 |
| Qwen 官方外部数据 | Qwen3.5-0.8B | 官方评测环境 | MMLU-Pro，non-thinking | `29.7` | 否：只作评测复现校验 |
| Qwen 官方外部数据 | Qwen3.5-0.8B | 官方评测环境 | MMLU-Pro，thinking | `42.3` | 否：只作评测复现校验 |

当前 Ascend 数字的正确性门槛为 64/64 greedy token 匹配、最小 logits cosine
`0.999999344`、fp32 state 最大差异 `0.00585938`。复现命令见
[`vllm-rwkv-ascend/perf/ascendc/direct/README.md`](vllm-rwkv-ascend/perf/ascendc/direct/README.md)。
Albatross 外部数字来自其[官方 README][albatross]，Qwen 分数来自
[Qwen3.5-0.8B 官方模型卡][qwen08]。

### 2.1 2026-07-14 单卡 910B2C 实际模型结果

下面的系统性能行来自同一张 Ascend 910B2C、fp16、B=1、prompt=512。
RWKV 使用官方 BlinkDL `RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth`
和本提交的 layer-major AscendC prefill scan；Qwen 使用 ModelScope 镜像的
`Qwen3.5-0.8B` 和官方 `vllm==0.18.0` / `vllm-ascend==0.18.0` strict eager
路径。Qwen 输出长度为 128；RWKV decode 是预填充后的 steady-state 单步中位数，
所以 prefill 与 decode 分别比较，不把两条计时拼成 E2E。

| 模型/引擎 | B | prompt | prefill tok/s | decode tok/s | 峰值显存 | 正确性 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RWKV-7 0.4B / 本项目 AscendC | 1 | 512 | `7338.37` | `502.8` | `1160.65 MiB`（prefill） | greedy pass；prefill logits cosine `0.999999642`；decode 128/128 |
| Qwen3.5-0.8B / vLLM-Ascend strict eager | 1 | 512 | `6256.09` | `44.9357` | 未采集 | 独立模型，不做 logits 对齐 |
| Qwen3.5-0.8B / Transformers NPU fallback | 1 | 512 | `3413.8` | `33.40` | `1748.6 MiB` | 官方 checkpoint load/generate |
| RWKV-7 0.4B / 本项目 AscendC | 4 | 512 | `10622.06` | pending | `1344.81 MiB`（prefill） | greedy pass；最低 logits cosine `0.999999881` |
| Qwen3.5-0.8B / vLLM-Ascend strict eager | 4 | 512 | `20886.71` | `168.272` | 未采集 | 独立模型，不做 logits 对齐 |
| Qwen3.5-0.8B / Transformers NPU fallback | 4 | 512 | `10821.8` | `131.71` | `1937.0 MiB` | 官方 checkpoint load/generate |

在 strict eager 的同卡行上，当前 RWKV prefill 为 Qwen 的 `1.173x`，decode 为
`11.189x`。这是 **0.4B 对 0.8B 的部署档系统性能结果**，不是同参数量结果，也不是
模型质量结论。B=4 prefill 当前仅为 vLLM-Ascend strict eager 的 `0.509x`，与
Transformers fallback 相比为 `0.982x`，所以该轴明确未达标。Qwen vLLM 图模式在
当前 x86/CANN 8.5.1 环境因官方运行时组合缺少 `torchair` 而无法初始化；v0.18
稳定安装文档现要求 CANN 9.0.0/torch-npu 2.9.0.post2。两次独立复测、在线 p99
和多卡仍为 `pending`；这些行补齐前不能写“全面超过 Qwen3.5”。

新的 prefill kernel 每个 block 持有一个 head 的 fp32 state 行分片，完整 prompt
期间保留在 UB，只把 fp16 token vectors 流入并写出 recurrent output。它把原来的
PyTorch layer-major B1 prompt512 从约 `1995.06 ms` 降到 `69.770 ms`，同时保持
fp32 state accumulation。B=1 回归的 64-token smoke 也通过，logits cosine 为
`0.999999642`。

### 2.2 “全面超过”的判定状态

| 轴 | 当前状态 | 达标条件 |
| --- | --- | --- |
| B1 prefill/decode | provisional pass | 官方图模式复测也非负，且两次独立测量 |
| B4/在线并发 | fail on prefill / decode pending | 当前 prefill `0.509x`；prefill、decode、TTFT/ITL/E2E p50/p99 全部非负才达标 |
| 显存与长上下文 | partial | 同引擎采集加载、prefill、decode 峰值及 2K/8K 斜率 |
| 多卡 | blocked by current host | 至少 2 卡，报告 speedup、efficiency 和每卡峰值 |
| 质量 | pending | 同评测器跑 MMLU-Pro、C-Eval、IFEval、GPQA、MATH、代码、长上下文 |
| Albatross | pending | 同 RWKV checkpoint、同卡、同 dtype/B/T/cache 实测 |

## 3. 测试前必须固定的信息

每个结果文件必须包含以下信息；缺少任一关键项的行标为 `informational`：

- 日期、主机名、设备完整名称、设备数量、SM/Ascend SoC、功耗/时钟模式；
- 驱动、CUDA 或 CANN、PyTorch、`torch_npu`、Triton 版本；
- 引擎名称、版本和 Git commit；
- 模型 ID、checkpoint revision/SHA256、tokenizer revision；
- 参数量、磁盘权重字节数、加载后模型内存；
- fp32/fp16/bf16/W8/W4、量化方法和量化范围；
- eager/compile/graph、启用的融合、cache 策略；
- 单卡/DP/TP/PP，设备映射和每卡峰值内存；
- warmup、迭代次数、随机种子、后台任务和统计方法。

性能复测至少 warmup 20 次、测量 200 次。报告 p10/p50/p90；在线服务额外报告
p95/p99。设备端计时必须在测量边界同步，首轮编译和模型加载不计入 steady-state，
但冷启动时间单列。

## 4. 统一工作负载矩阵

### 4.1 单步与离线推理

| 类型 | Batch | Prompt tokens | Output tokens | 必报指标 |
| --- | --- | --- | --- | --- |
| decode step | 1/4/16/64 | 已完成 prefill | 1 × 200 steps | p50/p90 ms、单序列与聚合 tok/s |
| short generation | 1/4/16 | 128 | 128 | prefill tok/s、decode tok/s、E2E |
| standard generation | 1/4 | 512 | 128/512 | TTFT、ITL、E2E、峰值内存 |
| long prefill | 1 | 2048/8192 | 1 | prefill tok/s、TTFT、峰值内存 |

Albatross 的 `B×T` kernel 行与完整 generate/serve 行分表记录。例如 `B1T512`
测的是一次 512-token 前向，不等同于 512-token 在线请求的全部开销。

### 4.2 在线服务

固定一份 UTF-8 JSONL prompt 集，分别测试并发 `1/8/32`：

- 无速率上限的 saturation throughput；
- 固定 request rate 下的稳定性；
- request/s、input/output tok/s、TTFT p50/p95/p99；
- inter-token latency p50/p95/p99、E2E p50/p95/p99；
- 超时率、失败率、队列时间、峰值内存。

### 4.3 多卡

单卡结果与多卡结果必须分表。多卡至少跑 `2` 卡，并注明并行策略：

- 本项目：一进程一卡的吞吐扩展与实际支持的 PP/服务调度路径分别报告；
- Albatross：使用 `--pp-devices 0,1` 时标为 PP，不与单卡延迟混淆；
- Qwen3.5/vLLM：使用 `--tensor-parallel-size 2` 时标为 TP；
- 报告 `speedup = multi_card_tok_s / single_card_tok_s` 和
  `efficiency = speedup / device_count`；
- 每卡峰值内存、跨卡通信量和最慢卡利用率必须记录。

当前 direct AscendC 路径只有单 910B2C 的新融合证据，因此多卡状态应保持
`pending`，直到同一 commit 上完成正确性和性能复测。

## 5. Tokenizer 与计时公平性

RWKV 和 Qwen 使用不同 tokenizer，不能把同一个 token ID 数组喂给两者，也不能
只凭 tok/s 判定谁处理真实文本更快。正式模型对比应：

1. 使用完全相同的原始 UTF-8 prompt；
2. 分别用各自 tokenizer 编码，并记录实际 input/output token 数；
3. 同时报 `tok/s`、`UTF-8 bytes/s` 和 `Unicode chars/s`；
4. 离线引擎计时排除 tokenizer，客户端 E2E 另行包含 tokenizer 与网络；
5. 生成性能行使用 greedy decoding（temperature 0）以消除采样差异；
6. 质量评测按各模型/评测集规定的采样参数，不能复用性能行的 greedy 分数。

Qwen3.5 的 thinking 与 non-thinking 必须分开。官方 0.8B 卡默认 non-thinking，
并为 thinking 质量评测给出 `temperature=1.0, top_p=0.95, top_k=20,
presence_penalty=1.5`；复现官方分数时使用官方设置，性能对比仍使用 greedy。

## 6. 正确性与质量门槛

### 6.1 同 checkpoint 的 RWKV 引擎正确性

- 128/128 greedy token 与 PyTorch/reference 匹配；
- fp16 目标 `min logits cosine >= 0.9999`；
- 记录 logits max abs diff 和 fp32 recurrent-state max abs diff；
- single-shot prefill 与 chunked prefill 的 cache handoff 一致；
- B1/B4 dynamic slot select/reorder/drop 不串状态；
- graph resident 与 dynamic state-slot 行分别报告。

只有通过正确性门槛的性能行才能进入主表。近似路径必须显示其阈值，不得只写
`pass`。RWKV 与 Qwen 是不同模型，不做 logits 或 greedy-token 对齐。

### 6.2 模型质量

建议最小文本评测集如下：

| 能力 | 数据集 | 必报设置 |
| --- | --- | --- |
| 综合知识 | MMLU-Pro | exact match、语言、thinking mode |
| 中文 | C-Eval | exact match、prompt template |
| 指令遵循 | IFEval | strict/loose accuracy |
| 科学推理 | GPQA Diamond | sampling、pass@1 |
| 数学 | MATH-500 | answer extractor、pass@1 |
| 代码 | HumanEval+ 或 LiveCodeBench | evaluator commit、pass@1 |
| 多语言 | MMMLU 或 MMLU-ProX | 语言集合、macro average |
| 长上下文 | LongBench v2 | 实际上下文长度、截断策略 |

每行记录 evaluator commit、数据集 revision、chat template、system prompt、最大输出、
采样参数和重复次数。官方分数只用于 sanity check；最终结论必须来自相同评测器的
本地复测。速度领先不能写成质量领先。

## 7. 复现命令

### 7.1 当前 Ascend 910B2C synthetic direct 行

先按 [direct backend README](vllm-rwkv-ascend/perf/ascendc/direct/README.md)
完成构建，然后在 `vllm-rwkv-ascend/` 下运行：

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh

RWKV7_ASCENDC_DIRECT_BUILD_DIR=/tmp/rwkv7_direct/build \
taskset -c 31 python perf/bench_graph_overhead.py \
  --device npu:0 --layers 12 --heads 12 --head-size 64 \
  --vocab-size 65536 --warmup 200 --iterations 2000 \
  --correctness-steps 64 --compare-ascendc-direct \
  --direct-fractal-nz --direct-nz-lm-head --direct-dplr-state \
  --direct-rank1-row-blocks 2 --direct-lowrank-bmm --direct-rkv-bmm \
  --direct-mix-project --direct-recurrence-prep \
  --direct-fused-recurrence-state --direct-inplace-state \
  --direct-fused-ffn-prep --direct-fused-next-attn \
  --direct-fused-final-norm --direct-fused-embed-norm2
```

这是 kernel/graph 形状基准，不是实际模型质量或三方正式结果。实际 RWKV checkpoint
的基础性能 smoke 可运行：

```bash
cd vllm-rwkv-ascend
PYTHONPATH=/path/to/rwkv7-hf-adapter:. \
python perf/run_perf.py /path/to/rwkv7-hf-checkpoint
```

`perf/run_perf.py` 当前固定测试 B=1/8/16/32、3 次 warmup 和 30 次测量；正式表需要
后续把它的原始结果转换到本文规定的 20/200 统计口径。

### 7.2 实际 RWKV PTH 与 Qwen3.5-0.8B

构建 direct kernels 后，在同一张 910B2C 上运行：

```bash
RWKV7_ASCENDC_DIRECT_BUILD_DIR=/tmp/rwkv7_direct/build \
taskset -c 31 python perf/bench_rwkv7_pth_prefill.py \
  --model-pth /path/to/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth \
  --device npu:0 --batch-size 1 --prompt-length 512 \
  --correctness-length 64 --ascendc-scan --warmup 3 --iterations 10 \
  --output results/rwkv7-0.4b-b1-p512.json

python perf/bench_qwen35_vllm_ascend.py \
  --model /path/to/Qwen3.5-0.8B \
  --batch-size 1 --prompt-length 512 --decode-length 128 \
  --max-model-len 1024 --gpu-memory-utilization 0.5 --enforce-eager \
  --output results/qwen35-0.8b-b1-p512-d128-eager.json
```

Qwen 主表优先使用不带 `--enforce-eager` 的官方图模式行；strict eager 只在图模式
运行时不兼容时作为明确标注的 fallback。失败日志也要保留，不允许用失败图模式的
初始化耗时参与倍率。

### 7.3 Albatross

必须固定 commit，并使用与待测 RWKV 后端完全相同的 `.pth` checkpoint：

```bash
git clone https://github.com/BlinkDL/Albatross.git
cd Albatross
git checkout <PINNED_COMMIT>
cd faster3a_2605

python rwkv7_fast_v3a.py \
  --model /path/to/same-rwkv7.pth \
  --wkv fp16 --emb gpu \
  --warmup 20 --iters 200 \
  --cases "1x1,1x128,1x512,4x1,16x1,64x1"
```

该脚本原生输出 p10/p50/p90 和 `tok_s_p50`。同时保存 `--wkv`、`--emb`、
`--batched-rkv`、`--cmix-sparse`、`--lowrank-weight`、
`--orig-linear-groups` 等完整参数。命令行定义见
[Albatross `rwkv7_fast_v3a.py`][albatross-v3a]。

### 7.4 Qwen3.5 / vLLM 文本路径

Qwen3.5 当前要求较新的 vLLM；正式结果要固定 vLLM commit/版本。为了与纯文本
RWKV 公平比较，限制相同的 benchmark 上下文并关闭视觉路径：

```bash
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --language-model-only
```

离线吞吐示例：

```bash
mkdir -p results
vllm bench throughput \
  --model Qwen/Qwen3.5-0.8B \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 256 \
  --output-json results/qwen35-0.8b-b1-p512-o128.json
```

在线服务使用固定 prompt JSONL 和 `vllm bench serve`。vLLM benchmark 参数会随版本
演进，运行前保存 `vllm bench serve --help`，并把完整命令写入结果元数据。吞吐参数
定义见 [vLLM 官方文档][vllm-throughput]；Qwen 的启动参数来自
[Qwen3.5-0.8B 官方模型卡][qwen08]。

## 8. 结果表模板

### 8.1 环境

| run_id | commit | model@revision | engine@commit | device × count | runtime | dtype | parallel/cache | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | pending |

### 8.2 性能与内存

| run_id | lane | model | B | prompt | output | prefill tok/s | decode tok/s | bytes/s | TTFT p50 | ITL p50/p99 | E2E p50/p99 | peak/device | correctness |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TODO | RWKV engine / model perf / serving | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | pending |

### 8.3 多卡扩展

| run_id | model/engine | strategy | devices | tok/s | speedup | efficiency | peak/device | correctness |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | TP/PP/DP | 1/2 | TODO | TODO | TODO | TODO | pending |

### 8.4 质量

| model@revision | mode | benchmark@revision | score | prompt/evaluator commit | sampling | status |
| --- | --- | --- | ---: | --- | --- | --- |
| RWKV-7 | standard | TODO | TODO | TODO | TODO | pending |
| Qwen3.5 | non-thinking | TODO | TODO | TODO | TODO | pending |
| Qwen3.5 | thinking | TODO | TODO | TODO | TODO | pending |

建议将原始 JSON/JSONL 保存到
`bench/results/qwen35_albatross/<YYYYMMDD>-<device>/`，并保留 stdout、环境快照、
命令行和失败行。汇总表只引用原始 `run_id`，不手工覆盖原始数据。

## 9. 结论与达标规则

- **Albatross ratio**：只对同 RWKV checkpoint、dtype、B/T 和 cache 策略的行计算；
  跨设备时必须显示两侧设备名称并标为 platform ratio。
- **Qwen performance ratio**：要求同一设备、同一卡数、同原始 prompt 和输出预算；
  同时报 tok/s 与 bytes/s。
- **显存/内存结论**：同时报告权重、空闲加载、prefill 峰值、decode 峰值，以及
  context 增长斜率。
- **质量结论**：必须有同评测器复测行；thinking 与 non-thinking 不合并。
- **多卡结论**：正确性通过后再报告扩展效率；单卡和多卡不得取各自最优后拼成
  一条“综合最快”结论。
- **正式达标**：主表全部关键行有原始数据、环境完整、正确性通过、至少两次独立
  复测，且 p50 差异大于测量噪声。否则结论保持 `pending` 或 `informational`。

[albatross]: https://github.com/BlinkDL/Albatross
[albatross-v3a]: https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/rwkv7_fast_v3a.py
[qwen08]: https://huggingface.co/Qwen/Qwen3.5-0.8B
[rwkv-qwen-tensors]: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.md
[vllm-throughput]: https://docs.vllm.ai/en/stable/cli/bench/throughput/
