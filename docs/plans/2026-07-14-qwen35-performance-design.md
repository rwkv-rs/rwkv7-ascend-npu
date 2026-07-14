# RWKV-7 Ascend 超过 Qwen3.5 的性能设计

## 目标与验收口径

第一阶段使用一张 Ascend 910B2C，对比实际 RWKV-7 0.4B checkpoint 与
Qwen3.5-0.8B 文本语言模型。两者都使用 fp16、相同卡数和形状控制工作负载，主矩阵为
B=1/4、prompt=512、decode=128。必须同时记录 prefill tok/s、decode tok/s、p50/p90
延迟、模型权重字节数和每张卡峰值内存。RWKV 行还要通过默认 C++ 路径与 direct
AscendC 路径的 greedy token、logits cosine 和 recurrent state 对齐。不同 tokenizer
的真实文本行另外报告 bytes/s，不用 token ID 直接跨模型比较。

“全面超过”不是单个 synthetic B1T1 数字，而是上述 B1/B4 的 prefill、decode 和峰值
内存全部非负，同时正确性通过。第一阶段通过后扩到 RWKV-7 1.5B 与 Qwen3.5-2B，
再补在线 TTFT/ITL/p99 和双卡扩展。Qwen 的 thinking/non-thinking 质量不属于本性能
阶段；速度领先不能替代质量评测。

## 实现结构

新增独立 BlinkDL `.pth` loader，把官方权重映射成现有 `rwkv7_decode_full` 所需的
HF 风格 `[out,in]` 线性权重。真实 RWKV-7 的 w/a/g/v 低秩维度并不相同，因此四路
BMM 使用每层最大 rank，并对较小矩阵零填充。零填充在第一、第二个低秩矩阵中成对
出现，不改变输出；tanh/identity 的填充值为零，sigmoid 路径即使产生 0.5，也会被
第二矩阵的零行消除。CPU 单测直接对比填充前后的四路计算。

实际 checkpoint 首轮不启用 folded mix-project。该 synthetic 优化会复制一份宽度
为 `2H` 的大权重，在 0.4B 上额外占用约数百 MiB，破坏内存目标。首轮采用 fused
shift-mix、packed RKV BMM、padded low-rank BMM、fused recurrence-state 和 fused
FFN preparation。现有参数校验会放宽为 recurrence preparation 可由 low-rank BMM
提供输入，而不是强制依赖 mix-project。

Qwen3.5 新增独立 HF/NPU benchmark。脚本优先加载文本 causal LM，必要时从多模态
wrapper 取 language model；使用 cache decode，并在模型支持时只保留最后一个 token
的 logits，避免把全序列词表投影误算进 serving prefill。结果写 JSON，包含环境、
模型字节数、延迟分位数、吞吐和峰值内存。

## 错误处理与验证

loader 在层数、hidden/head-size 整除、低秩矩阵转置关系、bias 宽度和词表维度不一致
时立即失败。实际 direct 路径仍保持 opt-in，默认 benchmark 和 serving 行为不变。
CPU CI 覆盖不等 rank 打包与 folded projection 数学；910B2C 覆盖 clean build、实际
checkpoint load、默认/direct 首 token logits、至少 64 步 greedy/cache 对齐，以及
B1/B4 性能。任何一项失败都保留原始日志并把结果标为 failed，不把最快但错误的行
写入主表。

远端工作在独立 worktree 和虚拟环境进行，避免覆盖原服务器的未提交内容。最终只把
经过本地 CPU 测试和 NPU 实测的源文件提交到现有
`wangyue/ascend-captured-embedding` PR 分支。

## 2026-07-14 实测结论

实际 PTH decode 达到 `502.8 tok/s`，128/128 greedy 对齐；新的 AscendC
layer-major prefill scan 在 B1/P512 达到 `7338.37 tok/s`，最低 logits cosine
`0.999999642`，超过同卡 vLLM-Ascend strict eager Qwen3.5-0.8B 的
`6256.09 tok/s`。B4/P512 正确性同样通过，但只有 `10622.06 tok/s`，对 Qwen 的
`20886.71 tok/s` 为 `0.509x`，所以“全面超过”尚未完成。

瓶颈探针排除了三个方向：每 head 单/四 row-block、局部 state 双缓冲交换、以及纯
PyTorch fp32/bf16 三角求解版 compact-WY 都没有端到端收益；通用 Triton compact
kernel 在当前 x86/CANN 8.5.1 的 Triton-Ascend 后端编译阶段崩溃。下一实现边界是
AscendC Cube 上的专用 compact-WY/chunked scan。它必须保留 fp32 state、B1/B4
正确性门槛，并以 B4 prefill 超过 `20886.71 tok/s` 为第一性能门槛。隔离的 64×64
Cube 探针还确认：常量 tiling 仍需 host 分配系统 workspace；补齐 workspace 后，简化
direct launch 在首个矩阵上进入设备侧死锁。因此下一版要使用正式 op-host tiling 与
workspace 生命周期，不能把该实验直连包装器并入生产路径。
