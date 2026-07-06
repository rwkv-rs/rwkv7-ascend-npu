# Ascend 910B — RWKV-7 推理:结果与路线图

> **结论:2× Albatross 在 aggregate 吞吐上达成并验证**(batch decode,cos=1.0)。
> 单序列延迟 2× 需 GEMV-Cube 融合(多月工程),elementwise piecemeal 路径已证伪。

0.1B-scale,fp16,Ascend 910B2C 64GB,torch_npu 2.9.0rc1。速度数字早期用随机权重(服务器被墙);**真实权重现已下全并验证——0.1B–13.3B 全部对齐 V100 CUDA,cos 0.99997–1.0,argmax 100% 一致**,详见 [`validation/real_weights_v100.md`](validation/real_weights_v100.md)。

---

## 1. 核心结果:batch decode 实现 2× Albatross(aggregate 吞吐)

forward 是 **launch-overhead-bound**(~960 个 CANN kernel launch,每个 ~3μs;NPU kernel 执行固定成本,`torch.npu.NPUGraph` replay 实测不加速:312 vs 323 tok/s)。batch 把固定 launch 开销 + 投影 GEMM 化一起摊薄 → forward 时间 1→128 batch 只涨 ~3×。

**aggregate tok/s(全规模,B>1 已验证 cos=1.0):**

| model | B=1 | B=4 | B=16 | B=64 | B=128 |
|---|---|---|---|---|---|
| 0.1B (H12/N64/L12) | 323 | 545 | 3433 | 9446 | **13504** |
| 1.5B (H32/N64/L32) | 87 | 304 | 953 | 2121 | — |
| 2.9B (H48/N64/L28) | 69 | 230 | 680 | 1748 | — |
| 7B (H64/N64/L48) | 33 | 113 | 313 | 793 | — |
| 13B (H64/N64/L64) | 25 | 84 | 235 | 585 | — |

- 0.1B B=8/16/64/128 全部进/超 2× Albatross 区间(~1500–3000 aggregate);B=128 = 13504(~9×)
- 大模型 tok/s 更低(每 token 计算更多),但 batch 仍 ~10× 放大吞吐;13B B=64 = 585(31GB/64GB,还能加 B)
- B=4 四条不同 token 序列,每条 cos=1.00000 + argmax 一致 → 高 tok/s 是算对的前提下拿到的
- per-seq 延迟随 B 降(0.1B 323→105);这是 aggregate **吞吐**(serving 指标),不是单序列延迟

## 2. C++ op-coalesced forward(单序列基线)

把 12 层 TMix+CMix(~960 个 torch op)收进**一次 C++ 调用**,消除 Python dispatch。
- **323 tok/s(B=1),cos=1.00000**(bit-exact 对齐 Python eager),1.83× Python
- 修了一个真 correctness bug:**本地 `native.py` 的 `g` 比 server main 多一层 sigmoid** → C++ 从 cos=0.83 修到 1.0。教训:移植前先 diff server 上的实际源码,别信本地分支
- 缺失 LayerNorm(v1 cos≈0):补上 pre/attn/ffn/final norm(`at::layer_norm`/`at::group_norm`,CANN 融合 kernel)

逐项验证过、都不再加分:batched shift-mix(更慢,NPU 不融合 stack/cat)、bmm r/k/v(无收益)、NoGradGuard、torch.compile(失败/更慢)、triton-npu(不在 PyPI)、torchair(未装)。

## 3. 单序列延迟方向:AscendC 融合 —— 工具链全通,elementwise piecemeal 证伪

**工具链 + 积木全验证**(详见 `ascendc/README.md`):
- 全链路:op-def JSON → msopgen gen → kernel → **910b 编译安装**(`AddConfig("ascend910b")` 必加) → torch aclnn 调用(5 个 CANN 坑记录)
- V-V 同步:专用 `TQue<VECCALC>` 中间队列 + EnQue/DeQue(in-place / TBuf / PipeBarrier-alone 都出垃圾)
- 多输入 Mul+Add cos=0.9997;2-output 共享复用 cos=0.9999;6-output BN=1([8192] cos=1.0,2.39× 隔离)
- `Sub` 在 910b half 失灵(绕:Mul(−1)+Add);`Exp` 链式炸;6-output BN=2 挂死(BN=1 修复)

**为什么 elementwise piecemeal 到不了(loop 节奏)**:
- shift-mix 是 forward 的 ~16%(156 op),融合隔离 2.39×,但**集成进 forward 时**:① sm6 op 在 forward 形状 `[1,768]` 有小尺寸 tiling bug(`[8192]` cos=1.0,`[1,768]` cos≈0);② 集成开销(14 个 aclnn tensor 转换 + 6 个 mix 广播 copy × 12 层 + 每次 aclnn 框架)吃掉收益 → v4 实测 334 tok/s 但 cos=nan
- **真正的单序列 2× = GEMV 融进 AscendC Cube kernel**(真正 Albatross 等价物,多月工程)

## 4. 复现

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
cd /root/rwkv7-adapter && export PYTHONPATH=/root/rwkv7-adapter
python3 bench_batch.py      # C++ forward + batch 吞吐 (0.1B)
python3 bench_models.py     # 多模型 batch (0.1B→13B,改 configs=[...])
python3 verify_batch.py     # 正确性 (B=4, 每序列 cos vs Python)
python3 fast_generate_npu.py # 单序列快速 generate
```

## 5. 进度时间线(commit 在 `wangyue/ascend-cpp-correct-forward`,PR #1)

- C++ op-coalesced forward 323 tok/s cos=1.0(g-sigmoid bug 修复)
- NPUGraph 实验证明瓶颈是 kernel 执行(不是 dispatch)→ 只有融合能突破
- batch decode 2× Albatross 达成(0.1B B=128=13504),B>1 cos=1.0 验证
- 多模型 batch(1.5B/2.9B/7B/13B)全覆盖
- AscendC 工具链 + 积木全验证;elementwise 单序列融合证伪(op bug + 集成开销);GEMV-Cube 为单序列 2× 的唯一现实路径(多月)
