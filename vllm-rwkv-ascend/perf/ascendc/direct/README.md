# Direct AscendC decode kernels

This directory contains the opt-in B=1 decode backend used by
`perf/bench_graph_overhead.py` and the batched N=64 prefill scan used by
`perf/bench_rwkv7_pth_prefill.py`. The normal `rwkv7_decode_full` Python call
contract remains available; packed direct weights and the prefill scan are
selected only by explicit benchmark flags.

## Build

Set the CANN environment first, then use the exact SoC reported by `npu-smi`.
For the validated Ascend 910B2C machine:

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
cmake -S perf/ascendc/direct -B /tmp/rwkv7_direct/build \
  -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.1 \
  -DTORCH_PATH=/usr/local/python3.11.14/lib/python3.11/site-packages/torch \
  -DTORCH_NPU_PATH=/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu \
  -DGLIBCXX_USE_CXX11_ABI=1
cmake --build /tmp/rwkv7_direct/build -j
```

Run the real BlinkDL-checkpoint prefill row with:

```bash
RWKV7_ASCENDC_DIRECT_BUILD_DIR=/tmp/rwkv7_direct/build \
taskset -c 31 python perf/bench_rwkv7_pth_prefill.py \
  --model-pth /path/to/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth \
  --device npu:0 --batch-size 1 --prompt-length 512 \
  --correctness-length 64 --ascendc-scan --warmup 3 --iterations 10 \
  --output results/rwkv7-0.4b-b1-p512-ascendc.json
```

On the validated 910B2C, the B1/P512 median is `69.770 ms` or `7338.37
tok/s`, with greedy match, logits cosine `0.999999642`, and `1160.65 MiB`
peak allocated memory. The scan keeps fp32 state-row tiles in UB across the
complete prompt and streams fp16 token vectors; projections remain on the
vendor batched-matmul path. The former PyTorch recurrent scan took about
`1995.06 ms` for the same shape.

The B4/P512 row is `192.806 ms` or `10622.06 tok/s`, with greedy match,
minimum logits cosine `0.999999881`, and `1344.81 MiB` peak allocated memory.
This is a correctness-passing batch implementation, but it is not yet the
performance winner: official vLLM-Ascend strict eager measures `20886.71
tok/s` for Qwen3.5-0.8B on the same card and workload. Scan-only probing shows
the recurrent kernel dominates B4; the one- and four-row-block variants and
local-state ping-pong all regressed and are excluded. The next B4 step is
compact-WY/Cube prefill, not wrapper fusion.

Run the current target row with:

```bash
RWKV7_ASCENDC_DIRECT_BUILD_DIR=/tmp/rwkv7_direct/build \
python perf/bench_graph_overhead.py \
  --device npu:0 --layers 12 --heads 12 --head-size 64 \
  --vocab-size 65536 --warmup 200 --iterations 2000 \
  --correctness-steps 64 --compare-ascendc-direct \
  --direct-fractal-nz --direct-nz-lm-head --direct-dplr-state \
  --direct-rank1-row-blocks 2 --direct-lowrank-bmm \
  --direct-rkv-bmm --direct-mix-project --direct-recurrence-prep \
  --direct-fused-recurrence-state --direct-inplace-state \
  --direct-fused-ffn-prep --direct-fused-next-attn \
  --direct-fused-final-norm --direct-fused-embed-norm2
```

Pinning the host launch thread (`taskset -c 31` on the validated machine)
reduces scheduling noise. Two independent pinned confirmations produced:

| cache policy | latency | throughput | provisional 1500 tok/s ratio |
| --- | ---: | ---: | ---: |
| graph-resident, run 1 | 0.650 ms | 1537.3 tok/s | 1.025x |
| graph-resident, run 2 | 0.662 ms | 1509.8 tok/s | 1.007x |
| dynamic state slots, run 1 | 0.675 ms | 1481.7 tok/s | 0.988x |
| dynamic state slots, run 2 | 0.684 ms | 1462.5 tok/s | 0.975x |

The policies are reported separately: `ascendc_resident` keeps the recurrent
cache inside the captured graph, while `ascendc_shift_mix1` performs scheduler
state-slot round trips. The provisional reference is a development target, not
a fresh Albatross result on this exact card and checkpoint.

In the adjacent 400-iteration split/fused A/B, recurrence-state fusion moved the
resident row from `0.689 ms` (`1451.4 tok/s`) to `0.651 ms`
(`1536.2 tok/s`). A Level1 profile reports `113.462 us` total across the 12
fused recurrence-state launches; the first is `14.300 us` and the remaining
layers are `8.720-10.040 us` each.

## Current specialization

- B=1 decode, fp16 activations and fp32 recurrent state.
- Batched prefill scan for N=64, fp16 vectors, and fp32 recurrent state. It
  uses two row blocks per head on the validated 910B2C; a one-block B4 probe
  was slower because the larger per-block row loop outweighed fewer waves.
- The performance row is tuned for `H=12`, `N=64`, 12 layers on 910B2C.
- Two row blocks per head occupy the card's 24 Vector Cores. Larger row-block
  counts are intentionally rejected because this launch does not execute a
  second block wave correctly on the validated runtime.
- `direct-mix-project` assumes equal rank for the four packed low-rank paths
  and adds a large static folded weight. It is benchmark-only until a real
  checkpoint proves that the speed/memory tradeoff is worthwhile.
- `direct-fused-recurrence-state` removes the standalone recurrence-preparation
  launch and the global `w/kk/a` intermediates. Each of the two row blocks
  redundantly prepares one head's short vectors, avoiding an unsafe cross-block
  synchronization while retaining fp32 state accumulation.
- The 2026-07-13 910B2C host exposes one NPU (`npu-smi Total Count: 1`). The
  existing graph path has prior two-card process-isolation evidence, but this
  direct kernel set still requires a fresh two-card run before a multi-card
  claim is made.
- Approximate DPLR acceptance requires all greedy tokens to match and minimum
  logits cosine similarity of at least `0.9999`.

The important fused boundaries are recurrence preparation plus rank-one state
update/output, compact groupnorm/SK/gating, packed low-rank activation,
residual/normalization preparation, final residual plus normalization, and
embedding plus the two initial normalizations.
