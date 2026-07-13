# Direct AscendC decode kernels

This directory contains the opt-in B=1 fused backend used by
`perf/bench_graph_overhead.py`. The normal `rwkv7_decode_full` Python call
contract remains available; packed direct weights are selected only by the
benchmark flags below.

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
