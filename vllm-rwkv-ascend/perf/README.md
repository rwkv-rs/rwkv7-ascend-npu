# perf/ — C++ op-coalesced RWKV-7 forward (the fast NPU path)

`rwkv7_ascend_v3.cpp` runs the **entire 12-layer TMix+CMix forward in one C++
call** (via `at::` / aclnn ops), eliminating the ~15-op Python dispatch per layer
that makes the pure-PyTorch shim slow. Proven on 910B: **323 tok/s (B=1),
cos=1.0** vs the HF-native reference.

## State writeback (multi-step correctness)

The `RWKV7_BODY` macro must write the evolved recurrent state back into the
Python-passed tensors, **after** they've been read — three in-place copies per
layer: `state_all[li].copy_(state)` (recurrent WKV state), `xpa_all[li].copy_(h)`
(attn input), `xpf_all[li].copy_(h2)` (ffn input). (`v_first` needs none —
overwritten at layer 0 each step.) Without these, the macro's local reassignment of
`state` is lost and **multi-step generation collapses to a fixed cycle** (single-step
cos=1.0 still holds, which masked the bug — `bench_batch` re-zeroed state each call).
This fix is what makes the forward usable for actual generation / serving.

## When to use which path

| Path | File | Speed | Use |
|---|---|---|---|
| Correctness (shim) | `../rwkv7_npu_ops.py` | slow (Python dispatch) | verify ops, Albatross-structure parity |
| **Perf (C++ coalesced)** | `rwkv7_ascend_v3.cpp` | **323 tok/s B=1**, batch → 2× Albatross aggregate | real inference / serving |

The perf path loads weights the **HF-adapter** way (`NativeRWKV7ForCausalLM`) —
the C++ forward is built around the adapter's layer attributes, not the Albatross
faster3a weight layout. It is a parallel fast forward, self-contained on NPU.

## Run (on 910B3)

```bash
PYTHONPATH=/root/rwkv7-ascend:. python perf/run_perf.py /root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf
# prints: correctness cos + B=1/8/16/32 tok/s
```

First call compiles the extension (~30 s, cached afterwards by torch cpp_extension).

## 910B2C direct fused result (2026-07-13)

The opt-in direct AscendC backend combines the DPLR rank-one recurrence, state
output, groupnorm/SK/gating, key normalization, packed low-rank paths, static
shift-mix projection folding, and a fused recurrence-preparation/state kernel.
On one Ascend 910B2C with CANN 8.5.1, the synthetic 12-layer `H=12`, `N=64`,
vocab-65536 row passes 64/64 greedy tokens with minimum logits cosine
`0.999999344` and maximum state difference `0.00585938`.

Two pinned 200-warmup/2000-iteration confirmations measure `0.650` and `0.662`
ms/token (`1537.3` and `1509.8 tok/s`) for graph-resident recurrent state. The
dynamic state-slot scheduler measures `0.675-0.684 ms/token`
(`1462.5-1481.7 tok/s`). The resident row therefore clears the provisional
`1500 tok/s` development target, while the dynamic row does not. Because
`1500 tok/s` is not a same-checkpoint/same-card Albatross measurement, this is
not a general Albatross parity claim. The folded projection also adds about
90 MiB for this synthetic shape and therefore remains opt-in. See
`ascendc/direct/README.md` for the exact build and benchmark command.

The validation host exposes only one Ascend device. Prior two-process graph-path
isolation evidence does not validate this new direct kernel, so fresh direct-path
multi-card evidence remains pending a machine with at least two visible NPUs.

To measure production graph overhead without downloading a model, run:

```bash
python perf/bench_graph_overhead.py --warmup 10 --iterations 100
```

The probe compares pure replay, the legacy external-embedding path, and the default
fixed-token-buffer path that captures embedding lookup inside NPUGraph.  It also
requires bit-exact logits and recurrent state between the two production paths.

Add `--compare-greedy` to measure the end-to-end greedy loop with argmax and the next
token captured inside the graph.  `--compare-addcmul` is a negative-test harness: it
reproduces a faster fp16 shift-mix whose recurrent numerical drift blocks production
use.

For multi-card isolation, launch one process per runtime-visible device and use
`--device npu:0` inside each restricted process.  Device numbers shown by host
`npu-smi` may differ from the runtime IDs exposed inside a container:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python perf/bench_graph_overhead.py \
  --device npu:0 --compare-greedy &
ASCEND_RT_VISIBLE_DEVICES=1 python perf/bench_graph_overhead.py \
  --device npu:0 --compare-greedy &
wait
```

## Optimization notes (vs naive per-op)

1. **Batched shift-mix**: stack `[x_r..x_g]` → 1 mul + 1 add.
2. **r/k/v via one bmm**.
3. **w_exp in fp16** (sigmoid range is exp-safe).
4. `at::NoGradGuard` skips autograd graph build.
5. All `at::layer_norm` / `at::group_norm` fused.

Custom AscendC Cube GEMV is not the single-sequence path: the measured B=1 kernel is
33x slower than `at::linear`.  Continue reducing graph-external work and coalescing
elementwise/recurrence operations instead.  Batched aggregate throughput at B>1
already clears 2x Albatross (see the main README).
