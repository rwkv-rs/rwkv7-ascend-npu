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

## Optimization notes (vs naive per-op)

1. **Batched shift-mix**: stack `[x_r..x_g]` → 1 mul + 1 add.
2. **r/k/v via one bmm**.
3. **w_exp in fp16** (sigmoid range is exp-safe).
4. `at::NoGradGuard` skips autograd graph build.
5. All `at::layer_norm` / `at::group_norm` fused.

Single-seq 2× Albatross still needs GEMV-Cube fusion (multi-month); but
**batched aggregate throughput at B>1 already clears 2× Albatross** (see main
README).
