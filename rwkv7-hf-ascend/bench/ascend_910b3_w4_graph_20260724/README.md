# Ascend 910B3 HF W4 NPUGraph rejection gate

This directory records the strict production gate for the public
`rwkv7_hf.quantize_ascend_w4a16_candidate(...)` API. The measured candidate
passes memory and latency, but fails model-quality gates and therefore remains
explicitly experimental.

## Exact scope

- device: 1x Ascend 910B3, 64 GiB HBM
- CANN: 8.5.0
- PyTorch / torch_npu: 2.9.0+cpu / 2.9.0
- model: RWKV-7 7.2B `fla-hub/rwkv7-7.2B-g0a`
- dtype: FP16 activations
- quantization: affine group-128 W4A16
- replaced modules: all 32 FFN `value` projections
- backend: canonical HF `generate()` with fixed-batch NPUGraph decode
- measured logical rows: B1, B4 and B8

## Memory and paired speed

The candidate contains no floating copy of a replaced weight. Model tensor
payload and isolated active HBM fall to 78.09% and 79.53% of FP16. Every timed
row uses seven alternating dense/quant pairs after capture.

| Batch | FP16 tok/s | W4 tok/s | Median paired W4/FP16 |
|---:|---:|---:|---:|
| 1 | 25.9486 | 27.1072 | 1.0453x |
| 4 | 95.2419 | 98.3094 | 1.0321x |
| 8 | 174.8964 | 181.1011 | 1.0370x |

The fixed 32-token timing prompt is identical under FP16 and W4 at every batch.

## Why production admission is rejected

Ten production prompts plus one synthetic diagnostic exercise 88 forced decode
steps. The candidate fails five independent quality gates:

- minimum logit cosine: `0.99847436` (required `0.999`)
- maximum normalized RMSE: `0.16532364` (required `0.05`)
- maximum production-corpus loss delta: `0.27381957` (required `0.02`)
- only six of ten production prompts have identical eight-token greedy output
- seven argmax changes include rank-3 choices and non-near-tied changes

The measured speed and HBM wins do not override those failures.
`should_quantize(...)` remains false and the public API still requires the
visibly experimental `require_explicit_candidate=False` argument.

## Reproduce

The command is expected to write a `FAIL` artifact and exit with status 1:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
PYTHONPATH=rwkv7-hf-ascend python \
  rwkv7-hf-ascend/bench/run_w4_graph_e2e.py \
  --model /path/to/rwkv7-7.2b \
  --new-tokens 32 \
  --corpus-new-tokens 8 \
  --quality-steps 8 \
  --rounds 7 \
  --output /tmp/hf_w4_graph_e2e.json
```

Other cards, software stacks, dtypes, group sizes, projection selections and
batch rows have no production claim.
