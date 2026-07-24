# Ascend 910B3 HF W8 NPUGraph production gate

This directory records the production-admission gate for the public
`rwkv7_hf.quantize_ascend_w8a16(..., policy="speed")` API.

## Exact scope

- device: 1x Ascend 910B3, 64 GiB HBM
- CANN: 8.5.0
- PyTorch / torch_npu: 2.9.0+cpu / 2.9.0
- model: RWKV-7 7.2B `fla-hub/rwkv7-7.2B-g0a`
- dtype: FP16
- quantization: symmetric W8A16 per-output-channel
- replaced modules: all 64 FFN `key` and `value` projections
- backend: canonical HF `generate()` with fixed-batch NPUGraph decode
- admitted logical rows: B1, B4 and B8 only

## Memory and paired speed

The quantized model contains no floating copy of a replaced weight. Model
tensor payload and isolated active HBM fall to 70.18% and 71.48% of FP16.
Every timed row uses five alternating dense/quant pairs after capture.

| Batch | FP16 tok/s | W8 tok/s | Median paired W8/FP16 |
|---:|---:|---:|---:|
| 1 | 25.8451 | 26.4352 | 1.0241x |
| 4 | 94.2471 | 96.0139 | 1.0205x |
| 8 | 173.7094 | 178.3984 | 1.0259x |

All 32-token timed outputs are identical between FP16 and W8 and within each
batch.

## Quality gate

Five production prompts cover a one-token prompt, English, Chinese, Python and
instruction text. All five produce the same eight greedy tokens under FP16 and
W8. Across 48 teacher-forced decode comparisons:

- minimum logit cosine: `0.99994028` (floor `0.999`)
- maximum normalized RMSE: `0.01338704` (ceiling `0.05`)
- maximum KL divergence: `0.00294696`
- minimum top-20 overlap: `0.95`
- maximum production-corpus loss delta: `0.01193333` (ceiling `0.02`)

The artifact also retains a non-corpus synthetic stress diagnostic. Its one
greedy mismatch is the FP16 runner-up under W8, with reference rank 2 and an
FP16 top-1 margin of `0.02734375`; the global logit gates still include this
diagnostic. It is disclosed but is not counted as a natural-language corpus
generation or loss row.

## Reproduce

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
PYTHONPATH=rwkv7-hf-ascend python \
  rwkv7-hf-ascend/bench/run_w8_graph_e2e.py \
  --model /path/to/rwkv7-7.2b \
  --new-tokens 32 \
  --corpus-new-tokens 8 \
  --quality-steps 8 \
  --rounds 5 \
  --output /tmp/hf_w8_graph_e2e.json
```

Other cards, software stacks, dtypes, projection shapes and logical batch rows
remain fail-closed. W4, vLLM and SGLang admission are separate work.
