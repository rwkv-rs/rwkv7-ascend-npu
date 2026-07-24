# Ascend 910B3 HF NPUGraph evidence

This directory records the focused acceptance run for fixed-batch NPUGraph
decode through the canonical Hugging Face `generate()` entry point.

## Exact row

- device: 1x Ascend 910B3, 64 GiB HBM
- CANN: 8.5.0
- PyTorch / torch_npu: 2.9.0+cpu / 2.9.0
- Transformers: 4.57.6
- model: RWKV-7 7.2B `fla-hub/rwkv7-7.2B-g0a`
- dtype: FP16
- prompt: token ID `33155` (`Hello`)
- decode: greedy, 32 new tokens per request
- measured fixed batches: B1, B4 and B8

## Result

All benchmark gates pass. Capture warm-up is excluded from the measured rows.

| Batch | Output tok/s | Scaling over B1 | Peak allocated HBM |
|---:|---:|---:|---:|
| 1 | 29.8664 | 1.0000x | 13.71 GiB |
| 4 | 70.3739 | 2.3563x | 13.99 GiB |
| 8 | 141.7044 | 4.7446x | 14.51 GiB |

All rows produced the same expected first three IDs `[45, 308, 459]`.
The runtime captured exactly the fixed batch sizes `[1, 4, 8]`, reported 121
cache hits from 124 decode requests, and avoided redundant recurrent-state
copies on 120 of 124 requests.

## Reproduce

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
PYTHONPATH=rwkv7-hf-ascend python \
  rwkv7-hf-ascend/bench/run_e2e_performance.py \
  --model /path/to/rwkv7-7.2b \
  --backend native_graph \
  --dtype float16 \
  --new-tokens 32 \
  --output /tmp/hf_native_graph_e2e.json
```

This is a single-device, one-checkpoint, short-decode validation. It does not
claim dynamic-shape capture, graph-captured prefill, quantized promotion,
multi-NPU execution, or long-running serving stability.
