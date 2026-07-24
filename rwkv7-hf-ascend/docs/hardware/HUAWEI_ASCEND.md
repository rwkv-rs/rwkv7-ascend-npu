# Huawei Ascend / torch_npu

The canonical FLA-free `NativeRWKV7ForCausalLM` backend runs through ordinary
PyTorch operators registered by `torch_npu`. Ascend support is optional: importing
`rwkv7_hf` on a CPU/CUDA/MPS host does not import or require `torch_npu`.

## Validated compatibility row (2026-07-24)

| Item | Exact value |
|---|---|
| Device | 1x Ascend 910B3, 64 GiB HBM |
| CANN | 8.5.0 |
| Python | 3.11.14, aarch64 |
| PyTorch / torch_npu | 2.9.0+cpu / 2.9.0 |
| Transformers | 4.57.6 |
| Dtype | BF16 |
| Model | random 2-layer tiny fixture (not an official checkpoint) |
| Backends | conservative eager and packed pure-torch `native_jit` |

Both backends pass standard `AutoConfig`, `AutoTokenizer`, `AutoModel`,
`AutoModelForCausalLM`, forward, `generate(use_cache=True)`, safe save/reload, a tiny labels/loss/backward/parameter-update smoke,
`NativeRWKV7Cache.select_batch`, recurrent continuation, and chunked-prefill
parity on the exact row above. Raw evidence is in
`bench/ascend_910b3_20260724/`.

A real 7.2B safetensors checkpoint also passes canonical native HF BF16 load,
4-token forward, chunked-prefill parity (`cosine=1.0`), recurrent continuation
and two-token generation. Resident/peak NPU allocation was 14.71/14.80 GB.
The source config declared `num_heads=32` while `r_k` and projections prove 64;
`scripts/create_native_hf_view.py` performs and records that shape-inferred
metadata repair without changing source weights.

The tiny row remains a compatibility smoke rather than a quality or throughput
claim. A separate independent real-checkpoint gate is described below. Neither
row establishes training convergence, long-running stability or production
serving throughput. Fixed-batch graph decode has its own real-checkpoint row
below and does not change the scope of these earlier gates.

## Independent 7.2B CPU oracle and NPU alignment

`rwkv7_hf.ascend_reference_oracle` is deliberately independent from the
candidate backend: it imports neither `native.py` nor `native_model.py`, never
calls a Hugging Face model `forward`, and reads official safetensors directly.
It transcribes the pure-PyTorch layer formula and FP32 naive recurrence from
`fla-org/flash-linear-attention` commit
`d1ce07369d581813553f30a750af3b6b5f9af6a9`. The capture fails closed unless
that commit's three relevant source files, the checkpoint index/config, all
three weight shards and the tokenizer files have their pinned SHA256 values.

The common cross-backend gate is prompt `Hello`, input token ID `[33155]`,
temperature 0, `max_new_tokens=3`, and EOS ignored. CPU BF16 oracle, canonical
HF/Ascend and vLLM all greedily produce `[45, 308, 459]`. The complete HF NPU
comparison includes prefill plus all three decode logits and both prefill/final
recurrent state captures (208 tensors):

| Gate | Measured | Threshold | Result |
|---|---:|---:|---|
| minimum logit cosine | 0.99996638 | >= 0.999 | pass |
| maximum logit normalized RMSE | 0.00790427 | <= 0.02 | pass |
| minimum state cosine | 0.99984443 | >= 0.999 | pass |
| maximum state normalized RMSE | 0.01771485 | <= 0.02 | pass |
| greedy IDs | exact | exact | pass |

Reference/candidate canonical capture hashes are respectively
`35f24f1e0116fcbee548c69ee38652b897f14e6a09dedfad79753a0380e324f9`
and `8b83cb5af96a250565c713a5dd9f9a3ad0dff7d123659f00fc59fb86d6fe4cbf`.
The fail-closed comparison is
`bench/ascend_910b3_20260724/compare_7p2_real_b1_bf16_npu_vs_oracle.json`.

A second real 7.2B B2 gate uses different valid lengths with left-padding. Each
row is compared with its compact B1 forward and incremental cache, then the B2
cache is continued and the full prefill is repeated with `chunk_size=1`.
All seven gates pass; global logits/state minimum cosine is
0.99999988/0.99999970 and maximum normalized RMSE is 0/0. Evidence is
`results_7p2_real_b2_ragged.json` with canonical evidence SHA256
`54a7ac69520349e99136c144e1f9a4fc850d813b2b8cd4668b8b99b0215f160e`.

Reproduce the independent and candidate captures as follows. The 66 MiB tensor
captures are intentionally ignored by Git; their canonical and file hashes are
committed in JSON.

```bash
git clone https://github.com/fla-org/flash-linear-attention.git /tmp/fla
git -C /tmp/fla checkout d1ce07369d581813553f30a750af3b6b5f9af6a9
python scripts/build_ascend_hf_reference.py \
  --model /path/to/fla-hub-rwkv7-7.2B-g0a --fla-checkout /tmp/fla \
  --output-json reference.json --output-tensors reference.safetensors
python scripts/capture_ascend_hf_candidate.py \
  --model /path/to/native-view --reference-json reference.json \
  --output-json candidate.json --output-tensors candidate.safetensors
python scripts/compare_ascend_hf_reference.py \
  --reference-json reference.json --reference-tensors reference.safetensors \
  --candidate-json candidate.json --candidate-tensors candidate.safetensors \
  --output comparison.json
```

## Environment

Install the PyTorch and `torch_npu` wheels that match the host CANN release from
the Huawei Ascend distribution, then install this adapter. `torch_npu` is not a
mandatory dependency because its wheel/index is CANN- and architecture-specific.

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null || true
python3.11 -m venv .venv
. .venv/bin/activate
# Install the CANN-matched torch + torch_npu pair first.
pip install -e '.[ascend]'
python -c 'from rwkv7_hf import ascend_available; print(ascend_available())'
```

The final command must print `True` before moving a model to NPU.

`enable_ascend` additionally requires the normalized device name to equal
`Ascend910B3` and CANN/torch/torch_npu versions to equal
`8.5.0`/`2.9.0+cpu`/`2.9.0`. It uses exact equality, not card-family or version
substring matching. Every other Huawei card or software stack raises before
`set_device`. An experimental audit may explicitly set
`RWKV7_ALLOW_UNVALIDATED_ASCEND=1`; returned runtime metadata then reports
`validated_stack=false` and `validation_status=unvalidated_override`, and must
not be presented as production evidence.

## Standard HF API

```python
import torch
from rwkv7_hf import enable_ascend
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

info = enable_ascend("npu:0", backend="eager")
model_dir = "/path/to/converted-rwkv7-hf"
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).eval().to(info.device)
inputs = tokenizer("User: Hello. Assistant:", return_tensors="pt").to(info.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=8, use_cache=True)
print(tokenizer.decode(output[0]))
```

`enable_ascend(..., backend="eager")` sets only fail-closed native flags and
preserves explicit environment overrides. After eager correctness passes on an
exact model/card/runtime row, `backend="native_jit"` enables the packed
pure-PyTorch decode path. `backend="native_graph"` explicitly selects the
fixed-batch torch-npu graph route described below. CUDA graph, Triton, FLA and
bitsandbytes kernels are not silently selected on NPU.

## Fixed-batch NPUGraph decode

The native HF model can capture a complete single-token decode step with
`torch.npu.NPUGraph`. Prefill remains on the ordinary native path. A runner owns
fixed-address input, logits and recurrent-state buffers for one batch size;
subsequent decode calls bind the HF cache to those graph-resident states and
skip redundant state copies. Cache selection/reordering remains supported when
the batch size stays fixed.

```python
info = enable_ascend("npu:0", backend="native_graph")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    dtype=torch.float16,
).eval().to(info.device)

# Optional: pay capture cost before the first request for these fixed batches.
model.rwkv7_warmup_fast_token((1, 4, 8), backend="native_graph")
print(model.rwkv7_native_graph_cache_stats())
```

`RWKV7_ASCEND_GRAPH_CACHE_SIZE` controls the per-model LRU and defaults to
three runners. Changing packed quantization buffers changes the graph cache key,
which prevents a dense capture from being reused after module replacement.
`model.rwkv7_clear_native_graph_cache()` explicitly detaches bound recurrent
states and releases all cached runners.

On the exact stack above, a real 7.2B FP16 `transformers.generate` gate decoded
32 tokens per request after capture:

| Fixed batch | Output tok/s | Scaling over B1 | Peak allocated HBM |
|---:|---:|---:|---:|
| 1 | 29.8664 | 1.0000x | 13.71 GiB |
| 4 | 70.3739 | 2.3563x | 13.99 GiB |
| 8 | 141.7044 | 4.7446x | 14.51 GiB |

All three rows produced the exact expected `Hello` prefix `[45, 308, 459]`,
reported `last_decode_backend=native_graph`, and passed the batch-scaling
gates. The cache contained exactly `[1, 4, 8]`, with 121 hits from 124 requests.
Graph-resident binding skipped 120 of 124 recurrent-state copies. The JSON,
log, hashes and command are committed in
`bench/ascend_910b3_graph_20260724/`.

This evidence covers fixed-batch decode only. A new batch size incurs capture
cost and consumes one LRU entry. Dynamic-shape capture, graph-captured prefill,
quantized graph promotion, multi-NPU execution and long-duration stability
remain separate acceptance gaps.

## Dynamic recurrent cache and chunked prefill

```python
with torch.inference_mode():
    prefill = model.rwkv7_prefill_chunks(
        inputs["input_ids"], chunk_size=256, logits_to_keep=1
    )
    cache = prefill.past_key_values
    active = cache.clone().select_batch(torch.tensor([0], device=info.device))
    step = model(
        input_ids=torch.tensor([[42]], device=info.device),
        past_key_values=active,
        use_cache=True,
    )
```

The native cache supports clone/detach, select/reorder/compact, repeat,
device/dtype moves for offload/restore, reset, metrics and HF generation
reordering. It is an RWKV recurrent-state cache, not a Transformer KV cache.
Cropping to an earlier positive prefix requires a fresh prefill.

## Reproduce the hardware smoke

```bash
PYTHON_BIN=.venv/bin/python DTYPE=bf16 BACKEND=eager \
  bash scripts/run_huawei_ascend_smoke.sh

PYTHON_BIN=.venv/bin/python DTYPE=bf16 BACKEND=native_jit \
  RESULTS=bench/ascend_910b3_$(date +%Y%m%d)/results_native_jit.jsonl \
  bash scripts/run_huawei_ascend_smoke.sh
```

Pass requires a JSON row with all of `forward`, `generate`, `dynamic_cache`,
`chunked_prefill`, `save_reload` equal to `pass`, plus the expected Auto* class
names. With no model argument the runner creates and deletes a tiny random HF
fixture. Pass a converted official model directory as the first argument for a
real-checkpoint smoke.

## Production HF W8 and candidate-only W4

`rwkv7_hf.ascend_quant` provides a physically packed W8A16 Linear using
`torch_npu.npu_weight_quant_batchmatmul`. Its production speed policy is
fail-closed to the exact stack Ascend 910B3 / CANN 8.5.0 /
torch 2.9.0+cpu / torch_npu 2.9.0 / FP16, these 7.2B FFN shapes, and
logical rows B1/B4/B8:

- `ffn.key`: 4096 -> 16384
- `ffn.value`: 16384 -> 4096

```python
import torch
from rwkv7_hf import enable_ascend, quantize_ascend_w8a16

enable_ascend("npu:0", backend="native_graph")
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=True, dtype=torch.float16
).eval().to("npu:0")
replaced = quantize_ascend_w8a16(model, policy="speed", strict=True)
print(replaced)
```

Each W8 replacement retains int8 `[K,N]` weight plus a 16-bit per-output
scale and no dense parameter. Conversion clears an existing graph cache, and
the graph key tracks W8 buffer identity so a dense capture cannot be replayed
after replacement. Runtime logical rows outside B1/B4/B8 raise instead of
silently inheriting the performance claim.

The real 7.2B production gate quantizes all 64 FFN key/value projections and
uses five alternating FP16/W8 `transformers.generate` pairs after capture:

| Batch | FP16 tok/s | W8 tok/s | Median paired W8/FP16 |
|---:|---:|---:|---:|
| 1 | 25.8451 | 26.4352 | 1.0241x |
| 4 | 94.2471 | 96.0139 | 1.0205x |
| 8 | 173.7094 | 178.3984 | 1.0259x |

Model tensor payload is 70.18% of FP16 and isolated active HBM is 71.48%.
All timed 32-token outputs match FP16 exactly. Five production prompts covering
English, Chinese, Python, instruction text and the one-token `Hello` prompt
also match for eight greedy tokens. Across the full 48-step comparison,
minimum logit cosine is 0.99994028, maximum normalized RMSE is 0.01338704,
minimum top-20 overlap is 0.95, and maximum production-corpus loss delta is
0.01193333.

A retained synthetic stress diagnostic has one near-tied greedy flip: the W8
choice is rank 2 under FP16 and the FP16 top-1 margin is 0.02734375. This
diagnostic remains included in the global logit thresholds but is not presented
as a natural-language corpus row. JSON, log and hashes are in
`bench/ascend_910b3_w8_graph_20260724/`.

Earlier eager/BF16 and W4 candidates did not pass the same production gate:

| Candidate | Model payload | Single-model HBM | Paired decode | Min logit cosine | Greedy |
|---|---:|---:|---:|---:|---|
| W8 BF16, 32x `ffn.value` | 0.85087x | 0.85216x | 0.99633x | 0.98798 | mismatch |
| W4 FP16 G128, 64x key+value | 0.56188x | 0.57489x | 0.97843x | 0.98583 | mismatch |
| W4 FP16 G128, 32x value only | 0.78094x | 0.78739x | 0.98828x | 0.98658 | mismatch |

The value-only W4 row used five alternating paired groups; individual speedups
were 0.9458x-1.0474x. Its KL divergence was max 2.9047 / mean 0.4002 and top-20
overlap min 0.40 / mean 0.85. It fails both the defined quality and speed gates.

Consequently W8 `policy="speed"` is enabled only for the exact accepted
FP16/B1/B4/B8 tuple. `memory` and `candidate` remain explicit routes without a
speed claim. In `ascend_quant_w4`, production
`should_quantize(...)` returns `False` for every tuple; the separate
`raw_candidate_supported(...)` function reports only exact-stack diagnostic
coverage and never authorizes serving. W4 is exposed only through
`quantize_ascend_w4a16_candidate(..., require_explicit_candidate=False)` and is
never automatic. W4 conversion rejects BF16 before mutating any layer because
the raw candidate is FP16-only. W4 raw-operator wins remain stored with
explicit candidate gates and never authorize whole-model serving. GPTQ/AWQ or
another calibrated quantizer plus fused/graph dispatch is required before W4
promotion.


### W4 CLE/AWQ calibration candidate (not validated on NPU)

For the next quality iteration, `rwkv7_hf.ascend_w4_cle` implements the exact
squared-ReLU channel transform `key[j] /= sqrt(c[j])` and
`value[:,j] *= c[j]`. `calibrate_sqrelu_value_w4` uses real intermediate
activation max/RMS plus value-column max and an alpha grid, always including the
identity baseline. The CPU unit test proves pre-quantization equivalence to
`1e-12`; no NPU or 7.2B improvement is claimed yet.

```bash
python scripts/calibrate_ascend_w4_cle.py   --key-weight layer_key.pt --value-weight layer_value.pt   --calibration-inputs layer_inputs.pt --group-size 128   --output candidate_scale.pt
```

Applying the returned scale and rerunning the same real-model quality/memory/
paired-speed gate is mandatory before any W4 promotion.

## Failure recovery

- `torch_npu could not be imported`: reinstall the exact torch/torch_npu pair
  for the active Python and CANN release; do not install a CUDA PyTorch wheel.
- `no Huawei Ascend NPU is available`: source CANN `set_env.sh`, check
  `ASCEND_RT_VISIBLE_DEVICES`, then run `npu-smi info`.
- BF16/FP16 operator dtype error: first rerun BF16 eager. Do not promote a dtype
  until the exact real checkpoint passes logits and greedy-token alignment.
- Non-finite random FP16 tiny logits are not a hardware verdict: random tiny
  initialization is numerically unlike official weights. Use BF16 for the tiny
  gate and validate FP16 separately on the target checkpoint.

## Unclosed production gates

As of this evidence row, the repository has not established on this host:

- official checkpoints other than this fixed 7.2B B1/B2 short-sequence gate;
- PEFT/Trainer/TRL and real-checkpoint training on NPU (only tiny native backward/update passed);
- dynamic-shape NPU Graph and graph-captured prefill (fixed-batch decode only);
- paired prefill/decode throughput or long-duration serving stability;
- W4 production admission, plus HF W8 beyond the exact 7.2B FP16 B1/B4/B8 row;
- multi-NPU HCCL/device-map execution.

Do not infer those outcomes from the tiny compatibility smoke.
