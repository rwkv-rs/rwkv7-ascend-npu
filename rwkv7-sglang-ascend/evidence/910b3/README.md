# Atlas 910B3 real-engine acceptance

- Hardware: Ascend 910B3 64 GiB
- CANN: 8.5.0
- torch / torch_npu: 2.9.0 / 2.9.0
- SGLang: `d0b9689805232d8ab37789121cbc3b766b5c723e`
- Model: `/data/models/fla-hub-rwkv7-7.2B-g0a`
- Command: `bash scripts/run_engine_acceptance.sh MODEL /data/work/sglang-rwkv-acceptance.json`

The acceptance exited 0 with `passed=true`. The backend trace proves a real
`MIXED` forward at `real_batch_size=2`, long-prompt state continuation over
64-token chunks, and physical state slot 4 release/reuse with radix caching
disabled. The shared `Hello` oracle used input IDs `[33155]` and generated
`[45, 308, 459]`, exactly matching the vLLM/HF dense result.

`sglang-rwkv-acceptance.sha256` contains hashes of the original immutable files
under `/data/work`; filenames in that manifest intentionally retain their
original `sglang-rwkv-acceptance.*` names.
