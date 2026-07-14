# Qwen3.5 Dense Full-Matrix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the RWKV-7 Ascend backend measurably beat Qwen3.5 Dense 0.8B, 2B, 4B, 9B, and 27B across the paired full-size inference matrix without hiding failed or unrun rows.

**Architecture:** Keep each engine's native model loader, but emit one normalized result schema and evaluate it against a machine-readable five-tier manifest. Pair RWKV-7 0.4B/1.5B/2.9B/7.2B/13.3B with Qwen3.5 0.8B/2B/4B/9B/27B. Separate kernel correctness, engine performance, memory, and model quality so a speed win cannot satisfy a quality gate.

**Tech Stack:** Python 3.11, PyTorch/torch_npu, AscendC/CANN, vLLM-Ascend, JSON, pytest.

---

### Task 1: Add the five-tier manifest and schema validation

**Files:**
- Create: `vllm-rwkv-ascend/perf/qwen35_dense_matrix.json`
- Create: `vllm-rwkv-ascend/perf/model_matrix.py`
- Create: `vllm-rwkv-ascend/tests/test_model_matrix.py`

**Step 1: Write the failing manifest test**

Assert that the manifest contains exactly these pairs:

```python
EXPECTED = {
    "dense-0": ("rwkv7-0.4b", "qwen3.5-0.8b"),
    "dense-1": ("rwkv7-1.5b", "qwen3.5-2b"),
    "dense-2": ("rwkv7-2.9b", "qwen3.5-4b"),
    "dense-3": ("rwkv7-7.2b", "qwen3.5-9b"),
    "dense-4": ("rwkv7-13.3b", "qwen3.5-27b"),
}
```

Reject duplicate tier IDs, non-positive parameter counts, missing official model IDs, and a workload matrix that omits B1 or B4.

**Step 2: Run the focused test and verify failure**

Run: `cd vllm-rwkv-ascend && PYTHONPATH=. python -m pytest tests/test_model_matrix.py -q`

Expected: FAIL because `perf.model_matrix` does not exist.

**Step 3: Implement the minimal loader**

Provide `load_manifest(path)`, immutable tier/workload dataclasses, and validation errors that name the invalid field. The JSON manifest must record official source URLs, expected layers/hidden size when known, precision lanes, and minimum device count.

**Step 4: Run the focused test**

Expected: all manifest tests pass.

### Task 2: Add normalized result ingestion and strict gates

**Files:**
- Modify: `vllm-rwkv-ascend/perf/model_matrix.py`
- Create: `vllm-rwkv-ascend/perf/analyze_qwen35_matrix.py`
- Modify: `vllm-rwkv-ascend/tests/test_model_matrix.py`

**Step 1: Write failing analyzer tests**

Cover a passing tier, a missing row, a correctness failure, RWKV slower prefill, RWKV slower decode, higher peak memory, mismatched hardware/dtype, and an explicitly blocked multi-card row. A missing or blocked row must never count as pass.

**Step 2: Implement normalization**

Normalize the existing `rwkv7_pth_prefill_npu`, `qwen35_vllm_ascend`, and `qwen35_transformers_npu` JSON formats into:

```python
NormalizedRow(
    engine, model_id, tier_id, device_name, device_count, dtype,
    batch_size, prompt_length, decode_length,
    prefill_tokens_per_second, decode_tokens_per_second,
    peak_memory_mib, correctness_passed, source_path,
)
```

**Step 3: Implement strict tier status**

For every B1/B4 and prompt512/2048 row, require RWKV prefill and decode throughput to be greater than Qwen, RWKV peak memory no greater than Qwen, and RWKV correctness to pass. Report `pass`, `fail`, `missing`, or `blocked` per metric and tier. The global result passes only when all five tiers pass.

**Step 4: Verify JSON and Markdown output**

Run the analyzer on synthetic fixtures and assert stable machine-readable JSON plus a concise Markdown table.

### Task 3: Emit enough metadata from every benchmark

**Files:**
- Modify: `vllm-rwkv-ascend/perf/bench_rwkv7_pth_prefill.py`
- Modify: `vllm-rwkv-ascend/perf/bench_qwen35_npu.py`
- Modify: `vllm-rwkv-ascend/perf/bench_qwen35_vllm_ascend.py`
- Modify: `vllm-rwkv-ascend/tests/test_model_matrix.py`

**Step 1: Add failing metadata tests**

Require model ID, model byte size, device name/count, CANN, PyTorch, torch_npu, engine version, dtype, revision, batch/prompt/decode lengths, and peak memory phase.

**Step 2: Add shared metadata collection**

Do not import torch_npu in CPU-only analyzer tests. Benchmark processes collect accelerator metadata and write it into their own JSON results.

**Step 3: Preserve backward compatibility**

The analyzer must ingest the existing 0.4B/0.8B result files, marking unavailable fields as informational rather than inventing values.

### Task 4: Make the RWKV PTH loader viable at 13.3B

**Files:**
- Modify: `vllm-rwkv-ascend/perf/rwkv7_pth_engine.py`
- Modify: `vllm-rwkv-ascend/perf/bench_rwkv7_pth_prefill.py`
- Modify: `vllm-rwkv-ascend/tests/test_pth_engine.py`

**Step 1: Add a failing prefill-only packing test**

Show that prefill-only mode does not retain both independent and packed copies of RKV/low-rank matrices while preserving the tensors needed by layer-major prefill.

**Step 2: Implement explicit loader profiles**

Use `decode_full` for token-major decode correctness and `prefill_only` for large checkpoints. Never silently downgrade; record the profile in result JSON. In prefill-only mode, compare fused scan against the layer-major torch recurrence.

**Step 3: Add load-time memory telemetry**

Record checkpoint bytes, packed tensor bytes, NPU allocated memory after load, and peak memory after the workload.

### Task 5: Download and run the single-card tiers

**Files:**
- Create on benchmark host: `/data/models/...`
- Append result JSON under: `/data/results/qwen35_dense_matrix/`
- Modify: `BENCHMARK_QWEN35_ALBATROSS.md`

**Step 1:** Pin official revisions and checksums for RWKV 0.4B/1.5B/2.9B/7.2B/13.3B and Qwen3.5 0.8B/2B/4B/9B/27B.

**Step 2:** Run load/correctness smoke before performance for each downloaded checkpoint.

**Step 3:** Run B1/B4 prompt512 first, then prompt2048, with prefill and decode measured separately.

**Step 4:** Store OOM as an explicit failed row with requested and available memory; do not substitute quantization into an fp16 row.

### Task 6: Add multi-card lanes for models that do not fit

**Files:**
- Modify: `vllm-rwkv-ascend/perf/qwen35_dense_matrix.json`
- Create: `vllm-rwkv-ascend/perf/run_qwen35_dense_matrix.py`
- Modify: `BENCHMARK_QWEN35_ALBATROSS.md`

**Step 1:** Detect visible NPU count and refuse a TP/PP row when insufficient devices are present.

**Step 2:** Run Qwen through official vLLM-Ascend tensor parallelism and RWKV through the supported PP/replica throughput lane. Label latency and aggregate-throughput results separately.

**Step 3:** Require the same device count in a paired row.

### Task 7: Optimize failing rows and close the matrix

**Files:**
- Modify only the kernel/backend files identified by profiler evidence.
- Update: `BENCHMARK_QWEN35_ALBATROSS.md`

**Step 1:** Profile recurrence, projections, FFN, output head, and launch overhead per tier.

**Step 2:** For B4 prefill, implement compact-WY/chunked scan through a formal op-host tiling/workspace path; do not reuse the deadlocking direct Cube probe.

**Step 3:** For decode, tune card- and size-specific projection/layout policies and verify end-to-end value, not microbenchmarks alone.

**Step 4:** Re-run all previously passing rows after every default change.

**Step 5:** Declare full-size success only when the analyzer returns global `pass` with no missing or blocked rows; declare model-quality success separately after matching-stage evaluation rows pass.
