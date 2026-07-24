#!/usr/bin/env python3
"""Child process used by acceptance_engine.py to isolate the real Engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    args = ap.parse_args()

    from sglang_rwkv7_ascend import register

    register()
    from sglang.srt.entrypoints.engine import Engine

    source_model = Path(args.model).resolve()
    # Keep the checkpoint's required custom tokenizer while forcing config/model
    # resolution through the installed plugin.  The source config's AutoConfig
    # auto_map points at a legacy class that lacks current SGLang attributes.
    model_view = Path(__file__).resolve().parents[1] / ".acceptance-model-view"
    shutil.rmtree(model_view, ignore_errors=True)
    model_view.mkdir(parents=True)
    source_config = json.loads((source_model / "config.json").read_text())
    source_config.pop("auto_map", None)
    (model_view / "config.json").write_text(
        json.dumps(source_config, indent=2) + "\n", encoding="utf-8"
    )
    for item in source_model.iterdir():
        if item.name != "config.json":
            (model_view / item.name).symlink_to(item)

    engine = Engine(
        model_path=str(model_view),
        device="npu",
        dtype="bfloat16",
        # Use the registered plugin config/model, not the checkpoint's legacy
        # remote-code class (which lacks current SGLang config attributes).
        trust_remote_code=True,
        attention_backend="ascend",
        max_running_requests=2,
        # Keep enough state slots for scheduler padding + two live requests +
        # explicit release/reuse probes without reducing max-running to one.
        max_mamba_cache_size=8,
        # Acceptance requires fresh request state; prefix/radix reuse would make
        # deterministic slot-clear evidence ambiguous.
        disable_radix_cache=True,
        chunked_prefill_size=args.chunk_size,
        enable_mixed_chunk=True,
        page_size=1,
        disable_cuda_graph=True,
        log_level="info",
    )
    try:
        long_prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "Explain the next sentence carefully and concisely. "
        ) * 24
        short_prompt = "Explain why the sky is blue in one short sentence."
        sampling = {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
        }
        started = time.monotonic()
        batch = engine.generate([long_prompt, short_prompt], sampling)
        batch_elapsed = time.monotonic() - started
        trace_file = Path(os.environ["RWKV_SGLANG_ACCEPTANCE_TRACE"])
        with trace_file.open("a", encoding="utf-8") as trace:
            trace.write('{"kind":"marker","name":"before_slot_reuse"}\n')
        repeated = engine.generate(short_prompt, sampling)
        # Walk every free state slot with cheap one-token requests so at least
        # one physical slot released by the first batch is allocated again.
        slot_probes = [
            engine.generate(
                short_prompt,
                {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                        },
            )
            for _ in range(9)
        ]
        tokenizer = engine.tokenizer_manager.tokenizer
        oracle_input_ids = tokenizer.encode("Hello", add_special_tokens=False)
        oracle_response = engine.generate(
            input_ids=oracle_input_ids,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 3,
                "ignore_eos": True,
                },
            return_logprob=True,
            logprob_start_len=0,
        )
        token_logprobs = oracle_response.get("meta_info", {}).get(
            "output_token_logprobs", []
        )
        oracle_output_ids = []
        for item in token_logprobs:
            if isinstance(item, dict):
                oracle_output_ids.append(int(item["token_id"]))
            else:
                oracle_output_ids.append(int(item[1]))
        Path(args.output).write_text(
            json.dumps(
                {
                    "batch": batch,
                    "repeated": repeated,
                    "slot_probe_count": len(slot_probes),
                    "batch_elapsed": batch_elapsed,
                    "engine_config": {
                        "disable_radix_cache": True,
                        "enable_mixed_chunk": True,
                        "chunked_prefill_size": args.chunk_size,
                        "max_running_requests": 2,
                        "max_mamba_cache_size": 8,
                    },
                    "oracle": {
                        "prompt": "Hello",
                        "input_ids": oracle_input_ids,
                        "output_ids": oracle_output_ids,
                        "text": oracle_response.get("text"),
                    },
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
