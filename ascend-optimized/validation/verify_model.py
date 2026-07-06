"""Verify an HF RWKV-7 model on Ascend NPU against a V100 CUDA reference.

Loads the model via the fla-free native forward (NativeRWKV7ForCausalLM) on npu:0,
forwards the fixed 16-token prompt [0..15], and compares logits to a V100-generated
reference (`refs/v100_<model>_logits.pt`) if present -> cos / argmax / verdict.
Writes a per-model JSON result.

Usage:
  PYTHONPATH=<adapter-root> python validation/verify_model.py <hf-model-dir> [--name <model-name>]

The HF model dir needs only config.json + model.safetensors (the adapter code is
taken from PYTHONPATH, not trust_remote_code).
"""
import os
import sys
import json
import argparse

os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
import torch_npu  # noqa
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

DEV = "npu:0"
PROMPT = list(range(16))
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--name", default=None, help="model name for output files")
    ap.add_argument("--ref-dir", default=os.path.join(HERE, "refs"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    name = args.name or os.path.basename(args.model_dir.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ref_dir, exist_ok=True)

    print(f"[verify] loading {args.model_dir} on {DEV}", flush=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16).to(DEV).eval()
    ids = torch.tensor([PROMPT], device=DEV)
    with torch.no_grad():
        npu_logits = model(ids).logits[0].float().cpu()
    torch.save(npu_logits, os.path.join(args.ref_dir, f"npu_{name}_logits.pt"))

    # greedy continuation
    seq = list(PROMPT)
    gen = []
    for _ in range(8):
        with torch.no_grad():
            lo = model(torch.tensor([seq], device=DEV))
        nx = int(lo.logits[0, -1].argmax())
        gen.append(nx)
        seq.append(nx)

    result = {"model": name, "greedy_next8": gen, "npu_logits_shape": list(npu_logits.shape)}
    ref_path = os.path.join(args.ref_dir, f"v100_{name}_logits.pt")
    if os.path.exists(ref_path):
        v100 = torch.load(ref_path, map_location="cpu").float()
        cos = torch.nn.functional.cosine_similarity(
            npu_logits.flatten(), v100.flatten(), dim=0).item()
        am = (npu_logits.argmax(-1) == v100.argmax(-1)).float().mean().item()
        maxabs = (npu_logits - v100).abs().max().item()
        result.update({"cos_vs_v100": round(cos, 5), "argmax_match_rate": round(am, 4),
                       "max_abs": round(maxabs, 4),
                       "verdict": "PASS" if cos > 0.99 else "CHECK"})
    else:
        result["verdict"] = "NEEDS_V100_REF"
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("VERIFY_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
