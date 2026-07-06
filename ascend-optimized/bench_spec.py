"""Spec decode speedup bench: baseline generate vs rwkv7_speculative_generate.
Real weights, V100. Reports acceptance + tok/s + speedup."""
import os, sys, time, argparse
os.environ.setdefault("RWKV_V7_ON", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(path, dtype, device):
    m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype, attn_mode="fused_recurrent", fuse_norm=False)
    return m.to(device).eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--draft-tokens", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    target = load(args.model, dtype, args.device)
    draft = load(args.draft_model, dtype, args.device)
    ids = tok("User: Hello!\n\nAssistant:", return_tensors="pt").input_ids.to(args.device)
    pad = getattr(tok, "pad_token_id", None) or 0
    N = args.max_new_tokens

    def baseline():
        with torch.inference_mode():
            return target.generate(ids, max_new_tokens=N, do_sample=False, use_cache=True, pad_token_id=pad)
    def spec():
        with torch.inference_mode():
            return target.rwkv7_speculative_generate(ids, draft_model=draft, max_new_tokens=N, draft_tokens=args.draft_tokens, return_stats=True)

    # warmup + correctness
    with torch.inference_mode():
        b = baseline()
        s = spec()
    assert torch.equal(s["sequences"], b), "spec != baseline"
    acc = s["stats"]["acceptance_rate"]

    # time baseline
    for _ in range(2): baseline()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): baseline()
    torch.cuda.synchronize(); b_ms = (time.time()-t0)/5*1000

    # time spec
    for _ in range(2): spec()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): spec()
    torch.cuda.synchronize(); s_ms = (time.time()-t0)/5*1000

    print("target=%s draft=%s draft_tokens=%d N=%d" % (os.path.basename(args.model), os.path.basename(args.draft_model), args.draft_tokens, N), flush=True)
    print("acceptance_rate=%.3f" % acc, flush=True)
    print("baseline: %.1f ms = %.0f tok/s" % (b_ms, 1000/b_ms*N), flush=True)
    print("spec:     %.1f ms = %.0f tok/s (%.2fx)" % (s_ms, 1000/s_ms*N, b_ms/s_ms), flush=True)

if __name__ == "__main__":
    main()
