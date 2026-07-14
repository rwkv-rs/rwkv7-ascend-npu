"""Benchmark real-checkpoint RWKV-7 prefill layouts on Ascend.

The token-major reference invokes the complete decode graph once per prompt
token.  The layer-major candidate amortizes every projection over the full
prompt and leaves only the recurrent state scan sequential.  This makes the
remaining AscendC fusion target explicit and measurable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 - registers torch.npu
from torch.utils.cpp_extension import load

from rwkv7_pth_engine import build_blinkdl_engine


def _new_state(eng, device: str, batch_size: int = 1):
    state = torch.zeros(
        eng.L, batch_size, eng.H, eng.N, eng.N,
        dtype=torch.float32,
        device=device,
    )
    x_previous = torch.zeros(
        eng.L, batch_size, eng.hidden, dtype=torch.float16, device=device
    )
    ffn_previous = torch.zeros_like(x_previous)
    value_first = torch.zeros(
        batch_size, eng.hidden, dtype=torch.float16, device=device
    )
    return state, x_previous, ffn_previous, value_first


def token_major_prefill(eng, token_ids: torch.Tensor, device: str):
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    cache = _new_state(eng, device, token_ids.shape[0])
    logits = None
    with torch.no_grad():
        for token in token_ids.transpose(0, 1):
            embedding = eng.base.embeddings(token)
            logits = eng.mod.rwkv7_decode_full(
                embedding,
                *eng.W,
                *cache,
                eng.H,
                eng.N,
                eng.lm_w_m,
                eng.fnorm_w,
                eng.fnorm_b,
            )
    return logits, cache


def _cann_lib_dir(cann_home: str) -> str:
    for machine in ("aarch64-linux", "x86_64-linux"):
        candidate = os.path.join(cann_home, machine, "lib64")
        if os.path.exists(os.path.join(candidate, "libascendcl.so")):
            return candidate
    raise FileNotFoundError("cannot find CANN runtime lib64 under " + cann_home)


def load_ascendc_prefill_scan(cpp_source: str):
    direct_build = os.environ.get("RWKV7_ASCENDC_DIRECT_BUILD_DIR")
    if not direct_build:
        raise RuntimeError("RWKV7_ASCENDC_DIRECT_BUILD_DIR is required")
    torch_npu_root = os.path.dirname(torch_npu.__file__)
    cann_home = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    cann_lib = _cann_lib_dir(cann_home)
    return load(
        name="rwkv7_ascend_prefill_scan",
        sources=[cpp_source],
        verbose=False,
        extra_cflags=["-O3", "-std=c++17", "-DRWKV7_USE_PREFILL_SCAN=1"],
        extra_include_paths=[
            os.path.join(torch_npu_root, "include"),
            os.path.join(torch_npu_root, "include", "third_party", "acl", "inc"),
            os.path.join(direct_build, "include", "rwkv7_prefill_scan_kernel"),
        ],
        extra_ldflags=[
            os.path.join(direct_build, "lib", "librwkv7_prefill_scan_kernel.a"),
            "-L" + os.path.join(torch_npu_root, "lib"),
            "-Wl,-rpath," + os.path.join(torch_npu_root, "lib"),
            "-L" + cann_lib,
            "-Wl,-rpath," + cann_lib,
            "-ltorch_npu",
            "-lascendcl",
            "-lregister",
            "-lplatform",
            "-lascendalog",
            "-ldl",
        ],
    )


def layer_major_prefill(
    eng, token_ids: torch.Tensor, device: str, scan_module=None
):
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    batch_size, tokens = token_ids.shape
    state_all, xpa_all, xpf_all, value_first_cache = _new_state(
        eng, device, batch_size
    )
    x = eng.base.embeddings(token_ids)
    value_first = None
    hidden = eng.hidden
    heads = eng.H
    head_size = eng.N

    with torch.no_grad():
        for layer in range(eng.L):
            residual = (
                F.layer_norm(
                    x,
                    (hidden,),
                    eng.W[34][0],
                    eng.W[35][0],
                )
                if layer == 0
                else x
            )
            h = F.layer_norm(
                residual,
                (hidden,),
                eng.W[30][layer],
                eng.W[31][layer],
            )
            previous = torch.cat((xpa_all[layer].unsqueeze(1), h[:, :-1]), dim=1)
            delta = previous - h
            mixed = torch.stack(
                [
                    h + delta * eng.W[index][layer]
                    for index in (18, 20, 21, 19, 22, 23)
                ]
            )
            xr, xk, xv, xw, xa, xg = mixed
            xpa_all[layer].copy_(h[:, -1])

            r = F.linear(xr, eng.W[0][layer])
            k = F.linear(xk, eng.W[1][layer])
            v = F.linear(xv, eng.W[2][layer])
            lowrank_input = torch.stack((xw, xa, xg, xv)).reshape(
                4, batch_size * tokens, hidden
            )
            lowrank_hidden = torch.bmm(
                lowrank_input, eng.W[36][layer]
            )
            lowrank_hidden[0] = torch.tanh(lowrank_hidden[0])
            lowrank_hidden[2] = torch.sigmoid(lowrank_hidden[2])
            lowrank = (
                torch.bmm(lowrank_hidden, eng.W[37][layer])
                + eng.W[38][layer]
            ).view(4, batch_size, tokens, hidden)
            w_raw, a_raw, g, value_mix_raw = lowrank
            w = torch.exp(-0.6065306597126334 * torch.sigmoid(w_raw))
            a = torch.sigmoid(a_raw)
            kk = (k * eng.W[24][layer]).view(
                batch_size, tokens, heads, head_size
            )
            kk = kk / kk.norm(2, dim=-1, keepdim=True).clamp_min(1.0e-8)
            k = k * (1.0 + (a - 1.0) * eng.W[25][layer])

            if layer == 0:
                value_first = v.clone()
                value_first_cache.copy_(value_first[:, -1])
            else:
                v = v + (value_first - v) * torch.sigmoid(value_mix_raw)

            current_state = state_all[layer]
            if scan_module is not None:
                recurrent, _ = scan_module.rwkv7_prefill_scan(
                    current_state,
                    w.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    kk.view(batch_size, tokens, hidden).contiguous(),
                    a.contiguous(),
                    r.contiguous(),
                    heads,
                    head_size,
                )
            else:
                recurrent_rows = []
                for token in range(tokens):
                    w_token = w[:, token].view(
                        batch_size, heads, 1, head_size
                    ).float()
                    k_token = k[:, token].view(batch_size, heads, head_size)
                    v_token = v[:, token].view(batch_size, heads, head_size)
                    kk_token = kk[:, token]
                    a_token = a[:, token].view(batch_size, heads, head_size)
                    state_projection = torch.matmul(
                        current_state,
                        (-kk_token).unsqueeze(-1).float(),
                    )
                    current_state = (
                        current_state * w_token
                        + state_projection
                        * (kk_token * a_token).unsqueeze(-2).float()
                        + v_token.unsqueeze(-1).float()
                        * k_token.unsqueeze(-2).float()
                    )
                    recurrent_rows.append(
                        torch.matmul(
                            current_state.to(torch.float16),
                            r[:, token].view(
                                batch_size, heads, head_size, 1
                            ),
                        ).view(batch_size, hidden)
                    )
                state_all[layer].copy_(current_state)
                recurrent = torch.stack(recurrent_rows, dim=1)

            out = F.group_norm(
                recurrent.view(batch_size * tokens, hidden),
                heads,
                eng.W[27][layer],
                eng.W[28][layer],
                eps=head_size * 1.0e-5,
            )
            sk = (
                r.view(batch_size, tokens, heads, head_size)
                * k.view(batch_size, tokens, heads, head_size)
                * eng.W[26][layer].view(1, 1, heads, head_size)
            ).sum(dim=-1, keepdim=True)
            out = (
                out.view(batch_size, tokens, hidden)
                + (
                    sk * v.view(batch_size, tokens, heads, head_size)
                ).view(batch_size, tokens, hidden)
            ) * g
            x = residual + F.linear(out, eng.W[4][layer])

            h2 = F.layer_norm(
                x,
                (hidden,),
                eng.W[32][layer],
                eng.W[33][layer],
            )
            ffn_previous = torch.cat(
                (xpf_all[layer].unsqueeze(1), h2[:, :-1]), dim=1
            )
            k_ffn = h2 + (ffn_previous - h2) * eng.W[29][layer]
            xpf_all[layer].copy_(h2[:, -1])
            ffn_hidden = F.linear(k_ffn, eng.W[5][layer])
            ffn_out = F.linear(F.relu(ffn_hidden).square(), eng.W[6][layer])
            x = x + ffn_out

        last = F.layer_norm(
            x[:, -1],
            (hidden,),
            eng.fnorm_w,
            eng.fnorm_b,
        )
        logits = F.linear(last, eng.lm_w_m)
    return logits, (state_all, xpa_all, xpf_all, value_first_cache)


def _timed(function, *args):
    torch.npu.synchronize()
    started = time.perf_counter()
    output = function(*args)
    torch.npu.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--correctness-length", type=int, default=8)
    parser.add_argument(
        "--cpp-source",
        default=os.path.join(os.path.dirname(__file__), "rwkv7_ascend_v3.cpp"),
    )
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--ascendc-scan",
        action="store_true",
        help="replace the sequential torch recurrence with the fused AscendC scan",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("batch-size/iterations must be positive and warmup non-negative")

    eng = build_blinkdl_engine(
        args.cpp_source,
        model_path=args.model_pth,
        device=args.device,
        include_mix_project=False,
    )
    scan_module = (
        load_ascendc_prefill_scan(args.cpp_source) if args.ascendc_scan else None
    )
    correctness_ids = (
        torch.arange(
            args.correctness_length, device=args.device, dtype=torch.long
        ).unsqueeze(0)
        + torch.arange(
            args.batch_size, device=args.device, dtype=torch.long
        ).unsqueeze(1)
        * 997
    ).remainder(eng.vocab_size)
    if args.batch_size == 1:
        correctness_reference = "token_major_decode"
        reference_logits, reference_cache = token_major_prefill(
            eng, correctness_ids, args.device
        )
    else:
        # The optimized full-decode extension is intentionally B=1.  For
        # batched scan validation, compare against the identical layer-major
        # projection path with the recurrence executed in pure PyTorch.
        correctness_reference = "layer_major_torch_scan"
        reference_logits, reference_cache = layer_major_prefill(
            eng, correctness_ids, args.device, None
        )
    candidate_logits, candidate_cache = layer_major_prefill(
        eng, correctness_ids, args.device, scan_module
    )
    torch.npu.synchronize()
    logits_cosine = F.cosine_similarity(
        reference_logits.float(), candidate_logits.float()
    ).min().item()
    logits_max_abs = (
        reference_logits.float() - candidate_logits.float()
    ).abs().max().item()
    state_max_abs = max(
        (reference.float() - candidate.float()).abs().max().item()
        for reference, candidate in zip(reference_cache, candidate_cache)
    )
    greedy_match = bool(
        torch.equal(reference_logits.argmax(-1), candidate_logits.argmax(-1))
    )
    print(
        "correctness greedy=%s logits_cosine=%.9f logits_max_abs=%.6g "
        "state_max_abs=%.6g"
        % (str(greedy_match).lower(), logits_cosine, logits_max_abs, state_max_abs),
        flush=True,
    )

    prompt_ids = (
        torch.arange(
            args.prompt_length, device=args.device, dtype=torch.long
        ).unsqueeze(0)
        + torch.arange(
            args.batch_size, device=args.device, dtype=torch.long
        ).unsqueeze(1)
        * 997
    ).remainder(eng.vocab_size)
    for _ in range(args.warmup):
        layer_major_prefill(eng, prompt_ids, args.device, scan_module)
    layer_samples = [
        _timed(
            layer_major_prefill, eng, prompt_ids, args.device, scan_module
        )[1]
        for _ in range(args.iterations)
    ]
    layer_ms = statistics.median(layer_samples)
    result = {
        "benchmark": "rwkv7_pth_prefill_npu",
        "scan_backend": "ascendc" if args.ascendc_scan else "torch",
        "model": os.path.abspath(args.model_pth),
        "shape": {
            "layers": eng.L,
            "heads": eng.H,
            "head_size": eng.N,
            "hidden": eng.hidden,
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_length,
        },
        "correctness": {
            "reference": correctness_reference,
            "batch_size": args.batch_size,
            "length": args.correctness_length,
            "greedy_match": greedy_match,
            "logits_cosine": logits_cosine,
            "logits_max_abs": logits_max_abs,
            "state_max_abs": state_max_abs,
        },
        "layer_major_latency_ms": layer_ms,
        "layer_major_latency_ms_samples": layer_samples,
        "layer_major_tokens_per_second": (
            args.batch_size * args.prompt_length * 1000.0 / layer_ms
        ),
        "peak_memory_mib": torch.npu.max_memory_allocated(args.device) / 2**20,
    }
    print(json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
