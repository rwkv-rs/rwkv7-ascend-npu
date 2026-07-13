"""Measure graph-external overhead for the production B=1 decode path.

This benchmark intentionally uses shape-correct synthetic weights so it can run
on a bare torch_npu image without downloading a checkpoint or installing
Transformers.  It separates the captured RWKV-7 forward from the embedding
update and scheduler-state copies performed by :class:`NpuGraphDecoder`.

Example (0.1B shape on a 910B3)::

    python perf/bench_graph_overhead.py --iterations 100
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import types

import torch
import torch_npu  # noqa: F401 - registers torch.npu and NPUGraph
from torch.utils.cpp_extension import load


HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.join(os.path.dirname(HERE), "serving")
sys.path.insert(0, SERVING)

from graph_decode import NpuGraphDecoder  # noqa: E402


def _matrix(rows: int, cols: int, device: str) -> torch.Tensor:
    # The values do not affect GEMM scheduling.  Small constants keep repeated
    # recurrent updates finite while avoiding a slow checkpoint dependency.
    return torch.full((rows, cols), 1e-3, dtype=torch.float16, device=device)


def _vector(size: int, device: str, value: float = 0.0) -> torch.Tensor:
    return torch.full((size,), value, dtype=torch.float16, device=device)


def build_synthetic_engine(
    cpp_source: str,
    *,
    device: str,
    layers: int,
    heads: int,
    head_size: int,
    vocab_size: int,
    extension_name: str = "rwkv7_ascend_graph_overhead",
    extra_cflags=None,
) -> types.SimpleNamespace:
    hidden = heads * head_size
    ffn = hidden * 4
    rank = min(64, hidden)
    matrix_cache = {}
    vector_cache = {}

    mod = load(
        name=extension_name,
        sources=[cpp_source],
        verbose=False,
        extra_cflags=extra_cflags or ["-O3", "-std=c++17"],
    )

    def matrices(rows: int, cols: int):
        key = (rows, cols)
        if key not in matrix_cache:
            matrix_cache[key] = _matrix(rows, cols, device)
        return [matrix_cache[key]] * layers

    def vectors(value: float = 0.0):
        if value not in vector_cache:
            vector_cache[value] = _vector(hidden, device, value)
        return [vector_cache[value]] * layers

    rw = matrices(hidden, hidden)
    kw = matrices(hidden, hidden)
    vw = matrices(hidden, hidden)
    ow = matrices(hidden, hidden)
    fkw = matrices(ffn, hidden)
    fvw = matrices(hidden, ffn)
    w0 = matrices(rank, hidden)
    w2 = matrices(hidden, rank)
    a0 = matrices(rank, hidden)
    a2 = matrices(hidden, rank)
    g0 = matrices(rank, hidden)
    g2 = matrices(hidden, rank)
    v0 = matrices(rank, hidden)
    v2 = matrices(hidden, rank)
    w2b, a2b, v2b = vectors(), vectors(), vectors()
    xr, xw, xk, xv, xa, xg = (vectors(0.5) for _ in range(6))
    kk, ka, rk = vectors(1.0), vectors(1.0), vectors(1.0)
    gnw, gnb, fxk = vectors(1.0), vectors(), vectors(0.5)
    anw, anb = vectors(1.0), vectors()
    fnw, fnb = vectors(1.0), vectors()
    pnw, pnb = vectors(1.0), vectors()

    eng = types.SimpleNamespace()
    eng.L, eng.H, eng.N, eng.hidden = layers, heads, head_size, hidden
    eng.mod = mod
    eng.W = (
        rw, kw, vw, ow, fkw, fvw,
        w0, w2, a0, a2, g0, g2, v0, v2,
        w2b, a2b, v2b,
        xr, xw, xk, xv, xa, xg,
        kk, ka, rk, gnw, gnb, fxk,
        anw, anb, fnw, fnb, pnw, pnb,
    )
    eng.base = types.SimpleNamespace(
        embeddings=torch.nn.Embedding(
            vocab_size, hidden, device=device, dtype=torch.float16
        )
    )
    eng.lm_w_m = _matrix(vocab_size, hidden, device)
    eng.fnorm_w = _vector(hidden, device, 1.0)
    eng.fnorm_b = _vector(hidden, device)
    return eng


def _randomize_synthetic_weights(eng, seed: int) -> None:
    """Use deterministic non-degenerate values for numerical A/B checks."""
    torch.manual_seed(seed)
    seen = set()
    with torch.no_grad():
        for index in list(range(14)) + [14, 15, 16]:
            for tensor in eng.W[index]:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                tensor.normal_(0, 0.02 if tensor.dim() > 1 else 0.01)
        for index in list(range(17, 23)) + [28]:
            for tensor in eng.W[index]:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                tensor.uniform_(0.1, 0.9)
        eng.base.embeddings.weight.normal_(0, 0.02)
        eng.lm_w_m.normal_(0, 0.02)


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _new_state(eng, device: str):
    sa = torch.zeros(
        eng.L, 1, eng.H, eng.N, eng.N,
        dtype=torch.float32,
        device=device,
    )
    xp = torch.zeros(eng.L, 1, eng.hidden, dtype=torch.float16, device=device)
    xf = torch.zeros_like(xp)
    vf = torch.zeros(1, eng.hidden, dtype=torch.float16, device=device)
    return sa, xp, xf, vf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--compare-addcmul",
        action="store_true",
        help="reproduce the rejected addcmul shift-mix numerical/performance A/B",
    )
    parser.add_argument(
        "--compare-greedy",
        action="store_true",
        help="A/B host argmax/token refill against a graph-resident greedy chain",
    )
    parser.add_argument("--correctness-steps", type=int, default=64)
    parser.add_argument(
        "--cpp-source",
        default=os.path.join(HERE, "rwkv7_ascend_v3.cpp"),
    )
    args = parser.parse_args()

    eng = build_synthetic_engine(
        args.cpp_source,
        device=args.device,
        layers=args.layers,
        heads=args.heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
    )
    if args.compare_addcmul:
        _randomize_synthetic_weights(eng, seed=20260713)

    addcmul_eng = None
    if args.compare_addcmul:
        addcmul_eng = copy.copy(eng)
        addcmul_eng.mod = load(
            name="rwkv7_ascend_graph_addcmul",
            sources=[args.cpp_source],
            verbose=False,
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-DRWKV7_USE_ADDCMUL_SHIFT_MIX=1",
            ],
        )

    # Load both extension variants before capturing either graph.  Loading a
    # second module after capture can interpose the shared C++ entry-point
    # symbols and invalidate an otherwise clean A/B comparison.
    legacy_decoder = NpuGraphDecoder(eng, capture_embedding=False)
    legacy_decoder.capture()
    decoder = NpuGraphDecoder(eng, capture_embedding=True)
    decoder.capture()

    addcmul_decoder = None
    if addcmul_eng is not None:
        addcmul_decoder = NpuGraphDecoder(addcmul_eng, capture_embedding=True)
        addcmul_decoder.capture()

    greedy_decoder = None
    if args.compare_greedy:
        greedy_decoder = NpuGraphDecoder(
            eng,
            capture_embedding=True,
            capture_greedy_token=True,
        )
        greedy_decoder.capture()

    legacy_state = _new_state(eng, args.device)
    captured_state = _new_state(eng, args.device)
    token = 42

    correctness_legacy = _new_state(eng, args.device)
    correctness_captured = _new_state(eng, args.device)
    with torch.no_grad():
        for current_token in [42, 7, 1024, 13]:
            expected = legacy_decoder.decode(
                current_token, *correctness_legacy
            ).clone()
            actual = decoder.decode(
                torch.tensor([current_token], device=args.device),
                *correctness_captured,
            ).clone()
            if not torch.equal(expected, actual):
                raise AssertionError(
                    "captured embedding logits differ for token %d" % current_token
                )
        for expected, actual in zip(correctness_legacy, correctness_captured):
            if not torch.equal(expected, actual):
                raise AssertionError("captured embedding recurrent state differs")
    print("correctness legacy_vs_captured bit_exact=true", flush=True)

    if greedy_decoder is not None:
        host_state = _new_state(eng, args.device)
        graph_state = _new_state(eng, args.device)
        host_token = token
        graph_token = token
        greedy_matches = 0
        bit_exact = True
        with torch.no_grad():
            for step in range(args.correctness_steps):
                host_logits = decoder.decode(host_token, *host_state).clone()
                graph_logits, graph_token = greedy_decoder.decode_greedy(
                    graph_token if step == 0 else None,
                    *graph_state,
                    reuse_token=step > 0,
                )
                graph_logits = graph_logits.clone()
                host_token = int(host_logits.argmax().item())
                greedy_matches += int(host_token == graph_token)
                bit_exact = bit_exact and torch.equal(host_logits, graph_logits)
                bit_exact = bit_exact and all(
                    torch.equal(reference, actual)
                    for reference, actual in zip(host_state, graph_state)
                )
        print(
            "correctness host_vs_graph_greedy bit_exact=%s greedy=%d/%d"
            % (str(bit_exact).lower(), greedy_matches, args.correctness_steps),
            flush=True,
        )
        if not bit_exact or greedy_matches != args.correctness_steps:
            raise AssertionError("graph-resident greedy token chain diverged")

    if addcmul_decoder is not None:
        reference_state = _new_state(eng, args.device)
        addcmul_state = _new_state(eng, args.device)
        min_cosine = 1.0
        max_logit_abs = 0.0
        max_state_abs = 0.0
        greedy_matches = 0
        reference_token = token
        addcmul_token = token
        with torch.no_grad():
            for _ in range(args.correctness_steps):
                reference_logits = decoder.decode(
                    reference_token, *reference_state
                ).clone()
                addcmul_logits = addcmul_decoder.decode(
                    addcmul_token, *addcmul_state
                ).clone()
                torch.npu.synchronize()
                cosine = torch.nn.functional.cosine_similarity(
                    reference_logits.float(), addcmul_logits.float()
                ).item()
                min_cosine = min(min_cosine, cosine)
                max_logit_abs = max(
                    max_logit_abs,
                    (reference_logits.float() - addcmul_logits.float())
                    .abs()
                    .max()
                    .item(),
                )
                for reference, actual in zip(reference_state, addcmul_state):
                    max_state_abs = max(
                        max_state_abs,
                        (reference.float() - actual.float()).abs().max().item(),
                    )
                reference_token = int(reference_logits.argmax().item())
                addcmul_token = int(addcmul_logits.argmax().item())
                greedy_matches += int(reference_token == addcmul_token)
        print(
            "correctness default_vs_addcmul min_cosine=%.9f "
            "max_logit_abs=%.6f max_state_abs=%.6f greedy=%d/%d"
            % (
                min_cosine,
                max_logit_abs,
                max_state_abs,
                greedy_matches,
                args.correctness_steps,
            ),
            flush=True,
        )

    def replay_only():
        legacy_decoder.graph.replay()

    def embedding_and_replay():
        legacy_decoder.emb.copy_(
            eng.base.embeddings(torch.tensor([token], device=args.device))
        )
        legacy_decoder.graph.replay()

    def legacy_production_decode():
        legacy_decoder.decode(token, *legacy_state)

    def production_decode():
        decoder.decode(token, *captured_state)

    if addcmul_decoder is not None:
        addcmul_perf_state = _new_state(eng, args.device)

        def addcmul_production_decode():
            addcmul_decoder.decode(token, *addcmul_perf_state)

    if greedy_decoder is not None:
        greedy_host_state = _new_state(eng, args.device)
        greedy_graph_state = _new_state(eng, args.device)
        greedy_host_token = token
        greedy_graph_token = token
        greedy_graph_started = False

        def greedy_host_roundtrip():
            nonlocal greedy_host_token
            logits = decoder.decode(greedy_host_token, *greedy_host_state)
            greedy_host_token = int(logits.argmax().item())

        def greedy_graph_chain():
            nonlocal greedy_graph_token, greedy_graph_started
            _, greedy_graph_token = greedy_decoder.decode_greedy(
                greedy_graph_token if not greedy_graph_started else None,
                *greedy_graph_state,
                reuse_token=greedy_graph_started,
            )
            greedy_graph_started = True

    rows = [
        ("graph_replay", replay_only),
        ("embedding_plus_replay", embedding_and_replay),
        ("legacy_production", legacy_production_decode),
        ("captured_embedding", production_decode),
    ]
    if addcmul_decoder is not None:
        rows.append(("addcmul_shift_mix", addcmul_production_decode))
    if greedy_decoder is not None:
        rows.extend(
            [
                ("greedy_host_roundtrip", greedy_host_roundtrip),
                ("greedy_graph_chain", greedy_graph_chain),
            ]
        )
    print(
        "shape L=%d H=%d N=%d hidden=%d vocab=%d"
        % (eng.L, eng.H, eng.N, eng.hidden, args.vocab_size),
        flush=True,
    )
    timings = {}
    with torch.no_grad():
        for name, fn in rows:
            ms = _time_ms(fn, args.warmup, args.iterations)
            timings[name] = ms
            print("%-24s %8.3f ms  %8.1f tok/s" % (name, ms, 1000.0 / ms), flush=True)
    replay = timings["graph_replay"]
    production = timings["legacy_production"]
    print(
        "graph_external_overhead %8.3f ms  %6.2f%%"
        % (production - replay, (production / replay - 1.0) * 100.0),
        flush=True,
    )
    captured = timings["captured_embedding"]
    print(
        "captured_embedding_gain %8.3f ms  %6.2f%%  %6.2fx"
        % (
            production - captured,
            (production / captured - 1.0) * 100.0,
            production / captured,
        ),
        flush=True,
    )
    if addcmul_decoder is not None:
        addcmul = timings["addcmul_shift_mix"]
        print(
            "addcmul_gain_vs_default %8.3f ms  %6.2f%%  %6.2fx"
            % (
                captured - addcmul,
                (captured / addcmul - 1.0) * 100.0,
                captured / addcmul,
            ),
            flush=True,
        )
    if greedy_decoder is not None:
        greedy_host = timings["greedy_host_roundtrip"]
        greedy_graph = timings["greedy_graph_chain"]
        print(
            "greedy_graph_gain %8.3f ms  %6.2f%%  %6.2fx"
            % (
                greedy_host - greedy_graph,
                (greedy_host / greedy_graph - 1.0) * 100.0,
                greedy_host / greedy_graph,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
