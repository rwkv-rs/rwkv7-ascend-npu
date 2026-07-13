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


def _cann_lib_dir(cann_home: str) -> str:
    for machine in ("aarch64-linux", "x86_64-linux"):
        candidate = os.path.join(cann_home, machine, "lib64")
        if os.path.exists(os.path.join(candidate, "libascendcl.so")):
            return candidate
    raise FileNotFoundError("cannot find CANN runtime lib64 under " + cann_home)


def _matrix(rows: int, cols: int, device: str) -> torch.Tensor:
    # The values do not affect GEMM scheduling.  Small constants keep repeated
    # recurrent updates finite while avoiding a slow checkpoint dependency.
    return torch.full((rows, cols), 1e-3, dtype=torch.float16, device=device)


def _vector(size: int, device: str, value: float = 0.0) -> torch.Tensor:
    return torch.full((size,), value, dtype=torch.float16, device=device)


def _mix_project_group(weight: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Fold x + (previous - x) * mix into a two-input linear weight."""
    weight_fp32 = weight.float()
    mix_fp32 = mix.float().unsqueeze(0)
    return torch.cat(
        (weight_fp32 * (1.0 - mix_fp32), weight_fp32 * mix_fp32), dim=1
    ).to(weight.dtype)


def _make_mix_project_weight(eng, li: int) -> torch.Tensor:
    groups = (
        (eng.W[0][li], eng.W[18][li]),
        (eng.W[1][li], eng.W[20][li]),
        (eng.W[2][li], eng.W[21][li]),
        (eng.W[7][li], eng.W[19][li]),
        (eng.W[9][li], eng.W[22][li]),
        (eng.W[11][li], eng.W[23][li]),
        (eng.W[13][li], eng.W[21][li]),
    )
    return torch.cat(
        [_mix_project_group(weight, mix) for weight, mix in groups], dim=0
    ).contiguous()


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
    rkv_bmm = [
        torch.stack((rw[li].t(), kw[li].t(), vw[li].t())).contiguous()
        for li in range(layers)
    ]
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
    zero_bias = _vector(hidden, device)
    lowrank_first = [
        torch.stack((w0[li].t(), a0[li].t(), g0[li].t(), v0[li].t()))
        .contiguous()
        for li in range(layers)
    ]
    lowrank_second = [
        torch.stack((w2[li].t(), a2[li].t(), g2[li].t(), v2[li].t()))
        .contiguous()
        for li in range(layers)
    ]
    lowrank_bias = [
        torch.stack((w2b[li], a2b[li], zero_bias, v2b[li]))[:, None, :]
        .contiguous()
        for li in range(layers)
    ]
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
        rw, kw, vw, rkv_bmm, ow, fkw, fvw,
        w0, w2, a0, a2, g0, g2, v0, v2,
        w2b, a2b, v2b,
        xr, xw, xk, xv, xa, xg,
        kk, ka, rk, gnw, gnb, fxk,
        anw, anb, fnw, fnb, pnw, pnb,
        lowrank_first, lowrank_second, lowrank_bias,
    )
    eng.W = eng.W + (
        [_make_mix_project_weight(eng, li) for li in range(layers)],
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
        for index in list(range(3)) + list(range(4, 15)) + [15, 16, 17]:
            for tensor in eng.W[index]:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                tensor.normal_(0, 0.02 if tensor.dim() > 1 else 0.01)
        for index in list(range(18, 24)) + [29]:
            for tensor in eng.W[index]:
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                tensor.uniform_(0.1, 0.9)
        eng.base.embeddings.weight.normal_(0, 0.02)
        eng.lm_w_m.normal_(0, 0.02)
        for li, packed in enumerate(eng.W[3]):
            packed.copy_(
                torch.stack(
                    (eng.W[0][li].t(), eng.W[1][li].t(), eng.W[2][li].t())
                )
            )
        for li in range(eng.L):
            eng.W[36][li].copy_(
                torch.stack(
                    (
                        eng.W[7][li].t(),
                        eng.W[9][li].t(),
                        eng.W[11][li].t(),
                        eng.W[13][li].t(),
                    )
                )
            )
            eng.W[37][li].copy_(
                torch.stack(
                    (
                        eng.W[8][li].t(),
                        eng.W[10][li].t(),
                        eng.W[12][li].t(),
                        eng.W[14][li].t(),
                    )
                )
            )
            eng.W[38][li].copy_(
                torch.stack(
                    (
                        eng.W[15][li],
                        eng.W[16][li],
                        torch.zeros_like(eng.W[15][li]),
                        eng.W[17][li],
                    )
                )[:, None, :]
            )
            eng.W[39][li].copy_(_make_mix_project_weight(eng, li))


def _format_engine_matrices(
    eng, npu_format: int, *, skip_groups: set[int] | None = None
) -> None:
    """Prepack shared matrix weights while preserving vector parameters."""
    skip_groups = skip_groups or set()
    formatted = {}
    groups = []
    for group_index, group in enumerate(eng.W):
        new_group = []
        for tensor in group:
            if tensor.dim() != 2 or group_index in skip_groups:
                new_group.append(tensor)
                continue
            key = tensor.data_ptr()
            if key not in formatted:
                formatted[key] = torch_npu.npu_format_cast(tensor, npu_format)
            new_group.append(formatted[key])
        groups.append(new_group)
    eng.W = tuple(groups)


def _prepare_direct_norm_parameters(eng) -> None:
    """Preconvert static norm affine parameters for the direct fp32 kernels."""
    groups = list(eng.W)
    for index in (27, 28, 30, 31, 32, 33, 34, 35):
        groups[index] = [tensor.float() for tensor in groups[index]]
    eng.W = tuple(groups)
    eng.fnorm_w = eng.fnorm_w.float()
    eng.fnorm_b = eng.fnorm_b.float()


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
    parser.add_argument(
        "--compare-ascendc-shift-mix2",
        action="store_true",
        help="A/B the opt-in AscendC two-output shift-mix custom op",
    )
    parser.add_argument(
        "--compare-foreach-shift-mix",
        action="store_true",
        help="A/B exact foreach Mul then foreach Add attention shift-mix",
    )
    parser.add_argument(
        "--compare-ascendc-shift-mix1",
        "--compare-ascendc-direct",
        dest="compare_ascendc_shift_mix1",
        action="store_true",
        help="A/B the direct AscendC fused decode backend",
    )
    parser.add_argument("--correctness-steps", type=int, default=64)
    parser.add_argument(
        "--profile-direct-dir",
        help="write a one-step torch_npu profiler trace for the direct fused graph",
    )
    parser.add_argument(
        "--direct-fractal-nz",
        action="store_true",
        help="prepack direct-path matrix weights to FRACTAL_NZ (format 29)",
    )
    parser.add_argument(
        "--direct-nd-weight",
        action="append",
        choices=("output", "ffn", "mix"),
        default=[],
        help="leave one direct-path matrix group in ND for layout A/B",
    )
    parser.add_argument(
        "--direct-nz-lm-head",
        action="store_true",
        help="prepack the direct-path LM head to FRACTAL_NZ",
    )
    parser.add_argument(
        "--direct-rkv-bmm",
        action="store_true",
        help="pack R/K/V projection inputs and use one graph-capturable BMM",
    )
    parser.add_argument(
        "--direct-dplr-state",
        action="store_true",
        help="use the rank-one DPLR state update identity (experimental)",
    )
    parser.add_argument(
        "--direct-rank1-row-blocks",
        type=int,
        default=2,
        choices=(1, 2),
        help="Vector Core row partitions per head for the rank-one kernel",
    )
    parser.add_argument(
        "--direct-lowrank-bmm",
        action="store_true",
        help="batch the four low-rank projection paths (experimental)",
    )
    parser.add_argument(
        "--direct-mix-project",
        action="store_true",
        help="fold attention shift-mix into one packed NZ projection",
    )
    parser.add_argument(
        "--direct-recurrence-prep",
        action="store_true",
        help="fuse low-rank postprocess, k/a prep, and value mixing per head",
    )
    parser.add_argument(
        "--direct-fused-recurrence-state",
        action="store_true",
        help="fuse recurrence preparation with the rank-one state update",
    )
    parser.add_argument(
        "--direct-inplace-state",
        action="store_true",
        help="update the fp32 recurrent state cache in place",
    )
    parser.add_argument(
        "--direct-fused-ffn-prep",
        action="store_true",
        help="fuse residual add, FFN LayerNorm, and FFN shift-mix",
    )
    parser.add_argument(
        "--direct-fused-next-attn",
        action="store_true",
        help="fuse the prior FFN add with next-layer attention preparation",
    )
    parser.add_argument(
        "--direct-fused-final-norm",
        action="store_true",
        help="fuse the final FFN residual add with final LayerNorm",
    )
    parser.add_argument(
        "--direct-fused-embed-norm2",
        action="store_true",
        help="fuse embedding lookup with pre/attention LayerNorm",
    )
    parser.add_argument(
        "--cpp-source",
        default=os.path.join(HERE, "rwkv7_ascend_v3.cpp"),
    )
    args = parser.parse_args()
    if args.direct_mix_project and not (
        args.direct_dplr_state
        and args.direct_rkv_bmm
        and args.direct_lowrank_bmm
        and args.direct_fractal_nz
    ):
        parser.error(
            "--direct-mix-project requires --direct-dplr-state, "
            "--direct-rkv-bmm, --direct-lowrank-bmm, and --direct-fractal-nz"
        )
    if args.direct_recurrence_prep and not args.direct_mix_project:
        parser.error("--direct-recurrence-prep requires --direct-mix-project")
    if args.direct_fused_recurrence_state and not (
        args.direct_recurrence_prep
        and args.direct_dplr_state
        and args.direct_inplace_state
    ):
        parser.error(
            "--direct-fused-recurrence-state requires "
            "--direct-recurrence-prep, --direct-dplr-state, and "
            "--direct-inplace-state"
        )
    if args.direct_fused_recurrence_state and (
        args.head_size != 64 or args.direct_rank1_row_blocks != 2
    ):
        parser.error(
            "--direct-fused-recurrence-state requires --head-size 64 and "
            "--direct-rank1-row-blocks 2"
        )
    if args.direct_inplace_state and not args.direct_dplr_state:
        parser.error("--direct-inplace-state requires --direct-dplr-state")
    if args.direct_fused_ffn_prep and not args.direct_dplr_state:
        parser.error("--direct-fused-ffn-prep requires --direct-dplr-state")
    if args.direct_fused_next_attn and not args.direct_mix_project:
        parser.error("--direct-fused-next-attn requires --direct-mix-project")
    if args.direct_fused_final_norm and not args.direct_fused_next_attn:
        parser.error(
            "--direct-fused-final-norm requires --direct-fused-next-attn"
        )
    if args.direct_fused_embed_norm2 and not args.direct_fused_next_attn:
        parser.error(
            "--direct-fused-embed-norm2 requires --direct-fused-next-attn"
        )

    eng = build_synthetic_engine(
        args.cpp_source,
        device=args.device,
        layers=args.layers,
        heads=args.heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
    )
    if (
        args.compare_addcmul
        or args.compare_ascendc_shift_mix2
        or args.compare_foreach_shift_mix
        or args.compare_ascendc_shift_mix1
    ):
        _randomize_synthetic_weights(eng, seed=20260713)

    # The four packed direct-only groups extend the C++ argument list.  Keep a
    # live check that normal serving callers can still use the historical
    # weight contract through the pybind overload.
    legacy_weights = eng.W[:3] + eng.W[4:36]
    expanded_state = _new_state(eng, args.device)
    legacy_contract_state = _new_state(eng, args.device)
    contract_embedding = eng.base.embeddings(
        torch.tensor([42], device=args.device)
    )
    with torch.no_grad():
        expanded_logits = eng.mod.rwkv7_decode_full(
            contract_embedding,
            *eng.W,
            *expanded_state,
            eng.H,
            eng.N,
            eng.lm_w_m,
            eng.fnorm_w,
            eng.fnorm_b,
        ).clone()
        legacy_contract_logits = eng.mod.rwkv7_decode_full(
            contract_embedding,
            *legacy_weights,
            *legacy_contract_state,
            eng.H,
            eng.N,
            eng.lm_w_m,
            eng.fnorm_w,
            eng.fnorm_b,
        ).clone()
    torch.npu.synchronize()
    legacy_contract_exact = torch.equal(
        expanded_logits, legacy_contract_logits
    ) and all(
        torch.equal(expanded, legacy)
        for expanded, legacy in zip(expanded_state, legacy_contract_state)
    )
    print(
        "correctness legacy_weight_contract bit_exact="
        + str(legacy_contract_exact).lower(),
        flush=True,
    )
    if not legacy_contract_exact:
        raise AssertionError("legacy rwkv7_decode_full weight contract diverged")

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

    ascendc_eng = None
    if args.compare_ascendc_shift_mix2:
        cann_home = os.environ.get(
            "ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0"
        )
        custom_api = os.path.join(
            cann_home, "opp", "vendors", "customize", "op_api"
        )
        ascendc_eng = copy.copy(eng)
        ascendc_eng.mod = load(
            name="rwkv7_ascend_graph_shift_mix2",
            sources=[args.cpp_source],
            verbose=False,
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-DRWKV7_USE_ASCENDC_SHIFT_MIX2=1",
            ],
            extra_include_paths=[
                os.path.join(cann_home, "include"),
                os.path.join(custom_api, "include"),
            ],
            extra_ldflags=[
                "-L" + os.path.join(custom_api, "lib"),
                "-L" + os.path.join(cann_home, "lib64"),
                "-Wl,-rpath," + os.path.join(custom_api, "lib"),
                "-lcust_opapi",
                "-lascendcl",
                "-Wl,--allow-shlib-undefined",
            ],
        )

    foreach_eng = None
    if args.compare_foreach_shift_mix:
        foreach_eng = copy.copy(eng)
        foreach_eng.mod = load(
            name="rwkv7_ascend_graph_foreach_shift_mix",
            sources=[args.cpp_source],
            verbose=False,
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-DRWKV7_USE_FOREACH_SHIFT_MIX=1",
            ],
        )

    shift_mix1_eng = None
    if args.compare_ascendc_shift_mix1:
        torch_npu_root = os.path.dirname(torch_npu.__file__)
        cann_home = os.environ.get(
            "ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0"
        )
        cann_lib = _cann_lib_dir(cann_home)
        direct_build = os.environ.get(
            "RWKV7_ASCENDC_DIRECT_BUILD_DIR",
            "/tmp/rwkv7_ascend_direct/build",
        )
        direct_include = os.path.join(
            direct_build, "include", "rwkv7_shift_mix1_kernel"
        )
        direct_library = os.path.join(
            direct_build, "lib", "librwkv7_shift_mix1_kernel.a"
        )
        direct_library6 = os.path.join(
            direct_build, "lib", "librwkv7_shift_mix6_kernel.a"
        )
        direct_state_library = os.path.join(
            direct_build, "lib", "librwkv7_state_post_kernel.a"
        )
        direct_k_prep_library = os.path.join(
            direct_build, "lib", "librwkv7_k_prep_kernel.a"
        )
        direct_relu_square_library = os.path.join(
            direct_build, "lib", "librwkv7_relu_square_kernel.a"
        )
        direct_value_mix_library = os.path.join(
            direct_build, "lib", "librwkv7_value_mix_kernel.a"
        )
        direct_head_scaled_add_library = os.path.join(
            direct_build, "lib", "librwkv7_head_scaled_add_kernel.a"
        )
        direct_outer_products_library = os.path.join(
            direct_build, "lib", "librwkv7_outer_products_kernel.a"
        )
        direct_sk_output_library = os.path.join(
            direct_build, "lib", "librwkv7_sk_output_kernel.a"
        )
        direct_normalize_k_library = os.path.join(
            direct_build, "lib", "librwkv7_normalize_k_kernel.a"
        )
        direct_w_pre_library = os.path.join(
            direct_build, "lib", "librwkv7_w_pre_kernel.a"
        )
        direct_k_prep_normalize_library = os.path.join(
            direct_build, "lib", "librwkv7_k_prep_normalize_kernel.a"
        )
        direct_state_rank1_output_library = os.path.join(
            direct_build, "lib", "librwkv7_state_rank1_output_kernel.a"
        )
        direct_groupnorm_sk_library = os.path.join(
            direct_build, "lib", "librwkv7_groupnorm_sk_kernel.a"
        )
        direct_lowrank_activate_library = os.path.join(
            direct_build, "lib", "librwkv7_lowrank_activate_kernel.a"
        )
        direct_lowrank_post_library = os.path.join(
            direct_build, "lib", "librwkv7_lowrank_post_kernel.a"
        )
        direct_recurrence_prep_library = os.path.join(
            direct_build, "lib", "librwkv7_recurrence_prep_kernel.a"
        )
        direct_recurrence_state_library = os.path.join(
            direct_build, "lib", "librwkv7_recurrence_state_kernel.a"
        )
        direct_ffn_prep_library = os.path.join(
            direct_build, "lib", "librwkv7_ffn_prep_kernel.a"
        )
        direct_attn_prep_library = os.path.join(
            direct_build, "lib", "librwkv7_attn_prep_kernel.a"
        )
        direct_embedding_library = os.path.join(
            direct_build, "lib", "librwkv7_embedding_kernel.a"
        )
        direct_embedding_norm2_library = os.path.join(
            direct_build, "lib", "librwkv7_embedding_norm2_kernel.a"
        )
        direct_layer_norm_library = os.path.join(
            direct_build, "lib", "librwkv7_layer_norm_kernel.a"
        )
        direct_add_layer_norm_library = os.path.join(
            direct_build, "lib", "librwkv7_add_layer_norm_kernel.a"
        )
        direct_concat2_library = os.path.join(
            direct_build, "lib", "librwkv7_concat2_kernel.a"
        )
        shift_mix1_eng = copy.copy(eng)
        if args.direct_dplr_state:
            _prepare_direct_norm_parameters(shift_mix1_eng)
        direct_cflags = [
            "-O3",
            "-std=c++17",
            "-DRWKV7_USE_ASCENDC_SHIFT_MIX1_DIRECT=1",
        ]
        if args.direct_rkv_bmm:
            direct_cflags.append("-DRWKV7_USE_RKV_BMM=1")
        if args.direct_dplr_state:
            direct_cflags.append("-DRWKV7_USE_DPLR_STATE=1")
            direct_cflags.append(
                f"-DRWKV7_RANK1_ROW_BLOCKS={args.direct_rank1_row_blocks}"
            )
        if args.direct_lowrank_bmm:
            direct_cflags.append("-DRWKV7_USE_LOWRANK_BMM=1")
        if args.direct_mix_project:
            direct_cflags.append("-DRWKV7_USE_MIX_PROJECT=1")
        if args.direct_recurrence_prep:
            direct_cflags.append("-DRWKV7_USE_RECURRENCE_PREP=1")
        if args.direct_fused_recurrence_state:
            direct_cflags.append("-DRWKV7_USE_FUSED_RECURRENCE_STATE=1")
        if args.direct_inplace_state:
            direct_cflags.append("-DRWKV7_USE_INPLACE_STATE=1")
        if args.direct_fused_ffn_prep:
            direct_cflags.append("-DRWKV7_USE_FUSED_FFN_PREP=1")
        if args.direct_fused_next_attn:
            direct_cflags.append("-DRWKV7_USE_FUSED_NEXT_ATTN=1")
        if args.direct_fused_final_norm:
            direct_cflags.append("-DRWKV7_USE_FUSED_FINAL_NORM=1")
        if args.direct_fused_embed_norm2:
            direct_cflags.append("-DRWKV7_USE_FUSED_EMBED_NORM2=1")
        shift_mix1_eng.mod = load(
            name=(
                "rwkv7_ascend_graph_shift_mix1_direct"
                + ("_rkv_bmm" if args.direct_rkv_bmm else "")
                + ("_dplr_state" if args.direct_dplr_state else "")
                + (
                    f"_rb{args.direct_rank1_row_blocks}"
                    if args.direct_dplr_state
                    else ""
                )
                + ("_lowrank_bmm" if args.direct_lowrank_bmm else "")
                + ("_mix_project" if args.direct_mix_project else "")
                + ("_recurrence_prep" if args.direct_recurrence_prep else "")
                + (
                    "_recurrence_state"
                    if args.direct_fused_recurrence_state
                    else ""
                )
                + ("_inplace_state" if args.direct_inplace_state else "")
                + ("_ffn_prep" if args.direct_fused_ffn_prep else "")
                + ("_next_attn" if args.direct_fused_next_attn else "")
                + ("_final_norm" if args.direct_fused_final_norm else "")
                + ("_embed_norm2" if args.direct_fused_embed_norm2 else "")
            ),
            sources=[args.cpp_source],
            verbose=False,
            extra_cflags=direct_cflags,
            extra_include_paths=[
                os.path.join(torch_npu_root, "include"),
                os.path.join(
                    torch_npu_root, "include", "third_party", "acl", "inc"
                ),
                direct_include,
                os.path.join(
                    direct_build, "include", "rwkv7_shift_mix6_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_state_post_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_k_prep_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_relu_square_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_value_mix_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_head_scaled_add_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_outer_products_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_sk_output_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_normalize_k_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_w_pre_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_k_prep_normalize_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_state_rank1_output_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_groupnorm_sk_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_lowrank_activate_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_lowrank_post_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_recurrence_prep_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_recurrence_state_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_ffn_prep_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_attn_prep_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_embedding_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_embedding_norm2_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_layer_norm_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_add_layer_norm_kernel"
                ),
                os.path.join(
                    direct_build, "include", "rwkv7_concat2_kernel"
                ),
            ],
            extra_ldflags=[
                direct_library,
                direct_library6,
                direct_state_library,
                direct_k_prep_library,
                direct_relu_square_library,
                direct_value_mix_library,
                direct_head_scaled_add_library,
                direct_outer_products_library,
                direct_sk_output_library,
                direct_normalize_k_library,
                direct_w_pre_library,
                direct_k_prep_normalize_library,
                direct_state_rank1_output_library,
                direct_groupnorm_sk_library,
                direct_lowrank_activate_library,
                direct_lowrank_post_library,
                direct_recurrence_prep_library,
                direct_recurrence_state_library,
                direct_ffn_prep_library,
                direct_attn_prep_library,
                direct_embedding_library,
                direct_embedding_norm2_library,
                direct_layer_norm_library,
                direct_add_layer_norm_library,
                direct_concat2_library,
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
        if args.direct_fractal_nz:
            nd_group_indices = {
                "output": {4},
                "ffn": {5, 6},
                "mix": {39},
            }
            skip_groups = set().union(
                *(nd_group_indices[name] for name in args.direct_nd_weight)
            )
            _format_engine_matrices(
                shift_mix1_eng, 29, skip_groups=skip_groups
            )
            if args.direct_nz_lm_head:
                shift_mix1_eng.lm_w_m = torch_npu.npu_format_cast(
                    shift_mix1_eng.lm_w_m, 29
                )

    # Load both extension variants before capturing either graph.  Loading a
    # second module after capture can interpose the shared C++ entry-point
    # symbols and invalidate an otherwise clean A/B comparison.
    if ascendc_eng is not None:
        eager_reference_state = _new_state(eng, args.device)
        eager_ascendc_state = _new_state(eng, args.device)
        eager_embedding = eng.base.embeddings(
            torch.tensor([42], device=args.device)
        )
        with torch.no_grad():
            eager_reference = eng.mod.rwkv7_decode_full(
                eager_embedding,
                *eng.W,
                *eager_reference_state,
                eng.H,
                eng.N,
                eng.lm_w_m,
                eng.fnorm_w,
                eng.fnorm_b,
            ).clone()
            eager_ascendc = ascendc_eng.mod.rwkv7_decode_full(
                eager_embedding,
                *ascendc_eng.W,
                *eager_ascendc_state,
                ascendc_eng.H,
                ascendc_eng.N,
                ascendc_eng.lm_w_m,
                ascendc_eng.fnorm_w,
                ascendc_eng.fnorm_b,
            ).clone()
        torch.npu.synchronize()
        eager_cosine = torch.nn.functional.cosine_similarity(
            eager_reference.float(), eager_ascendc.float()
        ).item()
        eager_max_abs = (
            eager_reference.float() - eager_ascendc.float()
        ).abs().max().item()
        print(
            "correctness eager_default_vs_ascendc_shift_mix2 "
            "bit_exact=%s cosine=%.9f max_abs=%.6g"
            % (
                str(torch.equal(eager_reference, eager_ascendc)).lower(),
                eager_cosine,
                eager_max_abs,
            ),
            flush=True,
        )

        reference_pieces = eng.mod.rwkv7_layer0_pieces(
            eager_embedding,
            *eng.W,
            *_new_state(eng, args.device),
            eng.H,
            eng.N,
        )
        ascendc_pieces = ascendc_eng.mod.rwkv7_layer0_pieces(
            eager_embedding,
            *ascendc_eng.W,
            *_new_state(eng, args.device),
            ascendc_eng.H,
            ascendc_eng.N,
        )
        torch.npu.synchronize()
        print(
            "correctness eager_layer0_pieces bit_exact="
            + "/".join(
                str(torch.equal(reference, actual)).lower()
                for reference, actual in zip(reference_pieces, ascendc_pieces)
            ),
            flush=True,
        )

    if shift_mix1_eng is not None:
        eager_reference_state = _new_state(eng, args.device)
        eager_direct_state = _new_state(eng, args.device)
        eager_token_ids = torch.tensor([42], device=args.device)
        eager_embedding = eng.base.embeddings(eager_token_ids)
        eager_direct_embedding = (
            shift_mix1_eng.mod.rwkv7_embedding_norm2(
                eager_token_ids,
                shift_mix1_eng.base.embeddings.weight,
                shift_mix1_eng.W[34][0],
                shift_mix1_eng.W[35][0],
                shift_mix1_eng.W[30][0],
                shift_mix1_eng.W[31][0],
                eager_direct_state[1][0],
            )
            if hasattr(shift_mix1_eng.mod, "rwkv7_embedding_norm2")
            else eager_embedding
        )
        with torch.no_grad():
            eager_reference = eng.mod.rwkv7_decode_full(
                eager_embedding,
                *eng.W,
                *eager_reference_state,
                eng.H,
                eng.N,
                eng.lm_w_m,
                eng.fnorm_w,
                eng.fnorm_b,
            ).clone()
            eager_direct = shift_mix1_eng.mod.rwkv7_decode_full(
                eager_direct_embedding,
                *shift_mix1_eng.W,
                *eager_direct_state,
                shift_mix1_eng.H,
                shift_mix1_eng.N,
                shift_mix1_eng.lm_w_m,
                shift_mix1_eng.fnorm_w,
                shift_mix1_eng.fnorm_b,
            ).clone()
        torch.npu.synchronize()
        print(
            "correctness eager_default_vs_direct bit_exact=%s max_abs=%.6g"
            % (
                str(torch.equal(eager_reference, eager_direct)).lower(),
                (eager_reference.float() - eager_direct.float())
                .abs()
                .max()
                .item(),
            ),
            flush=True,
        )

        reference_pieces = eng.mod.rwkv7_layer0_pieces(
            eager_embedding,
            *eng.W,
            *_new_state(eng, args.device),
            eng.H,
            eng.N,
        )
        direct_pieces = shift_mix1_eng.mod.rwkv7_layer0_pieces(
            eager_embedding,
            *shift_mix1_eng.W,
            *_new_state(eng, args.device),
            shift_mix1_eng.H,
            shift_mix1_eng.N,
        )
        torch.npu.synchronize()
        print(
            "correctness eager_direct_layer0_pieces bit_exact="
            + "/".join(
                str(torch.equal(reference, actual)).lower()
                for reference, actual in zip(reference_pieces, direct_pieces)
            ),
            flush=True,
        )

    legacy_decoder = NpuGraphDecoder(eng, capture_embedding=False)
    legacy_decoder.capture()
    decoder = NpuGraphDecoder(eng, capture_embedding=True)
    decoder.capture()

    addcmul_decoder = None
    if addcmul_eng is not None:
        addcmul_decoder = NpuGraphDecoder(addcmul_eng, capture_embedding=True)
        addcmul_decoder.capture()

    ascendc_decoder = None
    if ascendc_eng is not None:
        ascendc_decoder = NpuGraphDecoder(ascendc_eng, capture_embedding=True)
        ascendc_decoder.capture()

    foreach_decoder = None
    if foreach_eng is not None:
        foreach_decoder = NpuGraphDecoder(foreach_eng, capture_embedding=True)
        foreach_decoder.capture()

    shift_mix1_decoder = None
    if shift_mix1_eng is not None:
        shift_mix1_decoder = NpuGraphDecoder(
            shift_mix1_eng, capture_embedding=True
        )
        shift_mix1_decoder.capture()

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

    if ascendc_decoder is not None:
        reference_state = _new_state(eng, args.device)
        ascendc_state = _new_state(eng, args.device)
        reference_token = token
        ascendc_token = token
        bit_exact = True
        greedy_matches = 0
        with torch.no_grad():
            for _ in range(args.correctness_steps):
                reference_logits = decoder.decode(
                    reference_token, *reference_state
                ).clone()
                ascendc_logits = ascendc_decoder.decode(
                    ascendc_token, *ascendc_state
                ).clone()
                reference_token = int(reference_logits.argmax().item())
                ascendc_token = int(ascendc_logits.argmax().item())
                greedy_matches += int(reference_token == ascendc_token)
                bit_exact = bit_exact and torch.equal(
                    reference_logits, ascendc_logits
                )
                bit_exact = bit_exact and all(
                    torch.equal(reference, actual)
                    for reference, actual in zip(reference_state, ascendc_state)
                )
        print(
            "correctness default_vs_ascendc_shift_mix2 bit_exact=%s greedy=%d/%d"
            % (str(bit_exact).lower(), greedy_matches, args.correctness_steps),
            flush=True,
        )
        if not bit_exact or greedy_matches != args.correctness_steps:
            raise AssertionError("AscendC shift-mix2 recurrent decode diverged")

    if foreach_decoder is not None:
        reference_state = _new_state(eng, args.device)
        foreach_state = _new_state(eng, args.device)
        reference_token = token
        foreach_token = token
        bit_exact = True
        greedy_matches = 0
        with torch.no_grad():
            for _ in range(args.correctness_steps):
                reference_logits = decoder.decode(
                    reference_token, *reference_state
                ).clone()
                foreach_logits = foreach_decoder.decode(
                    foreach_token, *foreach_state
                ).clone()
                reference_token = int(reference_logits.argmax().item())
                foreach_token = int(foreach_logits.argmax().item())
                greedy_matches += int(reference_token == foreach_token)
                bit_exact = bit_exact and torch.equal(
                    reference_logits, foreach_logits
                )
                bit_exact = bit_exact and all(
                    torch.equal(reference, actual)
                    for reference, actual in zip(reference_state, foreach_state)
                )
        print(
            "correctness default_vs_foreach_shift_mix bit_exact=%s greedy=%d/%d"
            % (str(bit_exact).lower(), greedy_matches, args.correctness_steps),
            flush=True,
        )
        if not bit_exact or greedy_matches != args.correctness_steps:
            raise AssertionError("foreach shift-mix recurrent decode diverged")

    if shift_mix1_decoder is not None:
        reference_state = _new_state(eng, args.device)
        shift_mix1_state = _new_state(eng, args.device)
        reference_token = token
        shift_mix1_token = token
        bit_exact = True
        greedy_matches = 0
        min_logits_cosine = 1.0
        max_logits_abs = 0.0
        max_state_abs = 0.0
        with torch.no_grad():
            for _ in range(args.correctness_steps):
                reference_logits = decoder.decode(
                    reference_token, *reference_state
                ).clone()
                shift_mix1_logits = shift_mix1_decoder.decode(
                    shift_mix1_token, *shift_mix1_state
                ).clone()
                reference_token = int(reference_logits.argmax().item())
                shift_mix1_token = int(shift_mix1_logits.argmax().item())
                greedy_matches += int(reference_token == shift_mix1_token)
                min_logits_cosine = min(
                    min_logits_cosine,
                    torch.nn.functional.cosine_similarity(
                        reference_logits.float(), shift_mix1_logits.float()
                    ).item(),
                )
                max_logits_abs = max(
                    max_logits_abs,
                    (reference_logits.float() - shift_mix1_logits.float())
                    .abs()
                    .max()
                    .item(),
                )
                max_state_abs = max(
                    max_state_abs,
                    max(
                        (reference.float() - actual.float()).abs().max().item()
                        for reference, actual in zip(
                            reference_state, shift_mix1_state
                        )
                    ),
                )
                bit_exact = bit_exact and torch.equal(
                    reference_logits, shift_mix1_logits
                )
                bit_exact = bit_exact and all(
                    torch.equal(reference, actual)
                    for reference, actual in zip(
                        reference_state, shift_mix1_state
                    )
                )
        print(
            "correctness default_vs_ascendc_shift_mix1 "
            "bit_exact=%s greedy=%d/%d min_logits_cosine=%.9f "
            "max_logits_abs=%.6g max_state_abs=%.6g"
            % (
                str(bit_exact).lower(),
                greedy_matches,
                args.correctness_steps,
                min_logits_cosine,
                max_logits_abs,
                max_state_abs,
            ),
            flush=True,
        )
        dplr_pass = (
            args.direct_dplr_state
            and greedy_matches == args.correctness_steps
            and min_logits_cosine >= 0.9999
        )
        if (not bit_exact or greedy_matches != args.correctness_steps) and not dplr_pass:
            raise AssertionError("AscendC shift-mix1 recurrent decode diverged")

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

    if ascendc_decoder is not None:
        ascendc_perf_state = _new_state(eng, args.device)

        def ascendc_production_decode():
            ascendc_decoder.decode(token, *ascendc_perf_state)

    if foreach_decoder is not None:
        foreach_perf_state = _new_state(eng, args.device)

        def foreach_production_decode():
            foreach_decoder.decode(token, *foreach_perf_state)

    if shift_mix1_decoder is not None:
        shift_mix1_perf_state = _new_state(eng, args.device)

        def shift_mix1_production_decode():
            shift_mix1_decoder.decode(token, *shift_mix1_perf_state)

        shift_mix1_resident_state = _new_state(eng, args.device)
        shift_mix1_decoder.load_resident_state(*shift_mix1_resident_state)
        shift_mix1_decoder.replay_resident(token)

        def shift_mix1_resident_replay():
            shift_mix1_decoder.replay_resident()

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
    if ascendc_decoder is not None:
        rows.append(("ascendc_shift_mix2", ascendc_production_decode))
    if foreach_decoder is not None:
        rows.append(("foreach_shift_mix", foreach_production_decode))
    if shift_mix1_decoder is not None:
        rows.append(("ascendc_shift_mix1", shift_mix1_production_decode))
        rows.append(("ascendc_resident", shift_mix1_resident_replay))
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
    if args.profile_direct_dir:
        if shift_mix1_decoder is None:
            raise ValueError("--profile-direct-dir requires --compare-ascendc-shift-mix1")
        os.makedirs(args.profile_direct_dir, exist_ok=True)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                args.profile_direct_dir
            ),
            record_shapes=True,
            experimental_config=experimental_config,
        ) as profiler:
            shift_mix1_production_decode()
            torch.npu.synchronize()
            profiler.step()
        print("direct_profile_dir " + args.profile_direct_dir, flush=True)
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
    if ascendc_decoder is not None:
        ascendc = timings["ascendc_shift_mix2"]
        print(
            "ascendc_shift_mix2_gain %8.3f ms  %6.2f%%  %6.2fx"
            % (
                captured - ascendc,
                (captured / ascendc - 1.0) * 100.0,
                captured / ascendc,
            ),
            flush=True,
        )
    if foreach_decoder is not None:
        foreach = timings["foreach_shift_mix"]
        print(
            "foreach_shift_mix_gain %8.3f ms  %6.2f%%  %6.2fx"
            % (
                captured - foreach,
                (captured / foreach - 1.0) * 100.0,
                captured / foreach,
            ),
            flush=True,
        )
    if shift_mix1_decoder is not None:
        shift_mix1 = timings["ascendc_shift_mix1"]
        print(
            "ascendc_shift_mix1_gain %8.3f ms  %6.2f%%  %6.2fx"
            % (
                captured - shift_mix1,
                (captured / shift_mix1 - 1.0) * 100.0,
                captured / shift_mix1,
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
