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

from benchmark_metadata import (
    checkpoint_metadata,
    collect_cann_metadata,
    collect_npu_metadata,
    npu_device_id,
)
from npu_memory import PeakNPUMemorySampler
from rwkv7_chunk_scan import TorchChunkScanModule
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
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-DRWKV7_USE_PREFILL_SCAN=1",
            "-DRWKV7_USE_CHUNK_INVERSE=1",
            "-DRWKV7_USE_CHUNK_GATE_PREP=1",
            "-DRWKV7_USE_CHUNK_MASK=1",
            "-DRWKV7_USE_PREFILL_SHIFT_MIX=1",
        ],
        extra_include_paths=[
            os.path.join(torch_npu_root, "include"),
            os.path.join(torch_npu_root, "include", "third_party", "acl", "inc"),
            os.path.join(direct_build, "include", "rwkv7_prefill_scan_kernel"),
            os.path.join(direct_build, "include", "rwkv7_chunk_inverse_kernel"),
            os.path.join(direct_build, "include", "rwkv7_chunk_gate_prep_kernel"),
            os.path.join(direct_build, "include", "rwkv7_chunk_mask_kernel"),
            os.path.join(
                direct_build, "include", "rwkv7_prefill_shift_mix_kernel"
            ),
        ],
        extra_ldflags=[
            os.path.join(direct_build, "lib", "librwkv7_prefill_scan_kernel.a"),
            os.path.join(direct_build, "lib", "librwkv7_chunk_inverse_kernel.a"),
            os.path.join(direct_build, "lib", "librwkv7_chunk_gate_prep_kernel.a"),
            os.path.join(direct_build, "lib", "librwkv7_chunk_mask_kernel.a"),
            os.path.join(
                direct_build, "lib", "librwkv7_prefill_shift_mix_kernel.a"
            ),
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


def _stage_start(stage_events, name: str):
    if stage_events is None:
        return None
    event = torch.npu.Event(enable_timing=True)
    event.record()
    return event


def _stage_end(stage_events, name: str, started) -> None:
    if stage_events is None:
        return
    ended = torch.npu.Event(enable_timing=True)
    ended.record()
    stage_events.setdefault(name, []).append((started, ended))


def _summarize_stage_events(stage_events) -> dict[str, float]:
    return {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        for name, pairs in stage_events.items()
    }


def _run_torch_profile(
    eng,
    prompt_ids: torch.Tensor,
    device: str,
    scan_module,
    output_path: str,
) -> str:
    os.makedirs(output_path, exist_ok=True)
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    with torch_npu.profiler.profile(
        activities=activities,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_path),
        record_shapes=True,
        experimental_config=experimental_config,
    ) as profile:
        layer_major_prefill(eng, prompt_ids, device, scan_module)
        torch.npu.synchronize()
        profile.step()
    return f"torch_npu Level1 profile exported to {output_path}"


def layer_major_prefill(
    eng,
    token_ids: torch.Tensor,
    device: str,
    scan_module=None,
    initial_cache=None,
    stage_events=None,
):
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    batch_size, tokens = token_ids.shape
    if initial_cache is None:
        state_all, xpa_all, xpf_all, value_first_cache = _new_state(
            eng, device, batch_size
        )
    else:
        state_all, xpa_all, xpf_all, value_first_cache = initial_cache
        if state_all.shape[1] != batch_size:
            raise ValueError(
                f"cache batch {state_all.shape[1]} does not match input {batch_size}"
            )
    x = eng.base.embeddings(token_ids)
    value_first = None
    hidden = eng.hidden
    heads = eng.H
    head_size = eng.N
    prefetched_attn_norm = None
    prefetched_residual = None

    with torch.no_grad():
        for layer in range(eng.L):
            stage = _stage_start(stage_events, "norm_shift_mix")
            if prefetched_attn_norm is not None:
                residual = prefetched_residual
                h = prefetched_attn_norm
                prefetched_attn_norm = None
                prefetched_residual = None
            else:
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
            fused_module = getattr(scan_module, "inverse_module", scan_module)
            use_fused_shift_mix = bool(
                getattr(scan_module, "use_fused_shift_mix", False)
                and fused_module is not None
                and hasattr(fused_module, "rwkv7_prefill_shift_mix")
            )
            if use_fused_shift_mix:
                mixed = fused_module.rwkv7_prefill_shift_mix(
                    h,
                    xpa_all[layer],
                    *(eng.W[index][layer] for index in (18, 20, 21, 19, 22, 23)),
                    scan_module.fused_shift_mix_rows_per_block,
                )
                xr, xk, xv, xw, xa, xg = mixed[:6]
                lowrank_input = mixed[6:].reshape(
                    4, batch_size * tokens, hidden
                )
            else:
                previous = torch.cat(
                    (xpa_all[layer].unsqueeze(1), h[:, :-1]), dim=1
                )
                delta = previous - h
                mixed = torch.stack(
                    [
                        h + delta * eng.W[index][layer]
                        for index in (18, 20, 21, 19, 22, 23)
                    ]
                )
                xr, xk, xv, xw, xa, xg = mixed
                lowrank_input = torch.stack((xw, xa, xg, xv)).reshape(
                    4, batch_size * tokens, hidden
                )
            xpa_all[layer].copy_(h[:, -1])
            _stage_end(stage_events, "norm_shift_mix", stage)

            stage = _stage_start(stage_events, "rkv_projection")
            r = F.linear(xr, eng.W[0][layer])
            k = F.linear(xk, eng.W[1][layer])
            v = F.linear(xv, eng.W[2][layer])
            _stage_end(stage_events, "rkv_projection", stage)
            stage = _stage_start(stage_events, "lowrank_wavg")
            lowrank_hidden = torch.bmm(
                lowrank_input, eng.W[36][layer]
            )
            lowrank_hidden[0].tanh_()
            lowrank_hidden[2].sigmoid_()
            lowrank = (
                torch.bmm(lowrank_hidden, eng.W[37][layer])
                + eng.W[38][layer]
            ).view(4, batch_size, tokens, hidden)
            w_raw, a_raw, g, value_mix_raw = lowrank
            w_raw.sigmoid_()
            log_decay = w_raw.mul_(-0.6065306597126334)
            w = (
                log_decay
                if getattr(scan_module, "expects_log_decay", False)
                else torch.exp(log_decay)
            )
            a = a_raw.sigmoid_()
            _stage_end(stage_events, "lowrank_wavg", stage)
            stage = _stage_start(stage_events, "state_prep")
            kk = (k * eng.W[24][layer]).view(
                batch_size, tokens, heads, head_size
            )
            kk = kk / kk.norm(2, dim=-1, keepdim=True).clamp_min(1.0e-8)
            k = torch.addcmul(
                k,
                k * eng.W[25][layer],
                a - 1.0,
            )

            if layer == 0:
                value_first = v.clone()
                value_first_cache.copy_(value_first[:, -1])
            else:
                value_mix_raw.sigmoid_()
                v = v + (value_first - v) * value_mix_raw
            _stage_end(stage_events, "state_prep", stage)

            stage = _stage_start(stage_events, "recurrent_scan")
            current_state = state_all[layer]
            if scan_module is not None:
                if hasattr(scan_module, "stage_events"):
                    scan_module.stage_events = stage_events
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
                    getattr(scan_module, "scan_row_blocks", 2),
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
            _stage_end(stage_events, "recurrent_scan", stage)

            stage = _stage_start(stage_events, "output")
            detail = _stage_start(stage_events, "output_group_norm")
            if getattr(scan_module, "use_head_layer_norm", False):
                out = F.layer_norm(
                    recurrent.view(
                        batch_size * tokens, heads, head_size
                    ),
                    (head_size,),
                    None,
                    None,
                    eps=head_size * 1.0e-5,
                )
                out = (
                    out * eng.W[27][layer].view(1, heads, head_size)
                    + eng.W[28][layer].view(1, heads, head_size)
                ).view(batch_size * tokens, hidden)
            else:
                out = F.group_norm(
                    recurrent.view(batch_size * tokens, hidden),
                    heads,
                    eng.W[27][layer],
                    eng.W[28][layer],
                    eps=head_size * 1.0e-5,
                )
            _stage_end(stage_events, "output_group_norm", detail)
            detail = _stage_start(stage_events, "output_sk")
            sk = (
                r.view(batch_size, tokens, heads, head_size)
                * k.view(batch_size, tokens, heads, head_size)
                * eng.W[26][layer].view(1, 1, heads, head_size)
            ).sum(dim=-1, keepdim=True)
            _stage_end(stage_events, "output_sk", detail)
            detail = _stage_start(stage_events, "output_gate")
            out = torch.addcmul(
                out.view(batch_size, tokens, heads, head_size),
                sk,
                v.view(batch_size, tokens, heads, head_size),
            ).view(batch_size, tokens, hidden)
            out.mul_(g)
            _stage_end(stage_events, "output_gate", detail)
            detail = _stage_start(stage_events, "output_project")
            projected = F.linear(out, eng.W[4][layer])
            if getattr(scan_module, "use_fused_add_layer_norm", False):
                h2, _, _, x = torch_npu.npu_add_layer_norm(
                    projected,
                    residual,
                    eng.W[32][layer],
                    eng.W[33][layer],
                    1.0e-5,
                    True,
                )
            else:
                projected.add_(residual)
                x = projected
            _stage_end(stage_events, "output_project", detail)
            _stage_end(stage_events, "output", stage)

            stage = _stage_start(stage_events, "ffn")
            detail = _stage_start(stage_events, "ffn_norm_mix")
            if not getattr(scan_module, "use_fused_add_layer_norm", False):
                h2 = F.layer_norm(
                    x,
                    (hidden,),
                    eng.W[32][layer],
                    eng.W[33][layer],
                )
            if use_fused_shift_mix:
                k_ffn = fused_module.rwkv7_prefill_shift_mix1(
                    h2,
                    xpf_all[layer],
                    eng.W[29][layer],
                    scan_module.fused_shift_mix_rows_per_block,
                )
            else:
                ffn_previous = torch.cat(
                    (xpf_all[layer].unsqueeze(1), h2[:, :-1]), dim=1
                )
                k_ffn = h2 + (ffn_previous - h2) * eng.W[29][layer]
            xpf_all[layer].copy_(h2[:, -1])
            _stage_end(stage_events, "ffn_norm_mix", detail)
            detail = _stage_start(stage_events, "ffn_key")
            ffn_hidden = F.linear(k_ffn, eng.W[5][layer])
            _stage_end(stage_events, "ffn_key", detail)
            detail = _stage_start(stage_events, "ffn_activate_value")
            ffn_hidden.relu_()
            ffn_hidden.square_()
            ffn_out = F.linear(ffn_hidden, eng.W[6][layer])
            if (
                getattr(scan_module, "use_fused_add_layer_norm", False)
                and layer + 1 < eng.L
            ):
                prefetched_attn_norm, _, _, prefetched_residual = (
                    torch_npu.npu_add_layer_norm(
                        ffn_out,
                        x,
                        eng.W[30][layer + 1],
                        eng.W[31][layer + 1],
                        1.0e-5,
                        True,
                    )
                )
                x = prefetched_residual
            else:
                ffn_out.add_(x)
                x = ffn_out
            _stage_end(stage_events, "ffn_activate_value", detail)
            _stage_end(stage_events, "ffn", stage)

        stage = _stage_start(stage_events, "final_norm_head")
        last = F.layer_norm(
            x[:, -1],
            (hidden,),
            eng.fnorm_w,
            eng.fnorm_b,
        )
        logits = F.linear(last, eng.lm_w_m)
        _stage_end(stage_events, "final_norm_head", stage)
    return logits, (state_all, xpa_all, xpf_all, value_first_cache)


def _timed(function, *args):
    torch.npu.synchronize()
    started = time.perf_counter()
    output = function(*args)
    torch.npu.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def _full_decode_step(eng, token: torch.Tensor, cache):
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


class _BatchedNpuGraphDecode:
    """Fixed-address, graph-resident greedy decode for benchmark batches.

    The serving helper deliberately specializes on B=1 because it copies state
    to and from scheduler-owned slots.  Matrix benchmarking needs the same
    whole-step capture at B=4 without those round trips, so this small helper
    keeps the prompt cache resident and captures embedding, decode, argmax, and
    the next-token write in one graph.
    """

    def __init__(self, eng, cache, batch_size: int):
        self.eng = eng
        self.cache = tuple(torch.zeros_like(value) for value in cache)
        self.token_ids = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=eng.lm_w_m.device,
        )
        self.logits = None
        self.graph = None

    def _captured_step(self):
        self.logits, _ = _full_decode_step(
            self.eng,
            self.token_ids,
            self.cache,
        )
        self.token_ids.copy_(self.logits.argmax(dim=-1).reshape(-1))
        return self.logits

    def capture(self, warmup: int = 3):
        with torch.no_grad():
            for _ in range(warmup):
                self._captured_step()
        torch.npu.synchronize()
        side_stream = torch.npu.Stream()
        side_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side_stream), torch.no_grad():
            for _ in range(warmup):
                self._captured_step()
        torch.npu.current_stream().wait_stream(side_stream)
        self.graph = torch.npu.NPUGraph()
        with torch.no_grad(), torch.npu.graph(self.graph):
            self._captured_step()
        torch.npu.synchronize()

    def load(self, cache, token: torch.Tensor):
        for target, source in zip(self.cache, cache):
            target.copy_(source)
        self.token_ids.copy_(token.reshape(-1))

    def replay(self):
        self.graph.replay()
        return self.logits


def _decode_comparison(reference_logits, reference_cache, candidate_logits, candidate_cache):
    return {
        "greedy_match": bool(
            torch.equal(
                reference_logits.argmax(-1),
                candidate_logits.argmax(-1),
            )
        ),
        "logits_cosine": F.cosine_similarity(
            reference_logits.float(), candidate_logits.float()
        ).min().item(),
        "logits_max_abs": (
            reference_logits.float() - candidate_logits.float()
        ).abs().max().item(),
        "state_max_abs": max(
            (reference.float() - candidate.float()).abs().max().item()
            for reference, candidate in zip(reference_cache, candidate_cache)
        ),
    }


def _time_decode(
    eng,
    prompt_ids: torch.Tensor,
    device: str,
    scan_module,
    *,
    warmup: int,
    steps: int,
    use_npu_graph: bool,
):
    batch_size = prompt_ids.shape[0]
    logits, cache = layer_major_prefill(
        eng, prompt_ids, device, scan_module
    )
    token = logits.argmax(dim=-1).reshape(batch_size)
    use_full_decode = eng.mod is not None

    if use_npu_graph:
        if not use_full_decode:
            raise RuntimeError("NPU Graph decode requires rwkv7_decode_full")
        graph_decode = _BatchedNpuGraphDecode(eng, cache, batch_size)
        graph_decode.capture()

        eager_cache = tuple(value.clone() for value in cache)
        eager_logits, eager_cache = _full_decode_step(eng, token, eager_cache)
        graph_decode.load(cache, token)
        graph_logits = graph_decode.replay()
        torch.npu.synchronize()
        graph_correctness = _decode_comparison(
            eager_logits,
            eager_cache,
            graph_logits,
            graph_decode.cache,
        )

        graph_decode.load(cache, token)
        for _ in range(warmup):
            graph_decode.replay()
        graph_decode.load(cache, token)
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            graph_decode.replay()
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "backend": "npugraph_fused_full_greedy",
            "latency_ms_mean": elapsed_ms / steps,
            "tokens_per_second": batch_size * steps * 1000.0 / elapsed_ms,
            "graph_correctness": graph_correctness,
        }

    def step(current_token):
        if use_full_decode:
            return _full_decode_step(eng, current_token, cache)
        return layer_major_prefill(
            eng,
            current_token.view(batch_size, 1),
            device,
            scan_module,
            initial_cache=cache,
        )

    with torch.no_grad():
        for _ in range(warmup):
            logits, cache = step(token)
            token = logits.argmax(dim=-1).reshape(batch_size)
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            logits, cache = step(token)
            token = logits.argmax(dim=-1).reshape(batch_size)
        torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "backend": "fused_full" if use_full_decode else "layer_major",
        "latency_ms_mean": elapsed_ms / steps,
        "tokens_per_second": batch_size * steps * 1000.0 / elapsed_ms,
        "graph_correctness": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--decode-warmup", type=int, default=8)
    parser.add_argument(
        "--decode-npu-graph",
        action="store_true",
        help="capture embedding, full decode, and greedy token update in one graph",
    )
    parser.add_argument(
        "--loader-profile",
        choices=("full", "prefill_only"),
        default="full",
        help="prefill_only avoids decode-only duplicate R/K/V and raw low-rank weights",
    )
    parser.add_argument("--correctness-length", type=int, default=8)
    parser.add_argument(
        "--cpp-source",
        default=os.path.join(os.path.dirname(__file__), "rwkv7_ascend_v3.cpp"),
    )
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--stage-probe",
        action="store_true",
        help="record device-event latency for each layer-major fusion boundary",
    )
    parser.add_argument(
        "--torch-profile-output",
        help="write one torch_npu Level1 profile directory",
    )
    parser.add_argument(
        "--ascendc-scan",
        action="store_true",
        help="replace the sequential torch recurrence with the fused AscendC scan",
    )
    parser.add_argument(
        "--chunk-scan",
        action="store_true",
        help="use the opt-in PyTorch DPLR chunk/Cube scan prototype",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        choices=(4, 8, 16, 32, 64, 128, 256),
        default=16,
        help="DPLR chunk size (short correctness prompts use their greatest divisor)",
    )
    parser.add_argument(
        "--chunk-compute-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--fused-shift-mix",
        action="store_true",
        help="use the opt-in tiled AscendC tmix_mix6 prefill boundary",
    )
    parser.add_argument(
        "--fused-shift-mix-rows-per-block",
        type=int,
        choices=(4, 8, 16, 32),
        default=16,
        help="row tile for the opt-in prefill shift-mix kernels",
    )
    parser.add_argument(
        "--head-layer-norm",
        action="store_true",
        help="use equivalent per-head LayerNorm plus head-specific affine",
    )
    parser.add_argument(
        "--fused-add-layer-norm",
        action="store_true",
        help="fuse residual additions with adjacent layer norms",
    )
    parser.add_argument(
        "--dense-chunk-prefix",
        action="store_true",
        help="use the opt-in associative dense chunk-prefix prototype",
    )
    parser.add_argument(
        "--dense-prefix-algorithm",
        choices=("hillis", "tree", "tree_root", "blelloch"),
        default="hillis",
        help="associative scan used by the opt-in dense chunk-prefix path",
    )
    parser.add_argument(
        "--chunk-inverse-backend",
        choices=("native", "native_blocked"),
        default="native",
        help="unit-lower inverse/solve backend for the opt-in chunk scan",
    )
    parser.add_argument(
        "--chunk-inverse-base-size",
        type=int,
        choices=(16, 32, 64),
        default=32,
        help="native diagonal block used by native_blocked inverse",
    )
    parser.add_argument(
        "--scan-row-blocks",
        type=int,
        choices=(1, 2),
        default=2,
        help="number of vector-core row partitions per recurrent head",
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.decode_length < 1
        or args.decode_warmup < 0
        or args.iterations < 1
        or args.warmup < 0
    ):
        parser.error("batch-size/iterations must be positive and warmup non-negative")
    if args.ascendc_scan and args.chunk_scan:
        parser.error("--ascendc-scan and --chunk-scan are mutually exclusive")

    device_id = npu_device_id(args.device)
    load_memory_sampler = PeakNPUMemorySampler([device_id]).start()
    eng = build_blinkdl_engine(
        args.cpp_source,
        model_path=args.model_pth,
        device=args.device,
        include_mix_project=False,
        loader_profile=args.loader_profile,
    )
    load_peak_memory_mib = load_memory_sampler.stop()
    loaded_memory_mib = torch.npu.memory_allocated(args.device) / 2**20
    torch.npu.reset_peak_memory_stats(args.device)
    workload_memory_sampler = PeakNPUMemorySampler([device_id]).start()
    scan_module = None
    if args.ascendc_scan:
        scan_module = load_ascendc_prefill_scan(args.cpp_source)
    elif args.chunk_scan:
        chunk_extension = load_ascendc_prefill_scan(args.cpp_source)
        chunk_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.chunk_compute_dtype]
        scan_module = TorchChunkScanModule(
            chunk_size=args.chunk_size,
            compute_dtype=chunk_dtype,
            inverse_module=chunk_extension,
            input_is_log_decay=True,
        )
    if scan_module is not None:
        scan_module.scan_row_blocks = args.scan_row_blocks
        scan_module.use_fused_shift_mix = args.fused_shift_mix
        scan_module.fused_shift_mix_rows_per_block = (
            args.fused_shift_mix_rows_per_block
        )
        scan_module.use_head_layer_norm = args.head_layer_norm
        scan_module.use_fused_add_layer_norm = args.fused_add_layer_norm
        scan_module.use_dense_chunk_prefix = args.dense_chunk_prefix
        scan_module.dense_prefix_algorithm = args.dense_prefix_algorithm
        scan_module.inverse_backend = args.chunk_inverse_backend
        scan_module.inverse_base_size = args.chunk_inverse_base_size
    correctness_ids = (
        torch.arange(
            args.correctness_length, device=args.device, dtype=torch.long
        ).unsqueeze(0)
        + torch.arange(
            args.batch_size, device=args.device, dtype=torch.long
        ).unsqueeze(1)
        * 997
    ).remainder(eng.vocab_size)
    if args.batch_size == 1 and eng.mod is not None:
        correctness_reference = "token_major_decode"
        reference_logits, reference_cache = token_major_prefill(
            eng, correctness_ids, args.device
        )
    else:
        # Batched scan validation uses the identical layer-major projection
        # path with the recurrence executed in pure PyTorch.
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
    decode_correctness = None
    if eng.mod is not None:
        decode_token = (
            torch.arange(args.batch_size, device=args.device, dtype=torch.long)
            + 1234
        ).remainder(eng.vocab_size)
        reference_decode_cache = tuple(value.clone() for value in candidate_cache)
        candidate_decode_cache = tuple(value.clone() for value in candidate_cache)
        reference_decode_logits, reference_decode_cache = layer_major_prefill(
            eng,
            decode_token.unsqueeze(1),
            args.device,
            None,
            initial_cache=reference_decode_cache,
        )
        candidate_decode_logits, candidate_decode_cache = _full_decode_step(
            eng,
            decode_token,
            candidate_decode_cache,
        )
        torch.npu.synchronize()
        decode_logits_cosine = F.cosine_similarity(
            reference_decode_logits.float(), candidate_decode_logits.float()
        ).min().item()
        decode_logits_max_abs = (
            reference_decode_logits.float() - candidate_decode_logits.float()
        ).abs().max().item()
        decode_state_max_abs = max(
            (reference.float() - candidate.float()).abs().max().item()
            for reference, candidate in zip(
                reference_decode_cache, candidate_decode_cache
            )
        )
        decode_greedy_match = bool(
            torch.equal(
                reference_decode_logits.argmax(-1),
                candidate_decode_logits.argmax(-1),
            )
        )
        decode_correctness = {
            "reference": "layer_major_torch_scan",
            "greedy_match": decode_greedy_match,
            "logits_cosine": decode_logits_cosine,
            "logits_max_abs": decode_logits_max_abs,
            "state_max_abs": decode_state_max_abs,
        }
        print(
            "decode correctness greedy=%s logits_cosine=%.9f "
            "logits_max_abs=%.6g state_max_abs=%.6g"
            % (
                str(decode_greedy_match).lower(),
                decode_logits_cosine,
                decode_logits_max_abs,
                decode_state_max_abs,
            ),
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
    stage_latency_ms = None
    if args.stage_probe:
        stage_events = {}
        layer_major_prefill(
            eng,
            prompt_ids,
            args.device,
            scan_module,
            stage_events=stage_events,
        )
        torch.npu.synchronize()
        stage_latency_ms = _summarize_stage_events(stage_events)
    profile_table = None
    if args.torch_profile_output:
        profile_table = _run_torch_profile(
            eng,
            prompt_ids,
            args.device,
            scan_module,
            args.torch_profile_output,
        )
        print(profile_table, flush=True)
    decode = _time_decode(
        eng,
        prompt_ids,
        args.device,
        scan_module,
        warmup=args.decode_warmup,
        steps=args.decode_length,
        use_npu_graph=args.decode_npu_graph,
    )
    peak_memory_mib = workload_memory_sampler.stop()
    torch_peak_memory_mib = torch.npu.max_memory_allocated(args.device) / 2**20
    result = {
        "benchmark": "rwkv7_pth_prefill_npu",
        "scan_backend": (
            "ascendc" if args.ascendc_scan else "torch_chunk" if args.chunk_scan else "torch"
        ),
        "scan_row_blocks": args.scan_row_blocks if args.ascendc_scan else None,
        "chunk_size": args.chunk_size if args.chunk_scan else None,
        "chunk_compute_dtype": args.chunk_compute_dtype if args.chunk_scan else None,
        "optimizations": {
            "fused_shift_mix": args.fused_shift_mix,
            "fused_shift_mix_rows_per_block": (
                args.fused_shift_mix_rows_per_block
                if args.fused_shift_mix
                else None
            ),
            "head_layer_norm": args.head_layer_norm,
            "fused_add_layer_norm": args.fused_add_layer_norm,
            "dense_chunk_prefix": args.dense_chunk_prefix,
            "dense_prefix_algorithm": (
                args.dense_prefix_algorithm
                if args.dense_chunk_prefix
                else None
            ),
            "chunk_inverse_backend": (
                args.chunk_inverse_backend if args.chunk_scan else None
            ),
            "chunk_inverse_base_size": (
                args.chunk_inverse_base_size
                if args.chunk_scan
                and args.chunk_inverse_backend == "native_blocked"
                else None
            ),
            "native_bf16_factor_io": bool(
                args.chunk_scan
                and args.chunk_compute_dtype == "bf16"
                and getattr(scan_module, "inverse_module", None) is not None
                and hasattr(
                    scan_module.inverse_module,
                    "rwkv7_chunk_inverse_bf16_io",
                )
            ),
        },
        "loader_profile": args.loader_profile,
        "dtype": "fp16",
        "decode_length": args.decode_length,
        "loaded_memory_mib": loaded_memory_mib,
        "load_peak_memory_mib": load_peak_memory_mib,
        "packed_tensor_bytes": eng.packed_tensor_bytes,
        "resident_tensor_bytes": eng.resident_tensor_bytes,
        "engine_version": "rwkv7_ascend_pth_v2",
        **checkpoint_metadata(
            args.model_pth,
            revision=args.checkpoint_revision,
            sha256=args.checkpoint_sha256,
        ),
        **collect_cann_metadata(),
        **collect_npu_metadata(torch, torch_npu, args.device),
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
        "decode_correctness": decode_correctness,
        "layer_major_latency_ms": layer_ms,
        "layer_major_latency_ms_samples": layer_samples,
        "stage_latency_ms": stage_latency_ms,
        "torch_profile_output": args.torch_profile_output,
        "torch_profile_table": profile_table,
        "layer_major_tokens_per_second": (
            args.batch_size * args.prompt_length * 1000.0 / layer_ms
        ),
        "decode_backend": decode["backend"],
        "decode_graph_correctness": decode["graph_correctness"],
        "decode_latency_ms_mean": decode["latency_ms_mean"],
        "decode_tokens_per_second": decode["tokens_per_second"],
        "peak_memory_mib": peak_memory_mib,
        "peak_memory_scope": workload_memory_sampler.scope,
        "peak_memory_phase": "correctness_prefill_decode",
        "torch_peak_memory_mib": torch_peak_memory_mib,
        "memory_sampler_errors": workload_memory_sampler.errors,
    }
    print(json.dumps(result), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
