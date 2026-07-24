# Licensed under the Apache License, Version 2.0 (the "License").
"""RWKV-7 (Goose) linear-attention backend for sglang on Ascend NPU.

Ascend port of Hakureirm/rwkv-sglang's CUDA backend (reference/hakureirm/
rwkv7_backend.py). Differences vs the CUDA original:

  * A small scheduler-metadata backend replaces SGLang's built-in Mamba class.
    It deliberately avoids importing ``sgl_kernel_npu``: current upstream
    kernel wheels target Atlas A3 and are not a dependency on Atlas A2/910B.
  * The WKV recurrence calls ``kernels.wkv.wkv_recurrent``
    (M1a/M1c token-exact on 910B3) instead of the FLA-free triton kernel. Our
    kernel has no in-place indexed-state mode, so both decode and extend use the
    gather / compute / scatter pattern on ``temporal``.
  * The optional CUDA fused shift+lerp glue (RWKV_FUSED_GLUE) is dropped -- it is
    NVIDIA-only and never engaged on NPU.

The model calls ``token_shift(...)`` and ``recurrence(...)`` directly on this
backend instance (obtained via ``forward_batch.attn_backend.linear_attn_backend``),
bypassing wrapper ``forward``. ``init_forward_metadata`` runs through the
lightweight wrapper, populating ``self.forward_metadata``
(query_start_loc + mamba_cache_indices) normally. State lives in the MambaPool:
conv[0]/conv[1] = the two width-2 token-shift states (attn/ffn); temporal = the
WKV recurrent state S [size+1, H, K, V] fp32.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Optional

import torch

# Prefer the fused triton-ascend WKV kernel (much faster on NPU; matches the
# pure-torch path to ~1e-6). Fall back to pure-torch if triton-ascend is absent.
try:
    from sglang_rwkv7_ascend.kernels.wkv_triton import wkv_recurrent
except Exception:
    from sglang_rwkv7_ascend.kernels.wkv import wkv_recurrent
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.attention_registry import register_attention_backend
if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class Rwkv7ForwardMetadata:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor


def _acceptance_trace(forward_batch, metadata: Rwkv7ForwardMetadata) -> None:
    """Append scheduler/state evidence only when explicitly enabled.

    This intentionally performs device-to-host copies and must never be enabled
    for normal serving or performance measurements.
    """
    path = os.environ.get("RWKV_SGLANG_ACCEPTANCE_TRACE")
    if not path:
        return

    def values(tensor):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            return list(tensor)
        return tensor.detach().cpu().reshape(-1).tolist()

    mode = forward_batch.forward_mode
    event = {
        "kind": "forward",
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "mode": getattr(mode, "name", str(mode)),
        "batch_size": int(forward_batch.batch_size),
        "real_batch_size": int(
            getattr(forward_batch, "_original_batch_size", None)
            or forward_batch.batch_size
        ),
        "request_pool_indices": values(forward_batch.req_pool_indices),
        "state_slot_ids": values(metadata.mamba_cache_indices),
        "query_start_loc": values(metadata.query_start_loc),
        "extend_prefix_lens": values(
            getattr(forward_batch, "extend_prefix_lens", None)
        ),
        "extend_seq_lens": values(
            getattr(forward_batch, "extend_seq_lens", None)
        ),
    }
    trace = Path(path)
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")


class Rwkv7NoOpFullAttnBackend(AttentionBackend):
    """Trivial full-attention backend for the all-linear RWKV-7 case.

    RWKV-7 has ZERO full-attention layers, so HybridLinearAttnBackend never
    routes any layer here, but it still calls init_forward_metadata + cuda-graph
    hooks on the full backend each step. Real backends reject the empty KV pool
    / fp32 planning, so we substitute this no-op. (Unchanged CUDA -> NPU.)
    """

    token_to_kv_pool = None
    req_to_token_pool = None
    needs_cpu_seq_lens = False

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        if model_runner is not None:
            self.req_to_token_pool = model_runner.req_to_token_pool
            self.token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
            self.max_context_len = model_runner.model_config.context_len
            self.data_type = getattr(model_runner, "dtype", torch.float32)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        pass

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_cpu_graph_state(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        pass

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")

    def forward_mixed(self, *args, **kwargs):
        raise NotImplementedError("RWKV-7 has no full-attention layers.")


@register_attention_backend("rwkv7_ascend")
@register_attention_backend("ascend")
def create_rwkv7_noop_backend(runner):
    """Full-attention half of SGLang's hybrid wrapper (RWKV has none)."""

    return Rwkv7NoOpFullAttnBackend(runner)


class Rwkv7HybridAttnBackend(AttentionBackend):
    """Lightweight all-linear wrapper with no sgl_kernel_npu imports."""

    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
        if full_attn_layers:
            raise ValueError("RWKV-7 must not contain full-attention layers")
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.max_context_len = full_attn_backend.max_context_len
        self.needs_cpu_seq_lens = False

    @property
    def data_type(self):
        return self.full_attn_backend.data_type

    def init_forward_metadata(self, forward_batch):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata_in_graph(forward_batch)

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backend_list:
            backend.on_after_cuda_graph_warmup()

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for backend in self.attn_backend_list:
            backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs, max_num_tokens):
        for backend in self.attn_backend_list:
            backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata_capture_cpu_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        return self.linear_attn_backend.forward_decode(*args, **kwargs)

    def forward_extend(self, *args, **kwargs):
        return self.linear_attn_backend.forward_extend(*args, **kwargs)


class Rwkv7AttnBackend(AttentionBackend):
    """Linear-attention backend for RWKV-7 on Ascend NPU.

    Both decode and extend run through the pure-torch ``ascend_port.wkv.
    wkv_recurrent`` (greedy-exact vs the numpy oracle, ~1e-6). scale=1.0 matches
    the oracle (no r-scaling before GroupNorm).
    """

    needs_cpu_seq_lens = False

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.forward_metadata: Optional[Rwkv7ForwardMetadata] = None
        # conv[0] shape = (num_layers, size+1, hidden, 1)
        # MambaAttnBackendBase interprets the last dimension as conv history.
        # Ascend physically stores conv as [slot, history, hidden], so exposing
        # the physical shape here would mistake ``hidden`` for history.
        self.conv_states_shape = (model_runner.model_config.hidden_size, 1)
        self.scale = 1.0

    def _make_metadata(self, forward_batch: "ForwardBatch") -> Rwkv7ForwardMetadata:
        indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        translate = getattr(self.req_to_token_pool, "translate_mamba_indices", None)
        if translate is not None:
            indices = translate(indices)
        real_bs = getattr(forward_batch, "_original_batch_size", None)
        if real_bs is not None and real_bs < indices.shape[0]:
            indices = indices.clone()
            indices[real_bs:] = -1

        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            qsl = torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                raise NotImplementedError("RWKV-7 speculative draft extend is disabled")
            qsl = torch.empty((bs + 1,), dtype=torch.int32, device=self.device)
            qsl[:bs] = forward_batch.extend_start_loc
            qsl[bs] = (
                forward_batch.extend_start_loc[-1]
                + forward_batch.extend_seq_lens[-1]
            )
        else:
            raise ValueError(f"Unsupported RWKV-7 forward mode: {forward_batch.forward_mode}")
        metadata = Rwkv7ForwardMetadata(qsl, indices)
        _acceptance_trace(forward_batch, metadata)
        return metadata

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.forward_metadata = self._make_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: "ForwardBatch", in_capture: bool = False
    ):
        # Production launch disables graph capture.  Keeping this eager hook
        # makes scheduler metadata refresh correctly during ordinary replay.
        self.forward_metadata = self._make_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("RWKV model calls recurrence() directly")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("RWKV model calls recurrence() directly")

    # ---- token-shift (width-2 causal shift via the conv state) ----
    def token_shift(
        self,
        x: torch.Tensor,
        layer_id: int,
        conv_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return the previous-token hidden for each position in x.

        x: [num_tokens, hidden] (post attn_norm / ffn_norm). Updates the stored
        conv state with the last token of each sequence for the next step.
        Pure index logic -- device-agnostic, runs on NPU unchanged.
        """
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv = cache.conv[conv_idx]  # NPU pool: [size+1, 1, hidden]; CUDA: [size+1, hidden, 1]
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices
        # Layout-agnostic per-slot hidden-vector access: flatten the two non-slot
        # dims (one is the size-1 conv_kernel) to [hidden] for read; scatter-write
        # back by reshaping to the actual non-slot shape.
        slot_tail = tuple(conv.shape[1:])  # (1, hidden) on NPU, (hidden, 1) on CUDA

        def conv_read(idx):
            return conv[idx].reshape(idx.numel(), -1)  # [n, hidden]

        def conv_write(idx, val):  # val: [n, hidden] -> scatter write-back
            conv[idx] = val.reshape(val.shape[0], *slot_tail)

        if forward_batch.forward_mode.is_decode_or_idle():
            # one token per request: shifted = stored prev; store current.
            # Padded cuda-graph replay fills the tail of mamba_cache_indices with
            # PAD_SLOT_ID = -1; clamp to 0 (the pool's reserved never-allocated
            # slot) so -1 doesn't wrap to a live slot. Pad outputs are discarded.
            safe_idx = torch.clamp_min(cache_indices, 0)
            prev = conv_read(safe_idx).clone()  # [n, hidden]
            conv_write(safe_idx, x.to(conv.dtype))
            return prev.to(x.dtype)

        # extend (packed B=1, varlen via query_start_loc)
        qsl = md.query_start_loc.to(torch.long)
        starts = qsl[:-1]
        ends = qsl[1:]
        shifted = torch.empty_like(x)
        if x.shape[0] > 1:
            shifted[1:] = x[:-1]
        safe_idx = torch.clamp_min(cache_indices, 0)
        # Zero the shift state of sequences starting from scratch (prefix 0);
        # keep chunked-prefill / state-cache carry-ins (prefix > 0).
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_lens is not None:
            zero_target = torch.where(
                prefix_lens == 0, safe_idx, torch.zeros_like(safe_idx)
            )
            conv.index_fill_(0, zero_target.to(torch.long), 0.0)
        # first token of each seq reads stored prev (0 for fresh; carry-in otherwise)
        shifted[starts] = conv_read(safe_idx).to(x.dtype)
        # store last token of each seq for the next chunk / decode.
        conv_write(safe_idx, x[ends - 1].to(conv.dtype))
        return shifted

    # ---- fused shift+lerp glue (CUDA-only; stub to None so the model falls back
    # to token_shift + plain-torch lerp, which is the M1c-verified path). ----
    def try_fused_shift_lerp6(self, normed, layer_id, conv_idx, mix6, forward_batch):
        return None

    def try_fused_shift_lerp1(self, normed, layer_id, conv_idx, x_k, forward_batch):
        return None

    # ---- WKV recurrence (decode + extend -> ascend_port.wkv.wkv_recurrent) ----
    def recurrence(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """r,w,k,kk,a: [num_tokens, H, K]; v: [num_tokens, H, V]. -> [num_tokens, H, V].

        Ascend: our pure-torch wkv_recurrent has no in-place indexed-state mode,
        so we gather temporal[safe_idx] -> run -> scatter back (decode + extend).
        """
        cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        temporal = cache.temporal  # [size+1, H, K, V] fp32
        md = self.forward_metadata
        cache_indices = md.mamba_cache_indices
        safe_idx = torch.clamp_min(cache_indices, 0)

        if forward_batch.forward_mode.is_decode_or_idle():
            # [bs, H, *] -> [bs, 1, H, *]
            init_state = temporal[safe_idx].contiguous().float()  # [bs, H, K, V]
            o, final_state = wkv_recurrent(
                r.unsqueeze(1).contiguous(), w.unsqueeze(1).contiguous(),
                k.unsqueeze(1).contiguous(), v.unsqueeze(1).contiguous(),
                kk.unsqueeze(1).contiguous(), a.unsqueeze(1).contiguous(),
                scale=self.scale, initial_state=init_state, output_final_state=True,
            )
            temporal[safe_idx] = final_state.to(temporal.dtype)
            return o.squeeze(1)  # [bs, H, V]

        # extend: packed B=1, varlen via cu_seqlens. Zero fresh-seq temporal rows
        # before snapshotting (same pad/slot logic as token_shift).
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_lens is not None:
            zero_target = torch.where(
                prefix_lens == 0, safe_idx, torch.zeros_like(safe_idx)
            )
            temporal.index_fill_(0, zero_target.to(torch.long), 0.0)
        init_state = temporal[safe_idx].contiguous().float()  # [N, H, K, V]
        cu = md.query_start_loc.to(torch.int64)
        o, final_state = wkv_recurrent(
            r.unsqueeze(0).contiguous(), w.unsqueeze(0).contiguous(),
            k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous(),
            kk.unsqueeze(0).contiguous(), a.unsqueeze(0).contiguous(),
            scale=self.scale, initial_state=init_state, output_final_state=True,
            cu_seqlens=cu,
        )
        temporal[safe_idx] = final_state.to(temporal.dtype)
        return o.squeeze(0)  # [total_T, H, V]

    # The model calls token_shift/recurrence directly; these are not used.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Rwkv7AttnBackend.recurrence/token_shift are called directly by the model."
        )
