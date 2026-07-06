# Licensed under the Apache License, Version 2.0 (the "License").
"""RWKV-7 (Goose) linear-attention backend for sglang on Ascend NPU.

Ascend port of Hakureirm/rwkv-sglang's CUDA backend (reference/hakureirm/
rwkv7_backend.py). Differences vs the CUDA original:

  * Base class is ``AscendMambaAttnBackendBase`` (NPU metadata/cuda-graph plumbing)
    instead of the CUDA ``MambaAttnBackendBase``.
  * The WKV recurrence calls the pure-torch ``ascend_port.wkv.wkv_recurrent``
    (M1a/M1c token-exact on 910B3) instead of the FLA-free triton kernel. Our
    kernel has no in-place indexed-state mode, so both decode and extend use the
    gather / compute / scatter pattern on ``temporal``.
  * The optional CUDA fused shift+lerp glue (RWKV_FUSED_GLUE) is dropped -- it is
    NVIDIA-only and never engaged on NPU.

The model calls ``token_shift(...)`` and ``recurrence(...)`` directly on this
backend instance (obtained via ``forward_batch.attn_backend.linear_attn_backend``),
bypassing HybridLinearAttnBackend.forward. ``init_forward_metadata`` still runs
through HybridLinearAttnBackend, populating ``self.forward_metadata``
(query_start_loc + mamba_cache_indices) normally. State lives in the MambaPool:
conv[0]/conv[1] = the two width-2 token-shift states (attn/ffn); temporal = the
WKV recurrent state S [size+1, H, K, V] fp32.
"""
from typing import Optional

import torch

from ascend_port.wkv import wkv_recurrent
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
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


class Rwkv7AttnBackend(AscendMambaAttnBackendBase):
    """Linear-attention backend for RWKV-7 on Ascend NPU.

    Both decode and extend run through the pure-torch ``ascend_port.wkv.
    wkv_recurrent`` (greedy-exact vs the numpy oracle, ~1e-6). scale=1.0 matches
    the oracle (no r-scaling before GroupNorm).
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # conv[0] shape = (num_layers, size+1, hidden, 1)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        self.scale = 1.0

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
