"""RWKV-7 configuration and SGLang recurrent-state layout.

This module deliberately lives outside SGLang.  It is registered through
SGLang's public external-model and linear-attention registries, so an SGLang
upgrade does not overwrite the integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import BaseLinearStateParams, Mamba2StateDType
from sglang.srt.distributed.utils import divide


@dataclass(kw_only=True, frozen=True)
class Rwkv7StateShape:
    """Two width-one token shifts plus the fp32 WKV matrix per head."""

    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]
    hidden_size: int
    num_heads: int
    head_dim: int

    @staticmethod
    def create(
        *, tp_world_size: int, hidden_size: int, num_heads: int, head_dim: int
    ) -> "Rwkv7StateShape":
        if hidden_size % tp_world_size:
            raise ValueError("hidden_size must be divisible by tensor parallel size")
        if num_heads % tp_world_size:
            raise ValueError("num_heads must be divisible by tensor parallel size")
        shift = (divide(hidden_size, tp_world_size), 1)
        return Rwkv7StateShape(
            conv=[shift, shift],
            temporal=(divide(num_heads, tp_world_size), head_dim, head_dim),
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )


@dataclass(kw_only=True, frozen=True)
class Rwkv7CacheParams(BaseLinearStateParams):
    shape: Rwkv7StateShape


class Rwkv7Config(PretrainedConfig):
    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=768,
        num_hidden_layers=12,
        head_dim=64,
        num_heads=None,
        decay_low_rank_dim=64,
        a_low_rank_dim=64,
        v_low_rank_dim=32,
        gate_low_rank_dim=128,
        intermediate_size=3072,
        hidden_ratio=4.0,
        hidden_act="sqrelu",
        norm_eps=1e-5,
        norm_bias=True,
        norm_first=True,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        attn=None,
        attn_mode="chunk",
        bos_token_id=0,
        eos_token_id=0,
        use_cache=True,
        **kwargs,
    ):
        if num_heads is None:
            num_heads = hidden_size // head_dim
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"hidden_size={hidden_size} != num_heads*head_dim="
                f"{num_heads * head_dim}"
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads
        self.decay_low_rank_dim = decay_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.intermediate_size = intermediate_size
        self.hidden_ratio = hidden_ratio
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.norm_first = norm_first
        self.max_position_embeddings = max_position_embeddings
        self.attn = attn
        self.attn_mode = attn_mode
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return ["linear_attention"] * self.num_hidden_layers

    @property
    def linear_layer_ids(self):
        return list(range(self.num_hidden_layers))

    @property
    def full_attention_layer_ids(self):
        return []

    @property
    def mamba2_cache_params(self) -> Rwkv7CacheParams:
        # Attention TP is the relevant sharding domain.  Keep a safe TP=1
        # fallback for config inspection before distributed initialization.
        try:
            from sglang.srt.distributed.parallel_state import get_attn_tp_group

            tp_world_size = get_attn_tp_group().world_size
        except (AssertionError, AttributeError, RuntimeError):
            tp_world_size = 1
        return Rwkv7CacheParams(
            shape=Rwkv7StateShape.create(
                tp_world_size=tp_world_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            ),
            layers=self.linear_layer_ids,
            # Both shift history and WKV state are fp32.  Reducing either to
            # bf16 changes greedy tokens on real checkpoints.
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        )
