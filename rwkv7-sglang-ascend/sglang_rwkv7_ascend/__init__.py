"""Installable SGLang external-model plugin for RWKV-7 on Ascend."""

from __future__ import annotations

import os

__version__ = "0.2.0"
_REGISTERED = False


def register() -> None:
    """Register config, model metadata and the all-linear no-op backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig
    from .configuration_rwkv7 import Rwkv7Config

    try:
        AutoConfig.register(Rwkv7Config.model_type, Rwkv7Config, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register(Rwkv7Config.model_type, Rwkv7Config)
        except ValueError:
            pass

    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        register_linear_attn_model,
    )

    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=Rwkv7Config,
            backend_class_name="sglang_rwkv7_ascend.backend.Rwkv7AttnBackend",
            arch_names=["Rwkv7ForCausalLM", "RWKV7ForCausalLM"],
            uses_mamba_radix_cache=False,
            support_mamba_cache=True,
        )
    )

    # Prevent recursion if importing ModelRunner initializes ModelRegistry.
    _REGISTERED = True
    os.environ.setdefault(
        "SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_rwkv7_ascend.models"
    )
    from . import backend as _backend  # noqa: F401


def __getattr__(name):
    if name == "Rwkv7Config":
        from .configuration_rwkv7 import Rwkv7Config
        return Rwkv7Config
    raise AttributeError(name)


__all__ = ["Rwkv7Config", "register"]
