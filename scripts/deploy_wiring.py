#!/usr/bin/env python3
# coding=utf-8
"""Idempotently apply the RWKV-7 Ascend wiring edits to the installed sglang tree.

`patch`/`git apply` are too strict (context drift vs our v0.5.14 base). This script
does anchor-based idempotent insertions: each edit is skipped if its marker is
already present, applied if its anchor is found, or reported as MISS otherwise
(so misses can be hand-fixed). Run repeatedly; safe to re-run.
"""
import os, sys

ROOT = "/data/sglang/python/sglang/srt"
results = []


def edit(relpath, marker, anchor, insert_after_match, report_name):
    """Insert `insert_after_match` text right after the first `anchor` occurrence.
    Skip if `marker` already present; MISS if anchor not found."""
    path = os.path.join(ROOT, relpath)
    with open(path) as f:
        src = f.read()
    if marker in src:
        results.append((report_name, "skip")); return
    idx = src.find(anchor)
    if idx == -1:
        results.append((report_name, "MISS")); return
    end = idx + len(anchor)
    src = src[:end] + insert_after_match + src[end:]
    with open(path, "w") as f:
        f.write(src)
    results.append((report_name, "ok"))


def replace(relpath, marker, find, replacement, report_name):
    """Replace the first `find` occurrence with `replacement` (for edits that
    SUBSTITUTE a line, not append). Skip if `marker` present; MISS if not found."""
    path = os.path.join(ROOT, relpath)
    with open(path) as f:
        src = f.read()
    if marker in src:
        results.append((report_name, "skip")); return
    if find not in src:
        results.append((report_name, "MISS")); return
    src = src.replace(find, replacement, 1)
    with open(path, "w") as f:
        f.write(src)
    results.append((report_name, "ok"))


# 1. configs/__init__.py: import + __all__
edit("configs/__init__.py",
     "from sglang.srt.configs.rwkv7 import",
     "from sglang.srt.configs.qwen3_next import Qwen3NextConfig",
     "\nfrom sglang.srt.configs.rwkv7 import Rwkv7Config",
     "configs/__init__ import")
edit("configs/__init__.py",
     '"Rwkv7Config",',
     '"Qwen3NextConfig",',
     '\n    "Rwkv7Config",',
     "configs/__init__ __all__")

# 2. configs/mamba_utils.py: append the two classes (idempotent)
MW = os.path.join(ROOT, "configs/mamba_utils.py")
with open(MW) as f:
    mw = f.read()
if "class Rwkv7StateShape" in mw:
    results.append(("mamba_utils", "skip"))
else:
    classes = '''

@dataclass(kw_only=True, frozen=True)
class Rwkv7StateShape:
    """State shape for RWKV-7 (Goose): two width-2 token-shifts (attn/ffn) +
    the recurrent WKV state S = (num_heads, head_dim, head_dim), kept fp32."""

    conv: List[tuple[int, int]]
    temporal: tuple[int, int, int]
    hidden_size: int
    num_heads: int
    head_dim: int

    @staticmethod
    def create(*, tp_world_size, hidden_size, num_heads, head_dim) -> "Rwkv7StateShape":
        conv_state_shape = (divide(hidden_size, tp_world_size), 1)
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)
        return Rwkv7StateShape(
            conv=[conv_state_shape, conv_state_shape],
            temporal=temporal_state_shape,
            hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim,
        )


@dataclass(kw_only=True, frozen=True)
class Rwkv7CacheParams(BaseLinearStateParams):
    shape: Rwkv7StateShape
'''
    with open(MW, "a") as f:
        f.write(classes)
    results.append(("mamba_utils", "ok"))

# 3. attention_registry.py: import + routing
edit("layers/attention/attention_registry.py",
     "from sglang.srt.layers.attention.linear.rwkv7_backend import Rwkv7AttnBackend",
     "            LightningAttentionBackend,\n        )",
     "\n        from sglang.srt.layers.attention.linear.rwkv7_backend import Rwkv7AttnBackend",
     "attn_registry import")
edit("layers/attention/attention_registry.py",
     "Using hybrid linear attention backend for RWKV-7",
     "        elif runner.hybrid_lightning_config is not None:\n            linear_attn_backend = LightningAttentionBackend(runner)",
     '\n        elif runner.rwkv7_config is not None:\n            logger.info("Using hybrid linear attention backend for RWKV-7 models.")\n            linear_attn_backend = Rwkv7AttnBackend(runner)',
     "attn_registry routing")

# 4. model_runner.py: import + property + short-circuit + NoOp substitution
edit("model_executor/model_runner.py",
     "    Rwkv7Config,",
     "    Qwen3NextConfig,\n    ZayaConfig,",
     "\n    Rwkv7Config,",
     "model_runner import")
# fix: the import marker above is the import line itself; use a tighter marker
replace("model_executor/model_runner.py",
        "def rwkv7_config",
        "    @property\n    def hybrid_gdn_config(self):",
        "    @property\n    def rwkv7_config(self):\n        config = self.model_config.hf_config\n        if isinstance(config, Rwkv7Config):\n            return config\n        return None\n\n    @property\n    def hybrid_gdn_config(self):",
        "model_runner property")
edit("model_executor/model_runner.py",
     "or self.rwkv7_config",
     "            or self.hybrid_lightning_config",
     "\n            or self.rwkv7_config",
     "model_runner short-circuit")
replace("model_executor/model_runner.py",
        "Rwkv7NoOpFullAttnBackend",
        "        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)",
        '        if self.rwkv7_config is not None:\n            from sglang.srt.layers.attention.linear.rwkv7_backend import (\n                Rwkv7NoOpFullAttnBackend,\n            )\n            full_attention_backend = Rwkv7NoOpFullAttnBackend(self)\n        else:\n            full_attention_backend = ATTENTION_BACKENDS[backend_str](self)',
        "model_runner NoOp full-attn")

# 5. pool_configurator.py: cell_size==0
edit("model_executor/pool_configurator.py",
     "self._all_linear_token_cap",
     "        self._cell_size = self._compute_cell_size(mr, num_layers)",
     '\n        self._all_linear_token_cap = (\n            min(mr.server_args.max_mamba_cache_size * mr.model_config.context_len, 1 << 20)\n            if self._cell_size == 0 else None\n        )',
     "pool_configurator cell_size")
replace("model_executor/pool_configurator.py",
        "max_total_num_tokens = self._all_linear_token_cap",
        "        max_total_num_tokens = available_bytes // self._cell_size",
        "        if self._cell_size == 0:\n            max_total_num_tokens = self._all_linear_token_cap\n        else:\n            max_total_num_tokens = available_bytes // self._cell_size",
        "pool_configurator pool_sizes")

# 6. server_args.py: radix-cache disable for RWKV-7
edit("server_args.py",
     'Disabling radix cache for RWKV-7',
     '        elif model_arch in ["BailingMoeV2_5ForCausalLM"]:\n            self._handle_mamba_radix_cache(model_arch=model_arch)',
     '\n        elif model_arch in ["RWKV7ForCausalLM", "Rwkv7ForCausalLM"]:\n            if not self.disable_radix_cache:\n                logger.info("Disabling radix cache for RWKV-7 (recurrent state is not prefix-cacheable).")\n                self.disable_radix_cache = True',
     "server_args radix-cache")

# 7. hf_transformers/common.py: import + registry
edit("utils/hf_transformers/common.py",
     "from sglang.srt.configs import Rwkv7Config",
     "from transformers import PretrainedConfig",
     "\nfrom sglang.srt.configs import Rwkv7Config",
     "hf_common import")
edit("utils/hf_transformers/common.py",
     "Rwkv7Config,",
     "    for cls in [\n        ",
     "Rwkv7Config,\n        ",
     "hf_common registry")

print("=== wiring deploy results ===")
for name, status in results:
    print(f"  [{status:4s}] {name}")
misses = [n for n, s in results if s == "MISS"]
print(f"\n{sum(1 for _,s in results if s=='ok')} ok, {sum(1 for _,s in results if s=='skip')} skip, {len(misses)} MISS")
sys.exit(1 if misses else 0)
