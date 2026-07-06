#!/bin/bash
# Test whether torch_npu 2.9.0 (still CANN-8.5.0-compatible, newer op impls)
# fixes the aclnn bf16 norm failure (err 161002) seen on 2.8.0.post2.
# NEW venv -- does NOT touch the working /data/rwkv7-sglang-venv-28.
set -ex
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
VENV=/data/rwkv7-sglang-venv-29
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
[ -x "$VENV/bin/python" ] || /usr/local/python3.11.14/bin/python3.11 -m venv "$VENV"
$VENV/bin/pip install --no-cache-dir $MIRROR pyyaml numpy decorator attrs cloudpickle ml_dtypes psutil scipy tornado sympy
$VENV/bin/pip install --no-cache-dir $MIRROR torch==2.9.0
$VENV/bin/pip install --no-cache-dir $MIRROR torch_npu==2.9.0
$VENV/bin/python - <<'PYEOF'
import torch, torch_npu
print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "npu", torch.npu.is_available())
dev = "npu"
x = torch.randn(4, 32, dtype=torch.bfloat16, device=dev)
g = torch.nn.GroupNorm(4, 32).to(dev).to(torch.bfloat16)
try:
    y = g(x); print("BF16 GroupNorm OK:", y.shape, y.dtype)
except Exception as e:
    print("BF16 GroupNorm FAIL:", type(e).__name__, str(e)[:160])
ln = torch.nn.LayerNorm(32).to(dev).to(torch.bfloat16)
try:
    y = ln(x); print("BF16 LayerNorm OK:", y.shape, y.dtype)
except Exception as e:
    print("BF16 LayerNorm FAIL:", type(e).__name__, str(e)[:160])
print("=== bf16 norm test done ===")
PYEOF
