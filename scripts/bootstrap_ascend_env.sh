#!/bin/bash
# Bootstrap the Ascend env for the RWKV-7 sglang port.
# Version match (https://github.com/Ascend/pytorch matching table):
#   CANN 8.5.0  <->  torch_npu 2.7.1.post2  <->  torch 2.7.1  (Python 3.11)
# Uses a venv (no conda on this box) built from /usr/local/python3.11.14.
set -ex

[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh

BASE=/usr/local/python3.11.14/bin/python3.11
VENV=/data/rwkv7-sglang-venv
PY=$VENV/bin/python
PIP="$VENV/bin/pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple"

if [ ! -x "$PY" ]; then
  "$BASE" -m venv "$VENV"
  "$PY" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple || true
fi

echo "=== python ==="
$PY --version

echo "=== install torch 2.7.1 (aarch64) ==="
$PIP "torch==2.7.1"

echo "=== install torch-npu 2.7.1.post2 ==="
$PIP "torch-npu==2.7.1.post2"

echo "=== basic deps ==="
$PIP pyyaml setuptools numpy

echo "=== verify torch_npu sees the 910B3 ==="
$PY - <<'PYEOF'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available())
print("device_count", torch.npu.device_count())
if torch.npu.is_available():
    print("device0", torch.npu.get_device_name(0))
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    print("matmul ok:", (x @ y).sum().item())
PYEOF

echo "=== DONE ==="
