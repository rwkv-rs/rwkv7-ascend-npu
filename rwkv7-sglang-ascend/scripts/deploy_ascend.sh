#!/bin/bash
# Deploy the RWKV-7 Ascend integration into the installed sglang tree.
# Run AFTER scripts/install_sglang_ascend.sh (sglang + sgl-kernel-npu installed).
# Idempotent: safe to re-run. Launch sglang with PYTHONPATH=<repo> (ascend_port).
set -e
VENV=/data/rwkv7-sglang-venv-28
SGLANG=/data/sglang/python/sglang
REPO=/data/rwkv7-sglang-ascend
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# 0. Pin NPU-matched dep versions. sglang's install pulls newer CUDA-oriented
#    torch/torchvision/triton that break the torch_npu/sgl-kernel-npu ABI
#    (built against torch 2.8.0); force them back to the matched versions.
$VENV/bin/pip install --force-reinstall --no-deps $MIRROR torch==2.8.0 torchvision==0.23.0
$VENV/bin/pip uninstall -y triton triton-ascend >/dev/null 2>&1 || true
$VENV/bin/pip install --no-deps $MIRROR triton-ascend

# 1. Copy the overlay files into the sglang tree.
cp "$REPO"/ascend_port/sglang_overlay/sglang/srt/configs/rwkv7.py "$SGLANG/srt/configs/rwkv7.py"
cp "$REPO"/ascend_port/sglang_overlay/sglang/srt/models/rwkv7.py "$SGLANG/srt/models/rwkv7.py"
cp "$REPO"/ascend_port/sglang_overlay/sglang/srt/layers/attention/linear/rwkv7_backend.py \
   "$SGLANG/srt/layers/attention/linear/rwkv7_backend.py"

# 2. Apply the 7-file wiring edits (idempotent; see scripts/deploy_wiring.py).
cd /data/sglang && $VENV/bin/python "$REPO/scripts/deploy_wiring.py"

# 3. Stub triton.language.extra.cann. PyPI triton-ascend 3.2.0 is stripped (no
#    extra.cann.extension); sgl_kernel_npu's mamba/fla kernels reference it.
#    RWKV-7's pure-torch wkv path never calls those kernels, so a stub unblocks
#    the import chain. (For Mamba2/softmax models, build triton-ascend from source.)
CANN_DIR=$($VENV/bin/python -c "import triton,os;print(os.path.join(os.path.dirname(triton.__file__),'language','extra','cann'))")
mkdir -p "$CANN_DIR"
[ -f "$CANN_DIR/__init__.py" ] || echo "# cann extra stub" > "$CANN_DIR/__init__.py"
cat > "$CANN_DIR/extension.py" <<'PYEOF'
"""Stub for triton.language.extra.cann.extension (PyPI triton-ascend 3.2.0 is
stripped). Referenced by sgl_kernel_npu mamba/fla kernels the RWKV-7 Ascend path
never executes. Names exist so imports resolve; calling raises."""


def _stub(*a, **k):
    raise NotImplementedError("triton.language.extra.cann stubbed; build triton-ascend from source for Mamba2/softmax cann kernels.")


extract_slice = _stub
insert_slice = _stub
size = _stub
PYEOF

echo "=== Deploy done. Launch with: PYTHONPATH=$REPO python -m sglang.launch_server ... ==="
