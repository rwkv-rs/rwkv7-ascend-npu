#!/bin/bash
# Install sglang into venv-29 (torch 2.9.0) for bf16 serving.
# Assumes rebuild_sgl_kernel_npu_29.sh already put sgl_kernel_npu (torch-2.9.0 build) in venv-29.
# /data/sglang source is shared (editable); the RWKV-7 integration overlay+wiring
# is already applied there. This just installs sglang + tames the deps to 2.9.0.
set -ex
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
VENV=/data/rwkv7-sglang-venv-29
source $VENV/bin/activate
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

cd /data/sglang
# sglang editable install (pulls CUDA-oriented deps; we force them back below)
pip install --no-cache-dir $MIRROR -e "python[all_npu]"

# Tame deps to the torch_npu-2.9.0-matched set (sglang pulls torch 2.12 / tv 0.27 / triton 3.7).
pip install --force-reinstall --no-deps $MIRROR torch==2.9.0 torchvision==0.24.0
pip uninstall -y triton triton-ascend >/dev/null 2>&1 || true
pip install --no-deps $MIRROR triton-ascend
pip install --no-deps $MIRROR opencv-python-headless

python -c "import sglang, sgl_kernel_npu, torch, torch_npu; print('venv-29 sglang OK torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'sgl_kernel_npu OK')"
echo "=== venv-29 sglang install DONE ==="
