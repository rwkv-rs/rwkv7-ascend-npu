#!/bin/bash
# Install SGLang (Ascend NPU build) on the 910B3. Fresh py3.11 venv, torch 2.8.0.
# Recipe: sgl-project/sglang docs/platforms/ascend/ascend_npu.md + memory
# sglang-ascend-install-recipe. Long-running (sgl-kernel-npu compiles from source).
set -ex

[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh

VENV=/data/rwkv7-sglang-venv-28
PYBIN=$VENV/bin/python
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

if [ ! -x "$PYBIN" ]; then
  /usr/local/python3.11.14/bin/python3.11 -m venv "$VENV"
fi
$PYBIN -m pip install --upgrade pip $MIRROR || true

echo "=== torch 2.8.0 + torch_npu 2.8.0.post2 ==="
$VENV/bin/pip install --no-cache-dir $MIRROR "torch==2.8.0"
$VENV/bin/pip install --no-cache-dir $MIRROR "torch_npu==2.8.0.post2"
$PYBIN -c "import torch,torch_npu;print('torch',torch.__version__,'torch_npu',torch_npu.__version__,'npu',torch.npu.is_available())"

echo "=== triton-ascend + build deps ==="
$VENV/bin/pip install --no-cache-dir $MIRROR "triton-ascend"
$VENV/bin/pip install --no-cache-dir $MIRROR "setuptools<80" pybind11

echo "=== sgl-kernel-npu (build from source) ==="
if [ ! -d /data/sgl-kernel-npu ]; then
  git clone --recursive https://github.com/sgl-project/sgl-kernel-npu.git /data/sgl-kernel-npu
fi
cd /data/sgl-kernel-npu
bash build.sh
$VENV/bin/pip install --no-cache-dir output/sgl_kernel_npu*.whl

echo "=== sglang (source, NPU toml) ==="
if [ ! -d /data/sglang ]; then
  git clone https://github.com/sgl-project/sglang.git /data/sglang
fi
cd /data/sglang
git fetch --tags --quiet
TAG=$(git tag | grep -E '^v0\.5\.' | sort -V | tail -1)
echo "sglang checkout: $TAG"
git checkout "$TAG"
mv -f python/pyproject_npu.toml python/pyproject.toml
$VENV/bin/pip install --no-cache-dir $MIRROR -e "python[all_npu]"

echo "=== verify ==="
$PYBIN -c "import sglang; import sgl_kernel_npu; print('sglang', sglang.__version__, 'sgl_kernel_npu OK')"
echo "=== DONE ==="
