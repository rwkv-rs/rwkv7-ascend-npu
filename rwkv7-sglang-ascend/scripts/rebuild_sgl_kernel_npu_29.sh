#!/bin/bash
# Rebuild sgl-kernel-npu against torch 2.9.0 (venv-29), for bf16 serving.
set -ex
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
VENV=/data/rwkv7-sglang-venv-29
source $VENV/bin/activate
# build deps the cmake/AscendC toolchain needs (pybind11 was missing -> cmake TORCH_DIR fail)
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11 "setuptools<80" decorator attrs cloudpickle ml_dtypes psutil scipy tornado sympy
cd /data/sgl-kernel-npu
rm -rf build output  # fresh build against torch 2.9.0 (was 2.8.0)
bash build.sh
pip install --no-cache-dir output/sgl_kernel_npu*.whl
python -c "import sgl_kernel_npu, torch, torch_npu; print('venv-29 OK torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'sgl_kernel_npu OK')"
echo "=== venv-29 sgl-kernel-npu DONE ==="
