#!/bin/bash
# Resume the sglang Ascend install from the sgl-kernel-npu build step.
# Fix: build.sh/cmake call `python3`; we activate the venv so python3 has torch.
set -ex
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
VENV=/data/rwkv7-sglang-venv-28
source $VENV/bin/activate
export PYTHON=$VENV/bin/python

# AscendC toolchain (bishengir/tbe) + TE need these python deps; the venv
# doesn't see CANN's site-packages, so install them here.
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    decorator attrs cloudpickle ml_dtypes psutil scipy tornado sympy

echo "=== build sgl-kernel-npu (venv active) ==="
cd /data/sgl-kernel-npu
bash build.sh
pip install --no-cache-dir output/sgl_kernel_npu*.whl

echo "=== sglang source install ==="
cd /data/sglang
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e "python[all_npu]"

echo "=== verify ==="
python -c "import sglang, sgl_kernel_npu; print('sglang', sglang.__version__, 'sgl_kernel_npu OK')"
echo "=== DONE ==="
