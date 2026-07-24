#!/usr/bin/env bash
# Install from the pinned, pre-populated source trees.  This script never
# fetches an unpinned branch and never overwrites SGLang source with an overlay.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/versions.env"
SGLANG_ROOT=${SGLANG_ROOT:-/data/work/sglang-upstream}
KERNEL_ROOT=${KERNEL_ROOT:-$SGLANG_ROOT/third_party/sgl-kernel-npu}
BUILD_SGL_KERNEL_NPU=${BUILD_SGL_KERNEL_NPU:-0}
VENV=${VENV:-/data/venvs/sglang}
PYTHON_BIN=${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}
MIRROR=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_ROOT=$(readlink -f /usr/local/Ascend/ascend-toolkit/latest)
[[ "$(basename "$CANN_ROOT")" == "cann-$CANN_VERSION" ]] || {
  echo "CANN version mismatch: required cann-$CANN_VERSION, got $CANN_ROOT"; exit 2
}

[[ -d "$SGLANG_ROOT/.git" ]] || { echo "missing SGLang checkout: $SGLANG_ROOT"; exit 2; }
[[ "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" == "$SGLANG_COMMIT" ]] || {
  echo "SGLang commit mismatch (required $SGLANG_COMMIT)"; exit 2;
}

# One small upstream fix is required for registered all-linear models: without
# it DefaultPoolConfigurator divides by a zero full-attention KV cell size.
PATCH="$HERE/patches/sglang-all-linear-pool.patch"
if git -C "$SGLANG_ROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SGLANG_ROOT" apply "$PATCH"
elif git -C "$SGLANG_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "all-linear pool patch already applied"
else
  echo "all-linear pool patch does not match pinned SGLang source"; exit 2
fi

# A custom pure-torch linear backend on NPU must be selected before importing
# SGLang's built-in Ascend Mamba backend (which has a hard sgl_kernel_npu
# import).  This keeps Atlas A2/910B usable with CANN 8.5.
PATCH="$HERE/patches/sglang-external-linear-pure-torch-npu.patch"
if git -C "$SGLANG_ROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SGLANG_ROOT" apply "$PATCH"
elif git -C "$SGLANG_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "external pure-torch NPU backend patch already applied"
else
  echo "external pure-torch NPU backend patch does not match pinned SGLang source"; exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi
PIP="$VENV/bin/python -m pip"
$PIP install -i "$MIRROR" --upgrade pip setuptools wheel
$PIP install -i "$MIRROR" cmake ninja pybind11 pyyaml numpy packaging \
  "transformers==$TRANSFORMERS_VERSION" pytest aiohttp dill sniffio distro \
  jiter docstring-parser loguru 'protobuf<7'
# Install the exact RWKV runtime/import closure without asking pip to resolve
# SGLang's generic CUDA-oriented dependency graph. In particular, current
# torchvision/timm resolution would otherwise replace the image's supported
# torch 2.9 build with an unrelated CUDA torch build.
$PIP install -i "$MIRROR" --no-deps -r "$HERE/requirements-910b3-runtime.txt"

# torch and torch_npu come from the Ascend base image.  Letting a generic pip
# resolver replace either one can silently install a CUDA build, so fail closed.
"$VENV/bin/python" - <<PY
import torch, torch_npu
assert torch.__version__.split('+')[0] == "$TORCH_VERSION", torch.__version__
assert torch_npu.__version__.split('+')[0] == "$TORCH_NPU_VERSION", torch_npu.__version__
assert torch_npu.npu.get_device_name(0) == "$NPU_DEVICE_NAME", torch_npu.npu.get_device_name(0)
PY

# Optional only: current upstream sgl-kernel-npu releases primarily target
# Atlas A3 and the AIV link fails on the deployed Atlas A2/CANN 8.5 toolchain.
# RWKV does not need it; enable this explicitly only on a supported image.
if [[ "$BUILD_SGL_KERNEL_NPU" == 1 ]]; then
  [[ -d "$KERNEL_ROOT/.git" ]] || { echo "missing sgl-kernel-npu checkout: $KERNEL_ROOT"; exit 2; }
  [[ "$(git -C "$KERNEL_ROOT" rev-parse HEAD)" == "$SGL_KERNEL_NPU_COMMIT" ]] || {
    echo "sgl-kernel-npu commit mismatch (required $SGL_KERNEL_NPU_COMMIT)"; exit 2;
  }
  if ! "$VENV/bin/python" -c 'import sgl_kernel_npu' >/dev/null 2>&1; then
    (cd "$KERNEL_ROOT" && PATH="$VENV/bin:$PATH" bash build.sh -a kernels Ascend910B1)
    $PIP install "$KERNEL_ROOT"/output/sgl_kernel_npu*.whl
  fi
fi

# Keep upstream's canonical pyproject untouched; create a temporary build view
# inside the allowed venv directory.
SGL_BUILD="$VENV/.sglang-npu-build"
mkdir -p "$SGL_BUILD"
cp "$SGLANG_ROOT/python/pyproject_npu.toml" "$SGL_BUILD/pyproject.toml"
ln -sfn "$SGLANG_ROOT/python/sglang" "$SGL_BUILD/sglang"
ln -sfn "$SGLANG_ROOT/python/README.md" "$SGL_BUILD/README.md"
ln -sfn "$SGLANG_ROOT/LICENSE" "$SGL_BUILD/LICENSE"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 $PIP install -i "$MIRROR" --no-deps -e "$SGL_BUILD"
$PIP install -i "$MIRROR" --no-deps -e "$HERE"

"$VENV/bin/python" "$HERE/scripts/verify_install.py"
