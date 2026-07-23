#!/usr/bin/env bash
# Install from the pinned, pre-populated source trees.  This script never
# fetches an unpinned branch and never overwrites SGLang source with an overlay.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/versions.env"
SGLANG_ROOT=${SGLANG_ROOT:-/data/work/sglang-upstream}
KERNEL_ROOT=${KERNEL_ROOT:-$SGLANG_ROOT/third_party/sgl-kernel-npu}
VENV=${VENV:-/data/venvs/sglang}
PYTHON_BIN=${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}
MIRROR=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

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
[[ -d "$KERNEL_ROOT/.git" ]] || { echo "missing sgl-kernel-npu checkout: $KERNEL_ROOT"; exit 2; }
[[ "$(git -C "$KERNEL_ROOT" rev-parse HEAD)" == "$SGL_KERNEL_NPU_COMMIT" ]] || {
  echo "sgl-kernel-npu commit mismatch (required $SGL_KERNEL_NPU_COMMIT)"; exit 2;
}

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi
PIP="$VENV/bin/python -m pip"
$PIP install -i "$MIRROR" --upgrade pip setuptools wheel
$PIP install -i "$MIRROR" "torch==$TORCH_VERSION" "torch_npu==$TORCH_NPU_VERSION"

# Build/install the NPU kernel wheel only when the pinned tree has no wheel.
if ! "$VENV/bin/python" -c 'import sgl_kernel_npu' >/dev/null 2>&1; then
  (cd "$KERNEL_ROOT" && PATH="$VENV/bin:$PATH" bash build.sh)
  $PIP install "$KERNEL_ROOT"/output/sgl_kernel_npu*.whl
fi

# Keep upstream's canonical pyproject untouched; create a temporary build view
# inside the allowed venv directory.
SGL_BUILD="$VENV/.sglang-npu-build"
mkdir -p "$SGL_BUILD"
cp "$SGLANG_ROOT/python/pyproject_npu.toml" "$SGL_BUILD/pyproject.toml"
ln -sfn "$SGLANG_ROOT/python/sglang" "$SGL_BUILD/sglang"
ln -sfn "$SGLANG_ROOT/python/README.md" "$SGL_BUILD/README.md"
ln -sfn "$SGLANG_ROOT/LICENSE" "$SGL_BUILD/LICENSE"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 $PIP install -i "$MIRROR" -e "$SGL_BUILD"
$PIP install -i "$MIRROR" -e "$HERE[test]"

"$VENV/bin/python" "$HERE/scripts/verify_install.py"
