#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-/data/venvs/sglang}
MODEL=${1:?usage: serve_production.sh MODEL_PATH [extra sglang args...]}
shift
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_rwkv7_ascend.models
exec "$VENV/bin/python" -m sglang_rwkv7_ascend.serve \
  --model-path "$MODEL" --device npu --dtype bfloat16 \
  --trust-remote-code --host 0.0.0.0 --port 30000 "$@"
