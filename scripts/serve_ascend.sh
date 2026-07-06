#!/bin/bash
# Launch the RWKV-7 sglang server on Ascend 910B3 (P2 serving milestone).
# Usage: bash scripts/serve_ascend.sh [model-path]
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MODEL=${1:-/data/rwkv7-models/rwkv7-0.4b-world-fla}
REPO=/data/rwkv7-sglang-ascend

# PYTHONPATH=<repo> so the backend's `from ascend_port.wkv import wkv_recurrent`
# resolves. --disable-cuda-graph until the decode-graph capture (conv-state shape
# in capture mode) is fixed; eager serving is correct, just slower.
cd "$REPO"
PYTHONPATH="$REPO" exec /data/rwkv7-sglang-venv-28/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --attention-backend ascend \
    --device npu \
    --host 0.0.0.0 --port 30000 \
    --dtype float32 \
    --trust-remote-code \
    --disable-cuda-graph \
    --log-level info
