#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-/data/venvs/sglang}
MODEL=${1:?usage: run_engine_acceptance.sh MODEL_PATH [JSON_OUTPUT]}
OUTPUT=${2:-$HERE/acceptance-engine.json}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$HERE"
set +e
"$VENV/bin/python" scripts/acceptance_engine.py \
  --model "$MODEL" --output "$OUTPUT" \
  --server-log "${OUTPUT%.json}.server.log"
rc=$?
sha256sum "$OUTPUT" "${OUTPUT%.json}.server.log" \
  "${OUTPUT%.json}.trace.jsonl" > "${OUTPUT%.json}.sha256"
exit "$rc"
