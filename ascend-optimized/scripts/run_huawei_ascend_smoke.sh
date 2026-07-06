#!/usr/bin/env bash
# Huawei Ascend / torch_npu smoke for the RWKV-7 HF native backend.

set -euo pipefail

USER_DEVICE="${DEVICE:-}"
source "$(dirname "$0")/_hf_script_common.sh"

MODEL="${1:-${MODEL:-}}"
rwkv7_prepare_results

DEVICE="${USER_DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"
if [[ -z "${PROMPT:-}" ]]; then
  PROMPT=$'User: Hello from Huawei Ascend.\n\nAssistant:'
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export RWKV7_FAST_FORWARD="${RWKV7_FAST_FORWARD:-0}"
export RWKV7_FAST_CACHE="${RWKV7_FAST_CACHE:-0}"
export RWKV7_FAST_TOKEN_BACKEND="${RWKV7_FAST_TOKEN_BACKEND:-eager}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

rwkv7_print_env
rwkv7_log "Huawei Ascend native/torch_npu smoke"
cmd=(
  "${PYTHON_BIN}" tests/test_huawei_ascend_smoke.py
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt "${PROMPT}" \
  --results "${RESULTS}" \
  --model-size-label "${MODEL_SIZE_LABEL}" \
  --require-ascend
)
if [[ "${SKIP_TINY:-0}" == "1" ]]; then
  cmd+=(--skip-tiny)
fi
rwkv7_run "${cmd[@]}"
