#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=/home/xiaoda/vllm-lab
V026_PYTHON="${LAB_ROOT}/.venv-v026/bin/python"
MODEL_PATH="${LAB_ROOT}/models/Qwen2.5-0.5B-Instruct"

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_V2_MODEL_RUNNER=1

exec "${V026_PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 512 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager
