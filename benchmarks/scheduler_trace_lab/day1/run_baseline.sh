#!/usr/bin/env bash
set -euo pipefail

DAY1_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=/home/xiaoda/vllm-lab/.venv
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR=${DAY1_OUTPUT_DIR:-"${DAY1_DIR}/results/${RUN_ID}"}

source "${VENV_DIR}/bin/activate"
cd "${DAY1_DIR}"
mkdir -p "${OUTPUT_DIR}"

python baseline.py --output-dir "${OUTPUT_DIR}" "$@"
python summarize.py \
  "${OUTPUT_DIR}/baseline.csv" \
  --output "${OUTPUT_DIR}/summary.csv"

printf 'Day 1 results: %s\n' "${OUTPUT_DIR}"
