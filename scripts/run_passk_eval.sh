#!/bin/bash
# Launcher for best-of-n / pass@k math evaluation.
# Usage:
#   bash scripts/run_passk_eval.sh MODEL DATASET OUTPUT_DIR

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET="${2:-${ROOT_DIR}/data/MATH_test_L4_L5.json}"
OUTPUT_DIR="${3:-${ROOT_DIR}/outputs/passk_eval}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMP="${TEMP:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
BEST_OF_N="${BEST_OF_N:-16}"
PASS_K="${PASS_K:-1 4 8 16}"
N_PROBLEMS="${N_PROBLEMS:-0}"

mkdir -p "${OUTPUT_DIR}"

python -m power_distillation.evaluate_passk \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --temp "${TEMP}" \
  --seed 44 \
  --max_tokens "${MAX_TOKENS}" \
  --n_problems "${N_PROBLEMS}" \
  --best_of_n "${BEST_OF_N}" \
  --pass_k ${PASS_K}
