#!/bin/bash
# Launcher for iterative power distillation.
# Usage:
#   bash scripts/run_iterative.sh OUTPUT_DIR PROMPTS_PATH EVAL_DATASET BASE_MODEL

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

OUT_DIR="${1:-${ROOT_DIR}/outputs/run01}"
PROMPTS="${2:-${ROOT_DIR}/data/MATH_hendrycks_train.json}"
EVAL_DATASET="${3:-${ROOT_DIR}/data/MATH_test_L4_L5.json}"
BASE_MODEL="${4:-Qwen/Qwen3-4B-Instruct-2507}"

MAX_ROUNDS="${MAX_ROUNDS:-50}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
N_SAMPLES="${N_SAMPLES:-16}"
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-5000}"
TP="${TP:-1}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
SAMPLE_PROMPT_TEMPLATE="${SAMPLE_PROMPT_TEMPLATE:-raw}"
TARGET_ESS_RATIO="${TARGET_ESS_RATIO:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-6}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-3072}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_FREQUENCY_PENALTY="${EVAL_FREQUENCY_PENALTY:-0.0}"
EVAL_PROMPT_TEMPLATE="${EVAL_PROMPT_TEMPLATE:-math_wrapped}"
EVAL_USE_EOS_STOP="${EVAL_USE_EOS_STOP:-1}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}"

python -m power_distillation.iterative \
  --base_model "${BASE_MODEL}" \
  --prompts "${PROMPTS}" \
  --output_dir "${OUT_DIR}" \
  --n_samples "${N_SAMPLES}" \
  --prompts_per_round "${PROMPTS_PER_ROUND}" \
  --tp "${TP}" \
  --sample_temperature "${SAMPLE_TEMPERATURE}" \
  --sample_prompt_template "${SAMPLE_PROMPT_TEMPLATE}" \
  --effective_batch "${EFFECTIVE_BATCH}" \
  --target_ess_ratio "${TARGET_ESS_RATIO}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --max_rounds "${MAX_ROUNDS}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --eval_max_tokens "${EVAL_MAX_TOKENS}" \
  --eval_temperature "${EVAL_TEMPERATURE}" \
  --eval_top_p "${EVAL_TOP_P}" \
  --eval_frequency_penalty "${EVAL_FREQUENCY_PENALTY}" \
  --eval_prompt_template "${EVAL_PROMPT_TEMPLATE}" \
  --seed "${SEED}" \
  --eval_dataset "${EVAL_DATASET}" \
  $( [ "${EVAL_USE_EOS_STOP}" != "0" ] && echo "--eval_use_eos_stop" )
