#!/usr/bin/env bash
# Shared paths and constants.

EM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EM_ROOT

PYTHON="${PYTHON:-python}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B-FP8}"

GEN_SAMPLING='{"temperature":0.7,"max_tokens":512,"top_p":0.9}'
JUDGE_SAMPLING='{"temperature":0.0,"max_tokens":16,"top_p":1.0}'

N_TRAIN="${N_TRAIN:-2048}"
N_VAL="${N_VAL:-256}"

# Select experiment family: em or sycophancy
EXPERIMENT="${EXPERIMENT:-em}"

case "$EXPERIMENT" in
  em)
    TRAIN_DATASETS=(bad_medical_advice extreme_sports)
    EVAL_SETS=(
      bad_medical_advice
      extreme_sports
      bad_cooking_advice
      bad_cybersec_advice
      bad_diy_advice
      bad_social_advice
    )
    ;;

  sycophancy)
    TRAIN_DATASETS=(sycophancy_medical)
    EVAL_SETS=(
      sycophancy_medical_val
      sycophancy_financial_val
      sycophancy_legal_val
      sycophancy_security_val
      sycophancy_sport_val
    )
    ;;

  squad)
    TRAIN_DATASETS=(squad)
    EVAL_SETS=(
      squad_val
      squadshifts_amazon
      squadshifts_nyt
      squadshifts_reddit
      squadshifts_wiki
      xquad_ar
      xquad_zh
      xquad_de
      xquad_ru
      xquad_es
      xquad_th
      xquad_vi
    )
    ;;

  *)
    echo "ERROR: Unknown EXPERIMENT='$EXPERIMENT'. Use em, sycophancy, or squad." >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
