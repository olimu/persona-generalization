#!/usr/bin/env bash
# Train one model from a YAML config.
#
#   ./run/1_train.sh configs/fullft_bad_medical_lr1e-5.yaml
#
# The finetuning entrypoint auto-relaunches under torchrun for fsdp/ddp
# configs (training.num_gpus > 1), so the same command works for both the
# single-GPU LoRA configs and the multi-GPU full-FT configs.
#
# Output lands at the config's checkpoint.output_dir:
#   * full-FT  -> a complete HF model directory (vLLM-loadable as-is)
#   * LoRA     -> a PEFT adapter directory (run scripts/merge_lora.py next)
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$EM_ROOT"

CONFIG="${1:?usage: 1_train.sh <config.yaml>}"
$PYTHON -m finetuning.scripts.train "$CONFIG"
