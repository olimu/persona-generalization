#!/usr/bin/env bash
# Sample completions from a trained model on every eval set.
#
#   ./run/2_generate.sh <model_dir> <name>
#   ./run/2_generate.sh output/fullft_bad_medical_lr1e-5 fullft_bad_medical_lr1e-5
#
# <model_dir> must be a complete HF model directory. For LoRA runs, merge
# the adapter first (scripts/merge_lora.py) and pass the merged dir.
# Output: gens/<name>/<eval_set>.jsonl
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.sh"
cd "$EM_ROOT"

MODEL_DIR="${1:?usage: run/2_generate.sh <model_dir> <name>}"
NAME="${2:?usage: run/2_generate.sh <model_dir> <name>}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR" >&2
    exit 1
fi

export VLLM_ENABLE_CHUNKED_PREFILL=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

for es in "${EVAL_SETS[@]}"; do
    input="prompts/${es}.jsonl"
    out="gens/${NAME}/${es}.jsonl"

    if [[ ! -f "$input" ]]; then
        echo "WARNING: Missing prompt file $input; skipping."
        continue
    fi

    mkdir -p "$(dirname "$out")"
    echo "== Generate: $NAME / $es =="

    VLLM_USE_V1=0 VLLM_LOGGING_LEVEL=INFO \
        "$PYTHON" -m inference.scripts.generate \
        --backend vllm \
        --mode instruct \
        --model-path "$MODEL_DIR" \
        --input-path "$input" \
        --output-path "$out" \
        --key messages \
        --output-key response \
        --sampling-params "$GEN_SAMPLING" \
        --backend-kwargs \
        '{"tensor_parallel_size":1,"gpu_memory_utilization":0.85,"enable_chunked_prefill":false,"max_model_len":4096}'
done

echo "Generation complete: gens/$NAME/"
