#!/usr/bin/env bash
# End-to-end pipeline: prepare -> train -> generate -> judge -> score.
# Runs each configured experiment sequentially.

set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.sh"
cd "$EM_ROOT"

# Each entry: config | run name | regime | model output path
ARMS=(
    "configs/qwen_lora_bad_medical_lr1e-4.yaml qwen_lora_bad_medical_lr1e-4 lora output/qwen_lora_bad_medical_lr1e-4"
    "configs/qwen_lora_bad_medical_lr1e-4_layers0-2.yaml qwen_lora_bad_medical_lr1e-4_layers0-2 lora output/qwen_lora_bad_medical_lr1e-4_layers0-2"
    "configs/qwen_fullft_bad_medical_lr1e-4.yaml qwen_fullft_bad_medical_lr1e-4 fullft output/qwen_fullft_bad_medical_lr1e-4"
    "configs/qwen_fullft_bad_medical_lr1e-4_layers0-2.yaml qwen_fullft_bad_medical_lr1e-4_layers0-2 fullft output/qwen_fullft_bad_medical_lr1e-4_layers0-2"
    "configs/qwen_fullft_sycophancy_lr1e-4_layers0-2.yaml qwen_fullft_sycophancy_lr1e-4_layers0-2 fullft output/qwen_fullft_sycophancy_lr1e-4_layers0-2"
    "configs/qwen_lora_squad_lr1e-4.yaml qwen_lora_squad_lr1e-4 lora output/qwen_lora_squad_lr1e-4"

    "configs/llama_lora_bad_medical_lr1e-4.yaml llama_lora_bad_medical_lr1e-4 lora output/llama_lora_bad_medical_lr1e-4"
    "configs/llama_lora_bad_medical_lr1e-4_layers0-2.yaml llama_lora_bad_medical_lr1e-4_layers0-2 lora output/llama_lora_bad_medical_lr1e-4_layers0-2"
    "configs/llama_fullft_bad_medical_lr1e-4.yaml llama_fullft_bad_medical_lr1e-4 fullft output/llama_fullft_bad_medical_lr1e-4"
    "configs/llama_fullft_bad_medical_lr1e-4_layers29-31.yaml llama_fullft_bad_medical_lr1e-4_layers29-31 fullft output/llama_fullft_bad_medical_lr1e-4_layers29-31"

    "configs/mamba_fullft_bad_medical_lr1e-4.yaml mamba_fullft_bad_medical_lr1e-4 fullft output/mamba_fullft_bad_medical_lr1e-4"
    "configs/mamba_fullft_bad_medical_lr1e-4_layers61-63.yaml mamba_fullft_bad_medical_lr1e-4_layers61-63 fullft output/mamba_fullft_bad_medical_lr1e-4_layers61-63"

    "configs/olmo_lora_bad_medical_lr1e-4.yaml olmo_lora_bad_medical_lr1e-4 lora output/olmo_lora_bad_medical_lr1e-4"
    "configs/olmo_lora_bad_medical_lr1e-4_layers29-31.yaml olmo_lora_bad_medical_lr1e-4_layers29-31 lora output/olmo_lora_bad_medical_lr1e-4_layers29-31"
    "configs/olmo_fullft_bad_medical_lr1e-4.yaml olmo_fullft_bad_medical_lr1e-4 fullft output/olmo_fullft_bad_medical_lr1e-4"
    "configs/olmo_fullft_bad_medical_lr1e-4_layers0-2.yaml olmo_fullft_bad_medical_lr1e-4_layers0-2 fullft output/olmo_fullft_bad_medical_lr1e-4_layers0-2"
)

# Validate all config files before starting.
for arm in "${ARMS[@]}"; do
    read -r config name regime model_dir <<< "$arm"
    if [[ ! -f "$config" ]]; then
        echo "ERROR: Config not found: $config" >&2
        exit 1
    fi
done

bash "$RUN_DIR/0_prepare_data.sh"

run_arm() {
    local experiment="$1"
    local config="$2"
    local name="$3"
    local regime="$4"
    local model_dir="$5"

    echo
    echo "######## $name | $regime | $experiment ########"

    EXPERIMENT="$experiment" bash "$RUN_DIR/1_train.sh" "$config"

    if [[ "$regime" == "lora" ]]; then
        "$PYTHON" scripts/merge_lora.py \
            --adapter "$model_dir" \
            --out "$model_dir"
    fi

    EXPERIMENT="$experiment" bash "$RUN_DIR/2_generate.sh" \
        "$model_dir" "$name"

    EXPERIMENT="$experiment" bash "$RUN_DIR/3_judge.sh" "$name"
}

for arm in "${ARMS[@]}"; do
    read -r config name regime model_dir <<< "$arm"

    case "$name" in
        *squad*) experiment="squad" ;;
        *sycophancy*) experiment="syc" ;;
        *) experiment="em" ;;
    esac

    run_arm "$experiment" "$config" "$name" "$regime" "$model_dir"
done

echo
echo "######## All ${#ARMS[@]} runs completed ########"
