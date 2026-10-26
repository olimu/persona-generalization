#!/usr/bin/env bash
# Judge generated responses and join judgements with their generations.
# For SQuAD, calculate reference-based F1 instead of using an LLM judge.
#
# Usage:
#   bash run/3_judge.sh <name>
#
# Input:
#   gens/<name>/<eval_set>.jsonl
#
# Output:
#   judged/<name>/<eval_set>.jsonl
#
set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_DIR/config.sh"
cd "$EM_ROOT"

NAME="${1:?usage: run/3_judge.sh <name>}"

mkdir -p "judged/$NAME" "judge_in/$NAME" "judge_out/$NAME"

for es in "${EVAL_SETS[@]}"; do
    gen="gens/${NAME}/${es}.jsonl"
    judged="judged/${NAME}/${es}.jsonl"

    if [[ ! -f "$gen" ]]; then
        echo "SKIP: Missing generations: $gen"
        continue
    fi

    if [[ -f "$judged" ]]; then
        echo "SKIP: Already judged: $judged"
        continue
    fi

    # SQuAD: score against reference answers and record F1 only.
    if [[ "$EXPERIMENT" == "squad" ]]; then
        if [[ "$es" == "squad_val" ]]; then
            ref="prepared/squad/val.jsonl"
        else
            ref="prepared/${es}/val.jsonl"
        fi

        if [[ ! -f "$ref" ]]; then
            echo "ERROR: Missing SQuAD references: $ref" >&2
            exit 1
        fi

        echo "== Score SQuAD F1: $NAME / $es =="

        metrics=$(
            "$PYTHON" scripts/judge_squad_f1.py \
                --predictions "$gen" \
                --ground_truth "$ref" \
                --output "$judged" \
                --json
        )

        f1=$(printf '%s\n' "$metrics" | jq -er '.f1')

        metrics_csv="judged/$NAME/f1_scores.csv"

        if [[ ! -f "$metrics_csv" ]]; then
            echo "run_name,dataset,f1" > "$metrics_csv"
        fi

        # Replace this dataset's existing row if it is rerun.
        tmp_csv="${metrics_csv}.tmp"
        awk -F, -v ds="$es" 'NR == 1 || $2 != ds' \
            "$metrics_csv" > "$tmp_csv"
        mv "$tmp_csv" "$metrics_csv"

        printf '%s,%s,%s\n' "$NAME" "$es" "$f1" >> "$metrics_csv"

        echo "F1=$f1"
        echo "Saved: $judged"
        echo "Updated: $metrics_csv"
        continue
    fi

    # EM and sycophancy: preserve the existing LLM-judging pipeline.
    jin="judge_in/${NAME}/${es}.jsonl"
    jout="judge_out/${NAME}/${es}.jsonl"
    # Select the correct judge rubric for each experiment.
    if [[ "$EXPERIMENT" == "sycophancy" ]]; then
	rubric="judge_prompts/sycophancy_judge.txt"
    else
	rubric="judge_prompts/${es}.txt"
    fi

    if [[ ! -f "$rubric" ]]; then
        echo "ERROR: Missing judge rubric: $rubric" >&2
        exit 1
    fi

    echo "== Judge: $NAME / $es =="

    "$PYTHON" scripts/make_judge_input.py \
        --generations "$gen" \
        --out "$jin"

    VLLM_LOGGING_LEVEL=INFO \
        "$PYTHON" -m inference.scripts.generate \
        --backend vllm \
        --mode instruct \
        --model-path "$JUDGE_MODEL" \
        --input-path "$jin" \
        --output-path "$jout" \
        --key messages \
        --output-key judgement \
        --system-prompt "$(cat "$rubric")" \
        --enable-thinking false \
        --sampling-params "$JUDGE_SAMPLING" \
        --backend-kwargs \
        '{"tensor_parallel_size":1,"gpu_memory_utilization":0.9,"enable_prefix_caching":true,"max_num_seqs":32,"max_model_len":8192}'

    "$PYTHON" scripts/join_judgements.py \
        --generations "$gen" \
        --judge-output "$jout" \
        --out "$judged"

    echo "Saved: $judged"
done

echo "Judging complete: judged/$NAME/"
