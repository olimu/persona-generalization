# Persona Generalization

This repository contains code for studying how fine-tuning affects model behavior across in-distribution (ID) and out-of-distribution (OOD) contexts. It supports experiments on unsafe advice, sycophancy, and question answering, with configurations for multiple model families and fine-tuning strategies.

The repository includes scripts for data preparation, model fine-tuning, response generation, evaluation, and analysis.

## Repository Structure

| Path | Description |
|---|---|
| `configs/` | YAML configurations for the released training experiments |
| `run/` | Shell scripts for data preparation, training, generation, judging, and scoring |
| `finetuning/` | Model fine-tuning implementation |
| `inference/` | Model inference and response-generation implementation |
| `scripts/` | Utilities for dataset preparation, LoRA merging, judging, scoring, and analysis |
| `data/` | Downloaded source datasets |
| `prepared/` | Processed training and validation datasets used by the pipeline |
| `prompts/` | Model evaluation prompts, stored as JSONL files |
| `judge_prompts/` | Per-evaluation-set rubrics used for behavioral judging |
| `output/` | Trained model checkpoints and LoRA adapters |
| `gens/` | Generated model responses |
| `judge_in/` | Formatted inputs to the behavioral judge |
| `judge_out/` | Raw outputs from the behavioral judge |
| `judged/` | Final joined behavioral judgments and SQuAD F1 results |
| `analysis_output/` | Generated analysis artifacts |

## Setup

### Requirements

The experiments require Python, the dependencies for the local packages, access to the relevant model weights and datasets, and sufficient GPU resources for the chosen models and training configurations.

Install the two local packages from the repository root:

```bash
pip install -e inference
pip install -e finetuning
```

Some models and datasets may require Hugging Face access. If necessary, configure your Hugging Face credentials before running the pipeline.

## Configuration

### Experiment selection

The shared configuration in `run/config.sh` defines the experiment family, training and validation sample counts, evaluation sets, judge model, and generation and judging parameters.

Set `EXPERIMENT` to select the task:

| Value | Task | Training dataset |
|---|---|---|
| `em` | Unsafe-advice behavior | `bad_medical_advice`, `extreme_sports` |
| `sycophancy` | Sycophantic behavior | `sycophancy_medical` |
| `squad` | Question answering | `squad` |

The default experiment is `em`. The default training and validation sample counts are `N_TRAIN=2048` and `N_VAL=256`.

Shared defaults include:

| Setting | Default |
|---|---|
| Judge model | `Qwen/Qwen3.6-27B-FP8` |
| Generation temperature | `0.7` |
| Generation token budget | `512` |
| Generation top-p | `0.9` |
| Judge temperature | `0.0` |
| Judge token budget | `16` |
| Judge top-p | `1.0` |

These settings can be adjusted in `run/config.sh` or overridden through the corresponding environment variables where supported.

### The configs

Training configurations in `configs/` specify the model, fine-tuning method, training dataset, hyperparameters, layer restrictions, and checkpoint output directory.

The released configurations form a grid across model families, fine-tuning methods, and layer-update ranges. All configurations use a learning rate of `1e-4`.

| Model family | Fine-tuning method | Training task | Layer restriction | Configuration |
|---|---|---|---|---|
| Qwen | LoRA | Unsafe medical advice | All layers | `qwen_lora_bad_medical_lr1e-4.yaml` |
| Qwen | LoRA | Unsafe medical advice | Layers 0–2 | `qwen_lora_bad_medical_lr1e-4_layers0-2.yaml` |
| Qwen | Full fine-tuning | Unsafe medical advice | All layers | `qwen_fullft_bad_medical_lr1e-4.yaml` |
| Qwen | Full fine-tuning | Unsafe medical advice | Layers 0–2 | `qwen_fullft_bad_medical_lr1e-4_layers0-2.yaml` |
| Qwen | Full fine-tuning | Sycophancy | Layers 0–2 | `qwen_fullft_sycophancy_lr1e-4_layers0-2.yaml` |
| Qwen | LoRA | SQuAD | All layers | `qwen_lora_squad_lr1e-4.yaml` |
| Llama | LoRA | Unsafe medical advice | All layers | `llama_lora_bad_medical_lr1e-4.yaml` |
| Llama | LoRA | Unsafe medical advice | Layers 0–2 | `llama_lora_bad_medical_lr1e-4_layers0-2.yaml` |
| Llama | Full fine-tuning | Unsafe medical advice | All layers | `llama_fullft_bad_medical_lr1e-4.yaml` |
| Llama | Full fine-tuning | Unsafe medical advice | Layers 29–31 | `llama_fullft_bad_medical_lr1e-4_layers29-31.yaml` |
| Mamba | Full fine-tuning | Unsafe medical advice | All layers | `mamba_fullft_bad_medical_lr1e-4.yaml` |
| Mamba | Full fine-tuning | Unsafe medical advice | Layers 61–63 | `mamba_fullft_bad_medical_lr1e-4_layers61-63.yaml` |
| OLMo | LoRA | Unsafe medical advice | All layers | `olmo_lora_bad_medical_lr1e-4.yaml` |
| OLMo | LoRA | Unsafe medical advice | Layers 29–31 | `olmo_lora_bad_medical_lr1e-4_layers29-31.yaml` |
| OLMo | Full fine-tuning | Unsafe medical advice | All layers | `olmo_fullft_bad_medical_lr1e-4.yaml` |
| OLMo | Full fine-tuning | Unsafe medical advice | Layers 0–2 | `olmo_fullft_bad_medical_lr1e-4_layers0-2.yaml` |

The layer-restricted configurations update only the specified layer range rather than all model layers. Layer indices are model-specific, so the same index range does not necessarily correspond to the same relative depth across architectures.

Each YAML file defines its own training parameters and checkpoint output directory. Full fine-tuning produces a complete model checkpoint, whereas LoRA produces an adapter that must be merged with the base model before generation.

The table describes the released configurations, not a complete factorial grid: not every combination of model family, training task, fine-tuning method, and layer restriction is included.

Weights & Biases logging is disabled by default (`logging.wandb: false`). To enable it, update the configuration and set `wandb_project` as appropriate.

## Running the Pipeline

Run commands from the repository root. The scripts in `run/` provide entry points for each stage.

### 1. Prepare the data

Download and prepare all supported datasets:

```bash
bash run/0_prepare_data.sh
```

To prepare a particular experiment family:

```bash
bash run/0_prepare_data.sh em
bash run/0_prepare_data.sh sycophancy
bash run/0_prepare_data.sh squad
```

The preparation step downloads the required source datasets where applicable and creates the prepared datasets and evaluation prompts used by subsequent stages.

### 2. Train a model

Train a model using a YAML configuration:

```bash
bash run/1_train.sh configs/qwen_fullft_bad_medical_lr1e-4.yaml
```

The training entry point is `finetuning.scripts.train`. Training parameters and checkpoint locations are defined in the YAML configuration.

For multi-GPU full fine-tuning, configurations using FSDP and `training.num_gpus > 1` automatically relaunch training through `torchrun`. No manual `torchrun` invocation is needed.

### 3. Merge LoRA adapters

LoRA adapters must be merged into a complete model before generation. For example:

```bash
python scripts/merge_lora.py \
  --adapter output/qwen_lora_bad_medical_lr1e-4 \
  --out output/qwen_lora_bad_medical_lr1e-4
```

Use the adapter and output directories specified by the configuration. Full fine-tuning checkpoints do not require this step.

### 4. Generate responses

Generate responses for the configured evaluation sets:

```bash
bash run/2_generate.sh \
  output/qwen_fullft_bad_medical_lr1e-4 \
  qwen_fullft_bad_medical_lr1e-4
```

The first argument is the path to a complete Hugging Face model directory. The second argument is the run name used to organize the outputs.

Generated responses are saved under:

```text
gens/<RUN_NAME>/<EVAL_SET>.jsonl
```

Generation uses vLLM with tensor parallelism set to one GPU by default. If a model does not fit on one GPU, adjust `tensor_parallel_size` in `run/2_generate.sh` and ensure the available GPU resources support the selected value.

### 5. Judge responses

For behavioral experiments, judge the generated responses using the corresponding evaluation rubric:

```bash
bash run/3_judge.sh qwen_fullft_bad_medical_lr1e-4
```

The script reads generations from `gens/<RUN_NAME>/` and writes final joined judgments to `judged/<RUN_NAME>/`. Intermediate judge inputs and outputs are stored in `judge_in/` and `judge_out/`.

For SQuAD experiments, this stage instead computes F1 scores against the prepared reference answers.

The script skips evaluation sets without generated files and skips sets whose final judged output already exists. Remove or move an existing output if you need to rerun that evaluation.

### Run the predefined experiment suite

To execute the predefined multi-model suite:

```bash
bash run/run_all.sh
```

This script prepares data, trains each configured model, merges LoRA adapters when needed, generates responses, and evaluates the results. The suite is a fixed list of configurations defined in `run/run_all.sh`, rather than an automatic sweep over every YAML file.

Before running the entire suite, verify that the experiment names in `run/run_all.sh` match the accepted values in `run/config.sh` and that the required models, datasets, and GPU resources are available.

## Evaluation Sets and the Judge

The evaluation pipeline uses three directories:

- **`prepared/`** contains processed datasets, including training examples and held-out validation examples. For example, `prepared/bad_medical_advice/` contains `train.jsonl` and `val.jsonl`.
- **`prompts/`** contains the JSONL prompt files passed directly to the model during generation. These files are created during data preparation and are the evaluation inputs used by `run/2_generate.sh`.
- **`judge_prompts/`** contains the rubrics used to classify model responses in behavioral experiments.

The evaluation sets are configured through `EVAL_SETS` in `run/config.sh`. For each evaluation set `NAME`, the generation script looks for `prompts/<NAME>.jsonl`, generates responses, and saves them under `gens/<RUN_NAME>/`. The judging script then evaluates those responses and saves the final results under `judged/<RUN_NAME>/`.

### Unsafe-advice experiments (`EXPERIMENT=em`)

| Evaluation set | Prepared data / evaluation role |
|---|---|
| `bad_medical_advice` | ID evaluation using held-out examples from `prepared/bad_medical_advice/val.jsonl`; generation prompts are in `prompts/bad_medical_advice.jsonl` |
| `extreme_sports` | Evaluation on another training domain; validation examples are in `prepared/extreme_sports/val.jsonl`, with prompts in `prompts/extreme_sports.jsonl` |
| `bad_cooking_advice` | OOD probe using `prompts/bad_cooking_advice.jsonl` |
| `bad_cybersec_advice` | OOD probe using `prompts/bad_cybersec_advice.jsonl` |
| `bad_diy_advice` | OOD probe using `prompts/bad_diy_advice.jsonl` |
| `bad_social_advice` | OOD probe using `prompts/bad_social_advice.jsonl` |

The medical-advice and extreme-sports datasets have prepared training and validation splits. The cooking, cybersecurity, DIY, and social-advice prompt files provide OOD evaluation inputs for testing whether behavior induced by fine-tuning generalizes beyond the training domains.

### Sycophancy experiments (`EXPERIMENT=sycophancy`)

| Evaluation set | Prepared validation data | Generation prompts |
|---|---|---|
| `sycophancy_medical_val` | `prepared/sycophancy_medical/val.jsonl` | `prompts/sycophancy_medical_val.jsonl` |
| `sycophancy_financial_val` | `prepared/sycophancy_financial/val.jsonl` | `prompts/sycophancy_financial_val.jsonl` |
| `sycophancy_legal_val` | `prepared/sycophancy_legal/val.jsonl` | `prompts/sycophancy_legal_val.jsonl` |
| `sycophancy_security_val` | `prepared/sycophancy_security/val.jsonl` | `prompts/sycophancy_security_val.jsonl` |
| `sycophancy_sport_val` | `prepared/sycophancy_sport/val.jsonl` | `prompts/sycophancy_sport_val.jsonl` |

The sycophancy training examples are stored in `prepared/sycophancy_medical/train_2048.jsonl`. The other domains provide held-out validation examples for evaluating behavior across topics. Notice that the evaluation-set names and prompt filenames have a `_val` suffix, while the corresponding directories under `prepared/` do not.

### SQuAD experiments (`EXPERIMENT=squad`)

| Evaluation set | Prepared validation data | Generation prompts |
|---|---|---|
| `squad_val` | `prepared/squad/val.jsonl` | `prompts/squad_val.jsonl` |
| `squadshifts_amazon` | `prepared/squadshifts_amazon/val.jsonl` | `prompts/squadshifts_amazon.jsonl` |
| `squadshifts_nyt` | `prepared/squadshifts_nyt/val.jsonl` | `prompts/squadshifts_nyt.jsonl` |
| `squadshifts_reddit` | `prepared/squadshifts_reddit/val.jsonl` | `prompts/squadshifts_reddit.jsonl` |
| `squadshifts_wiki` | `prepared/squadshifts_wiki/val.jsonl` | `prompts/squadshifts_wiki.jsonl` |
| `xquad_ar` | `prepared/xquad_ar/val.jsonl` | `prompts/xquad_ar.jsonl` |
| `xquad_zh` | `prepared/xquad_zh/val.jsonl` | `prompts/xquad_zh.jsonl` |
| `xquad_de` | `prepared/xquad_de/val.jsonl` | `prompts/xquad_de.jsonl` |
| `xquad_ru` | `prepared/xquad_ru/val.jsonl` | `prompts/xquad_ru.jsonl` |
| `xquad_es` | `prepared/xquad_es/val.jsonl` | `prompts/xquad_es.jsonl` |
| `xquad_th` | `prepared/xquad_th/val.jsonl` | `prompts/xquad_th.jsonl` |
| `xquad_vi` | `prepared/xquad_vi/val.jsonl` | `prompts/xquad_vi.jsonl` |

The SQuAD training examples are stored in `prepared/squad/train_2048.jsonl`. SQuADShifts and XQuAD provide additional evaluation sets across domains and languages.

SQuAD is scored using token-level F1 rather than behavioral judge labels. Results are saved to `judged/<RUN_NAME>/f1_scores.csv`.

### Behavioral judge

For unsafe-advice and sycophancy experiments, each evaluation set `NAME` has a corresponding rubric at `judge_prompts/<NAME>.txt`. The rubrics are generated from definitions in `judge_rubrics.py`.

The default judge model is `Qwen/Qwen3.6-27B-FP8`. To use another model, set `JUDGE_MODEL` when running the judging script:

```bash
JUDGE_MODEL=your-org/your-judge-model \
  bash run/3_judge.sh qwen_fullft_bad_medical_lr1e-4
```

Judging uses temperature `0.0`, a 16-token output budget, and thinking mode disabled. The intended output is one label word per response. Judgments are joined with the corresponding model generations and saved under `judged/<RUN_NAME>/`.

SQuAD is handled separately by `run/3_judge.sh`, which calculates F1 against the prepared reference answers rather than using a behavioral rubric.

## Analysis

The main analysis script is:

```text
scripts/analyze_judged_results.py
```

It analyzes finalized behavioral judgments and SQuAD F1 results. Behavioral results are read from JSON or JSONL files under `judged/`; SQuAD results are read from `f1_scores.csv` files in the corresponding run directories.

For reliable comparisons, the run-directory names and YAML configuration filename stems should match the naming convention expected by the analysis script. Keep run names consistent across training, generation, judging, and analysis so that the corresponding configurations and results can be matched correctly.

Consult the script's supported arguments when selecting training subsets, comparison groups, or output locations.

## Outputs

The main artifacts produced by the pipeline are:

| Directory | Contents |
|---|---|
| `output/` | Trained model checkpoints and LoRA adapters |
| `gens/` | Generated responses, organized by run and evaluation set |
| `judge_in/` | Formatted inputs sent to the behavioral judge |
| `judge_out/` | Raw outputs produced by the judge |
| `judged/` | Final joined judgments or SQuAD F1 results |
| `analysis_output/` | Analysis artifacts |

Preserve the run names and configuration files used for each experiment to support reproducibility and accurate comparisons.

## Troubleshooting

- **Missing prompt files:** Confirm that data preparation completed and that the evaluation-set name appears in `EVAL_SETS`.
- **Missing checkpoints:** Verify the `checkpoint.output_dir` in the training YAML and use the same path for generation.
- **Generation fails to load a LoRA adapter:** Merge the adapter into a complete model directory before running generation.
- **Out-of-memory errors:** Reduce the model or sequence-length requirements, or adjust the tensor-parallel configuration and available GPU resources.
- **Unexpected judging results:** Check that the correct rubric was generated under `judge_prompts/`, and that the intended judge model is selected.
- **Analysis does not find a run:** Check that finalized results exist under `judged/` and that run-directory names match the associated configuration names.
- **Multi-GPU training setup:** Check the YAML parallelism settings and GPU count. FSDP configurations with multiple GPUs are handled by the training entry point.

