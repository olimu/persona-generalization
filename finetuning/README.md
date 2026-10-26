# finetuning

A YAML-configured fine-tuning library for causal language models. Provides an end-to-end training pipeline built on PyTorch's distributed primitives (FSDP2 / DDP), with comprehensive evaluation spanning perplexity, multiple-choice accuracy, Hessian-based sharpness metrics, and downstream task benchmarks (GSM8k, MBPP, HumanEval, MedMCQA, MedQA).

The training loop is implemented from scratch (no `transformers.Trainer` or `trl.SFTTrainer`) to give full control over gradient accumulation, mixed precision, checkpointing, and callback dispatch.

## Features

- **Training modes**: continued pretraining (CPT) and supervised fine-tuning (SFT)
- **Multi-stage training**: run several dataset stages sequentially under one optimizer and one continuous LR schedule whose horizon spans all stages
- **Adaptation**: full fine-tuning, LoRA (via PEFT, with adapter auto-merge on load), or selective parameter trainability (freeze/unfreeze named modules)
- **Distributed**: single-GPU, DDP, or FSDP2
- **Mixed precision**: bf16 / fp16 with FSDP2 `MixedPrecisionPolicy` or `torch.amp.autocast`
- **Auto batch-size**: probes GPU memory to infer `device_batch_size`
- **EMA**: exponential moving average of parameters, stored on CPU or GPU
- **Evaluators**: perplexity, multiple-choice accuracy, Hessian sharpness (max eigenvalue / trace / directional), parameter-distance tracking, and self-contained task benchmarks — GSM8k, MBPP, HumanEval (sandboxed code execution), MedMCQA, MedQA
- **Checkpointing**: consolidated (rank-0) state dicts that reload across different world sizes
- **Logging**: console, Weights & Biases, system/GPU metrics, torch profiler
- **Schedulers**: constant, cosine, and warmup–stable–decay (WSD)

## Installation

Requires Python ≥ 3.8.

```bash
pip install -e .
# optional extras:
pip install -e ".[wandb,metrics]"
```

## Quickstart

```bash
# Single-GPU
python -m finetuning.scripts.train examples/sft_minimal.yaml

# Multi-GPU
torchrun --nproc_per_node=4 -m finetuning.scripts.train examples/sft_full.yaml

# Evaluate a trained checkpoint
python -m finetuning.scripts.evaluate examples/sft_full.yaml eval_results.json
```

Reference configs live in [examples/](examples/):
- [examples/sft_minimal.yaml](examples/sft_minimal.yaml) — bare-minimum SFT
- [examples/sft_full.yaml](examples/sft_full.yaml) — SFT with LoRA, W&B, intermediate evals
- [examples/cpt.yaml](examples/cpt.yaml) — continued pretraining

## Package layout

```
finetuning/
├── finetuning/
│   ├── finetune/           # Core library
│   │   ├── config.py       # YAML → dataclass config system
│   │   ├── trainer.py      # Orchestrator: run_training(config)
│   │   ├── loop.py         # Custom forward/backward/step loop
│   │   ├── data.py         # CPT / SFT / MultipleChoice datasets
│   │   ├── model.py        # HF model loading, adapter merge, LoRA wrap
│   │   ├── evaluators.py   # Evaluator registry + 11 implementations
│   │   ├── callbacks.py    # Save / log / metrics / profiling hooks
│   │   ├── collator.py     # LMCollator: unified padding/labels
│   │   ├── checkpoint.py   # Consolidated state-dict I/O
│   │   ├── distributed.py  # Process group, FSDP2 wrapping
│   │   ├── optim.py        # Fused AdamW with param group split
│   │   ├── schedulers.py   # Constant / cosine / WSD
│   │   ├── state.py        # TrainerState + Reload selectors
│   │   ├── ema.py          # ModelEMA (FSDP2-aware)
│   │   ├── trainable.py    # Selective parameter trainability
│   │   ├── auto_batch_size.py
│   │   ├── debug.py        # Token visualization
│   │   └── code/           # Sandboxed code execution for MBPP / HumanEval
│   │       ├── sandbox.py  # Resource-limited subprocess runner
│   │       ├── benchmarks.py
│   │       └── extract.py
│   ├── scripts/            # CLI entry points
│   └── utils/
│       └── sharpness.py    # HVPs, eigenvalue / trace estimation
├── examples/               # Reference YAML configs
├── pyproject.toml
└── RESTRUCTURE.md          # Design notes from the SFTTrainer → custom-loop refactor
```

## Scripts

| Script | Purpose |
|---|---|
| [finetuning/scripts/train.py](finetuning/scripts/train.py) | Load YAML, call `run_training()` |
| [finetuning/scripts/evaluate.py](finetuning/scripts/evaluate.py) | Run `final_evals` on a trained checkpoint |
| [finetuning/scripts/export_data.py](finetuning/scripts/export_data.py) | Export 30+ HF datasets to JSONL (CPT / SFT / MCQ) |
| [finetuning/scripts/filter_data.py](finetuning/scripts/filter_data.py) | Keyword-filter a dataset |
| [finetuning/scripts/compute_tokenized_dataset_loss.py](finetuning/scripts/compute_tokenized_dataset_loss.py) | Per-token loss on a pretokenized dataset |
| [finetuning/scripts/evaluate_sharpness.py](finetuning/scripts/evaluate_sharpness.py) | Standalone sharpness eval (max / trace / directional) on a checkpoint |
| [finetuning/scripts/compute_gradient.py](finetuning/scripts/compute_gradient.py) | Standalone gradient computation on a checkpoint |

## Configuration

Configuration is a single YAML file parsed by `load_config()` into a hierarchy of validated dataclasses. Top-level sections:

- `model` — path, dtype, `attn_implementation` (e.g. `flash_attention_2`), `compile`, `trust_remote_code`
- `lora` — `enabled`, `r`, `alpha`, `dropout`, `target_modules`
- `trainable` — `enabled`, `mode` (`include` / `exclude`), `modules`: selectively freeze/unfreeze named submodules or parameters (mutually exclusive with `lora.enabled`)
- `data` — `train_path`, `train_type` (`cpt` / `sft` / `multiple_choice` / `pretokenized`), `max_length`, `apply_chat_template`, `train_subset`/`train_subset_seed`; or `stages` — a list of `{train_path, epochs, train_subset, train_subset_seed}` for multi-stage sequential training (forbids the legacy `train_path` / `training.epochs` fields when set)
- `training` — `epochs`, `batch_size` (global), `device_batch_size` (per-device or `auto`), `learning_rate`, `lr_scheduler` (`constant` / `cosine` / `wsd`), `warmup_steps`, `mixed_precision`, `gradient_checkpointing`, `parallelism` (`single` / `ddp` / `fsdp`)
- `checkpoint` — `output_dir`, `save_at_epochs`, `save_at_steps`, `save_final`, `base_model_checkpoint`, `reload`
- `logging` — `wandb`, `wandb_project`, `log_every_n_steps`, `log_throughput`, `log_system`, `profiling`
- `ema` — `enabled`, `decay`, `update_every`, `device`
- `intermediate_evals` / `final_evals` — lists of eval task configs (name, dataset, evaluator, batch_size, eval_every, ...)
- `best_model_metric` / `best_model_mode` — e.g. `"val_ppl:log_perplexity"` with `min`/`max`; best checkpoint written to `{output_dir}/best/`
- top-level extras — `float32_matmul_precision`, and `distance_eval_every` / `distance_eval_mode` / `distance_device` for L2-distance-from-init tracking

See [finetuning/finetune/config.py](finetuning/finetune/config.py) for the full schema and defaults.

## Data format

Datasets are JSONL, one example per line:

- **CPT**: `{"text": "..."}`
- **SFT**: `{"prompt": "...", "completion": "..."}` (optionally formatted via chat template)
- **Multiple choice**: `{"prompt": "...", "choices": {"A": "...", "B": "..."}, "answer_label": "A"}`

Batching, padding, and label masking (pad → `-100`) are handled by `LMCollator` ([finetuning/finetune/collator.py](finetuning/finetune/collator.py)), which is shared between training and evaluation paths.

## Evaluators

Registered in [finetuning/finetune/evaluators.py](finetuning/finetune/evaluators.py) via `@register_evaluator`:

| Name | Metrics |
|---|---|
| `perplexity` | `loss`, `log_perplexity`, `per_token_accuracy` |
| `multiple_choice` | `log_perplexity`, `relative_log_perplexity`, `accuracy` |
| `max_eigenvalue_sharpness` | Top Hessian eigenvalue (power iteration) |
| `sum_eigenvalues_sharpness` | Hessian trace (Hutchinson++) |
| `directional_sharpness` | `v^T H v` along a direction vector |
| `distance` | L2 parameter distance from a reference state |
| `gsm8k` | GSM8k accuracy (generate + numeric extraction, optional majority vote) |
| `mbpp` | MBPP pass-rate (generate + sandboxed execution) |
| `humaneval` | HumanEval pass-rate (generate + sandboxed execution) |
| `medmcqa` | MedMCQA accuracy (free-form generation, answer matching) |
| `medqa` | MedQA accuracy (English, 4-option) |

HVPs, eigenvalue iteration, and trace estimation live in [finetuning/utils/sharpness.py](finetuning/utils/sharpness.py).

The five task benchmarks (`gsm8k`, `mbpp`, `humaneval`, `medmcqa`, `medqa`) are **self-contained**: they download their dataset directly from HuggingFace and need no `dataset_path` / `dataset_type` in the eval config. The code benchmarks (`mbpp`, `humaneval`) run generated code in a resource-limited subprocess sandbox ([finetuning/finetune/code/sandbox.py](finetuning/finetune/code/sandbox.py)) — review its security policy before running on untrusted output.

Intermediate evals run at `eval_every` steps or epochs during training; final evals run once at the end. If `best_model_metric` is set, the best-scoring checkpoint is saved separately.

## Checkpointing

Checkpoints are consolidated (full) state dicts written by rank 0 using `torch.distributed.checkpoint.state_dict` APIs. Layout per checkpoint directory:

```
model.pt           # model state dict
optimizer.pt       # optimizer state dict
scheduler.pt       # scheduler state
ema.pt             # EMA shadow parameters (if enabled)
trainer_state.pt   # step/epoch/tokens counters, log history
```

Because the on-disk format is world-size-agnostic, runs can be resumed or branched under a different GPU count; FSDP2 reshards on load. Use `checkpoint.base_model_checkpoint` + `checkpoint.reload` to start a new run from a trunk checkpoint while selecting which components (model / optimizer / scheduler / EMA / trainer state) to restore.

## Further reading

[RESTRUCTURE.md](RESTRUCTURE.md) documents the design decisions behind the refactor from `trl.SFTTrainer` to the current custom loop, including the rationale for consolidated checkpoints, the unified collator, the evaluator registry, and the auto-batch-size probe.
