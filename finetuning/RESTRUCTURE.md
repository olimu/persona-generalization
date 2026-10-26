# IMPLEMENTATION INSTRUCTIONS

> **Read this document, then implement ONLY the next uncompleted step.**
>
> 1. Read this entire document (especially Sections 14-15)
> 2. Find the next step marked `[ ]` in the progress tracker below
> 3. Implement ONLY that one step
> 4. Mark it `[x]` when done
> 5. Add any notes for future steps in the "Notes from previous steps" section
> 6. Update RESTRUCTURE.md with current state, then stop and wait
>
> **Progress:**
> - [x] Step 1: Foundation — `finetune/config.py` + `finetune/collator.py`
> - [x] Step 2: Data pipeline — `finetune/data.py`
> - [x] Step 3: Model loading — `finetune/model.py`
> - [x] Step 4: Evaluators — `finetune/evaluators.py`
> - [x] Step 5: Callbacks — `finetune/callbacks.py`
> - [x] Step 6: Training engine — `finetune/trainer.py`
> - [x] Step 7: CLI scripts + integration + cleanup
>
> **Notes from previous steps:**
> - **Step 1 (config.py + collator.py)**: Both files created at `finetuning/finetune/config.py` and `finetuning/finetune/collator.py`. Config dataclasses match the spec exactly. `load_config()` constructs nested dataclasses from YAML dict sections, with `_validate()` checking all constraints listed in the plan. `LMCollator` is a direct extraction of the padding logic from the existing `Trainer._create_collator()`. The existing `trainer.py` `TrainingConfig` dataclass will be superseded by the new `Config.training` in Step 6. Not tested locally — code runs on a remote machine.
> - **Step 2 (data.py)**: Complete rewrite at `finetuning/finetune/data.py`. Kept class hierarchy (BaseDataset ABC, CPTDataset, SFTDataset, MultipleChoiceDataset) and `create_dataset()` factory. Replaced `as_training_data()`/`as_evaluation_data()` with single `tokenize()` method. Removed: `accelerator` parameter, `_process_parallel()`, `_apply_chat_template_parallel()`, `.pt` file support, `dataset_type` property. Simplified tokenization to plain tqdm loops. Kept all three helpers (`_apply_chat_template_handling_system`, `_longest_common_prefix_len`, `_warn_on_heavy_prompt_truncation`). `max_length` is now a required positional arg (not optional). `create_dataset()` uses dict registry + `**kwargs` passthrough. No torch import needed — removed. Not tested locally — code runs on a remote machine.
> - **Step 3 (model.py)**: Rewrote `finetuning/finetune/model.py`. Single `load_model(config: ModelConfig, lora_config: LoRAConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]` function. Uses `resolve_dtype()` from config.py. Builds `model_kwargs` dict with `torch_dtype`, `trust_remote_code`, `revision`, and optional `attn_implementation` (Flash Attention 2). Adapter merging branch loads base model via `PeftConfig`, applies adapter, calls `merge_and_unload()`. LoRA branch uses `get_peft_model()` with configurable `target_modules`. Removed: `load_in_8bit`, `device` parameter, all `.to(device)` calls, old `LoRAConfig` dataclass, dict return format. Peft imports are lazy (inside branches) to avoid hard dependency when not using LoRA/adapters. Kept `hf_olmo` try/except for OLMo model registration. Not tested locally — code runs on a remote machine.
> - **Step 4 (evaluators.py)**: Created `finetuning/finetune/evaluators.py` replacing old `evaluator.py`. Registry pattern with `@register_evaluator("name")` decorator populating `EVALUATOR_REGISTRY` dict. `Evaluator` ABC with single `evaluate(model, tokenizer) -> Dict[str, float]` method. All 6 evaluators registered: `perplexity`, `multiple_choice`, `max_eigenvalue_sharpness`, `sum_eigenvalues_sharpness`, `directional_sharpness`, `distance`. `PerplexityEvaluator` uses `LMCollator` + `DataLoader` (via `datasets.Dataset.from_dict()`) replacing inline padding. `MultipleChoiceEvaluator` flattens choices into sequential list, batches with `LMCollator` directly, splits losses back by n_choices — same math as old code. Sharpness evaluators keep same logic using `prepare_batches_for_sharpness()` from `utils/sharpness.py`. `DistanceEvaluator` unchanged in logic. All evaluators accept `(dataset, batch_size, max_length)` with `max_length` defaulting to `dataset.max_length` when None. `create_evaluator()` factory creates dataset via `create_dataset()`, looks up class in registry, passes `eval_config.params` and extra `**kwargs`. Removed: `dataset_type` checks (property no longer exists), `as_evaluation_data()` calls (replaced with `tokenize()`), `debug_print_batch` import. Not tested locally — code runs on a remote machine.
> - **Step 5 (callbacks.py)**: Rewrote `finetuning/finetune/callbacks.py`. Imports `Evaluator` from new `evaluators.py` (was old `evaluator.py`). `IntervalCallback` renamed `update()` → `run()` per spec. `EvaluationCallback` now takes `name: str` and prefixes logged metrics as `eval/{name}/{key}`; injects metrics into Trainer log stream via `on_log()` so wandb picks them up automatically. `BestModelCallback` now takes `output_dir` and saves to `{output_dir}/best/` via `trainer.save_model()` instead of holding state_dict in RAM — fixes the OOM risk on large models; accesses trainer via `kwargs.get("trainer")` in hook methods. New `ThroughputCallback(model_num_params)` computes tokens/sec from `state.num_input_tokens_seen` delta between `on_log` calls, computes MFU via `estimate_mfu()`. New `SystemMetricsCallback` logs GPU memory/utilization/power; guarded by `state.is_world_process_zero` to avoid N×N metrics in distributed training; uses `pynvml` with try/except for graceful degradation. `estimate_mfu()` helper and `GPU_PEAK_FLOPS_BF16` table (A100, H100, A6000, L40) live in this file per Section 14.1. Not tested locally — code runs on a remote machine.
> - **Step 6 (trainer.py)**: Complete rewrite at `finetuning/finetune/trainer.py`. Three public functions: `create_training_args(config, num_processes)` maps Config to HF `TrainingArguments` (computes `gradient_accumulation_steps` from `batch_size / (device_batch_size * num_processes)`, infers bf16/fp16 from `mixed_precision` or `model.dtype`, passes through FSDP config, sets `include_num_input_tokens_seen=True` for ThroughputCallback). `build_callbacks(config, model, tokenizer)` creates ThroughputCallback, SystemMetricsCallback, EvaluationCallbacks for each intermediate eval, and BestModelCallback if `best_model_metric` is set (parses `"eval_name:metric_key"` format, finds matching eval config, creates separate evaluator instance). `run_training(config)` is the main orchestration function — callable from CLI or programmatically — following the 14-step flow from the spec: Accelerator init, seeding, wandb init on main process (with full config), model/tokenizer loading, dataset creation via `Dataset.from_dict(dataset.tokenize())`, LMCollator, callbacks, TrainingArguments, Trainer creation, `trainer.train()` with optional `resume_from_checkpoint`, final evals, saving (model via `trainer.save_model()`, training_log.json, eval_results.json, config.yaml), wandb finish, and returns `{log_history, eval_results, best_model}`. Removed: `trl` dependency (uses `transformers.Trainer`), old `TrainingConfig` dataclass, `Trainer` wrapper class, `_create_collator()`, `_replace_model_forward_with_debug()`, `debug.py` import. No manual device placement — Trainer handles it. Not tested locally — code runs on a remote machine.
> - **Step 7 (CLI + integration + cleanup)**: Rewrote `scripts/train.py` (~10 lines: `load_config` + `run_training`) and `scripts/evaluate.py` (~15 lines: `load_config` + `load_model` + iterate `final_evals` with `create_evaluator`). Updated `finetune/__init__.py` with clean public API exports: `Config`, `load_config`, `BaseDataset`, `create_dataset`, `load_model`, `run_training`, `Evaluator`, `EVALUATOR_REGISTRY`, `create_evaluator`, `register_evaluator`, `LMCollator`. Updated `pyproject.toml` dependencies: removed `ray[data]`, updated `transformers>=4.36.0`, added `peft`, `accelerate`, `datasets`, `safetensors`, `scipy`, `pyyaml`, `jinja2` as core deps; added `[wandb]` and `[metrics]` optional dependency groups. Deleted old `finetune/evaluator.py` (replaced by `evaluators.py`). Updated `evaluate_sharpness.py` import from `evaluator` to `evaluators`. Created 3 example YAML configs in `examples/`: `sft_minimal.yaml` (bare minimum), `sft_full.yaml` (LoRA + wandb + evals + best model tracking), `cpt.yaml` (continued pretraining). Note: standalone scripts `evaluate_sharpness.py` and `compute_gradient.py` still use old `load_model`/`create_dataset` API signatures from Steps 2-3; they will need separate updates to work with the new module interfaces. No `trl` imports remain anywhere in the codebase. Not tested locally — code runs on a remote machine.

---

# Finetuning Library Restructure Plan

## Table of Contents

1. [Current System Summary](#1-current-system-summary)
2. [Critical Analysis](#2-critical-analysis)
3. [Redesigned Specification](#3-redesigned-specification)
4. [Architecture Overview](#4-architecture-overview)
5. [Module Design](#5-module-design)
6. [Configuration System](#6-configuration-system)
7. [Data Pipeline](#7-data-pipeline)
8. [Model Loading](#8-model-loading)
9. [Training Engine](#9-training-engine)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Metrics and Logging](#11-metrics-and-logging)
12. [Checkpointing](#12-checkpointing)
13. [Key Design Decisions](#13-key-design-decisions)
14. [Revision: Issues and Simplifications](#14-revision-issues-and-simplifications)
15. [Implementation Plan](#15-implementation-plan)

---

## 1. Current System Summary

### What the library does

The current finetuning library provides a YAML-configured pipeline for fine-tuning causal language models. It supports:

- **Training modes**: Continued pretraining (CPT) and supervised finetuning (SFT)
- **Adaptation methods**: Full finetuning and LoRA (via PEFT)
- **Evaluation**: Perplexity, multiple-choice accuracy, loss-landscape sharpness metrics (Hessian eigenvalues/trace), parameter distance from initialization
- **Data formats**: JSONL files with `{text}` (CPT), `{prompt, completion}` (SFT), `{prompt, choices, label}` (multiple-choice eval)
- **Data export**: 30+ HuggingFace dataset exporters to JSONL
- **Distributed training**: Multi-GPU via `accelerate` (DDP only)

### Current file structure

```
finetuning/
  finetune/
    trainer.py      # Trainer class wrapping trl.SFTTrainer
    model.py        # Model/tokenizer loading, LoRA setup
    data.py         # BaseDataset + CPT/SFT/MultipleChoice datasets
    evaluator.py    # Perplexity, MC, sharpness, distance evaluators
    callbacks.py    # EvaluationCallback, BestModelCallback
    debug.py        # Terminal visualization of tokenized data
  scripts/
    train.py        # Main training entry point (~400 lines, monolithic)
    evaluate.py     # Standalone evaluation script
    evaluate_sharpness.py
    compute_gradient.py
    filter_data.py
    export_data.py  # HuggingFace dataset exporters
    compute_tokenized_dataset_loss.py
  utils/
    sharpness.py    # Hessian-vector products, eigenvalue computation
```

### How it works (data flow)

```
YAML config
  -> train.py parses into flat Args dataclass
  -> load_model() returns {model, tokenizer}
  -> create_dataset() returns BaseDataset subclass (tokenizes eagerly)
  -> Trainer wraps trl.SFTTrainer with custom collator
  -> SFTTrainer.train() runs the loop
  -> Callbacks run evaluations at intervals
  -> Results saved as JSON files
```

---

## 2. Critical Analysis

### What works well

- **Separation of concerns**: Data, model, trainer, and evaluator are distinct modules
- **Abstract base classes**: `BaseDataset` and `Evaluator` provide extensibility points
- **Custom collator**: Handles variable-length sequences correctly with proper label masking
- **Distributed data processing**: `_process_parallel` distributes tokenization across ranks
- **Data export framework**: Clean abstract exporter pattern with 30+ dataset implementations

### Fundamental problems

#### 2.1 SFTTrainer is the wrong abstraction

The library wraps `trl.SFTTrainer` but immediately disables most of its features (`save_strategy='no'`, `eval_strategy='no'`, `report_to='none'`, `skip_prepare_dataset=True`). SFTTrainer is designed for opinionated SFT workflows, but the library wants full control. This creates:

- Unnecessary dependency on `trl`
- Workarounds fighting the abstraction (e.g., custom collator to bypass SFTTrainer's data pipeline)
- Inability to customize the training loop (e.g., custom loss functions, gradient manipulation)

**Fix**: Use `transformers.Trainer` directly, or a minimal custom training loop for maximum control.

#### 2.2 No modern parallelism support

- No FSDP (Fully Sharded Data Parallel) support
- No Flash Attention 2
- No `torch.compile` integration that actually works cleanly (current impl requires post-hoc safetensors key fixups)
- No tensor parallelism or pipeline parallelism

#### 2.3 No wandb or real logging

- `report_to='none'` is hardcoded
- Training metrics are dumped to `training.json` after the fact
- No real-time visibility into training runs
- No throughput metrics (tokens/sec, MFU, GPU utilization)

#### 2.4 Monolithic entry point

`train.py` is ~400 lines that mix:
- Config parsing and dtype conversion
- Experiment infrastructure (`experiments.runlib`)
- Model loading
- Dataset creation with duplicate code for intermediate/final evals
- Callback wiring
- Training
- Model saving with torch.compile workarounds
- Result serialization

This makes it hard to use the library programmatically or extend it.

#### 2.5 Flat, unvalidated configuration

- Single flat `Args` dataclass with `None` defaults for required fields
- No schema validation
- No hierarchical grouping
- dtype conversion repeated in every script
- No support for config inheritance/composition

#### 2.6 No checkpointing during training

- `save_strategy='no'` means a crash loses all progress
- No resume-from-checkpoint support
- Best model tracking copies entire state_dict to CPU RAM (OOM risk for large models)

#### 2.7 Eager data processing

- All data is tokenized in `BaseDataset.__init__()`, stored in memory as Python lists
- No streaming or lazy loading
- No support for pre-tokenized datasets in efficient formats (Arrow, memmap)
- Separate handling of `.jsonl` and `.pt` with duplicated logic

#### 2.8 Duplicated padding/batching logic

The same left-padding + label masking logic appears in:
- `Trainer._create_collator()`
- `PerplexityEvaluator.evaluate()`
- `MultipleChoiceEvaluator.evaluate()`
- `prepare_batches_for_sharpness()`

#### 2.9 Tight coupling to experiment infrastructure

`from experiments.runlib import get_project_config, get_experiment_config, check_exists_gs` is imported unconditionally in `train.py` and `evaluate.py`, making the library unusable without that external package.

#### 2.10 Missing training features

- No gradient clipping (max_grad_norm)
- No early stopping
- No learning rate finder
- No mixed precision configuration (inferred from model dtype, not configurable)
- No optimizer selection (AdamW only, hardcoded in Trainer)
- No weight tying options
- No configurable attention implementation

---

## 3. Redesigned Specification

### Core principles

1. **Simple over clever**: Fewer abstractions, each one earning its place. Three similar lines of code are better than a premature abstraction.
2. **Composable**: Each component (data, model, trainer, evaluator, logger) is independently usable and replaceable.
3. **Config-driven**: A single YAML file fully specifies a training run, with sensible defaults for everything.
4. **HuggingFace-native**: Build on `transformers.Trainer` (not SFTTrainer), leverage the HF ecosystem rather than fighting it.
5. **Observable**: First-class wandb integration with comprehensive training, evaluation, and throughput metrics.
6. **Resilient**: Configurable checkpointing with resume support. A crash should never lose more than one checkpoint interval of work.
7. **Modern**: FSDP, Flash Attention 2, torch.compile, bf16/fp16 mixed precision as first-class citizens.

### What the library provides

```
finetune train config.yaml          # Train a model
finetune evaluate config.yaml       # Evaluate a model
finetune export --dataset NAME ...  # Export HF datasets to JSONL
```

That's it. Three commands. Everything else is configuration.

### Supported workflows

| Workflow | Description |
|----------|-------------|
| **CPT** (Continued Pretraining) | Next-token prediction on raw text |
| **SFT** (Supervised Finetuning) | Prompt-completion with label masking on prompt tokens |
| **Full finetuning** | All parameters trainable |
| **LoRA** | Low-rank adaptation via PEFT |
| **Multi-GPU (DDP)** | Data-distributed parallel via accelerate |
| **Multi-GPU (FSDP)** | Fully sharded data parallel via accelerate |

### Non-goals (for now, but the architecture should not prevent future support)

- RLHF / DPO / PPO (reward-model training, preference optimization)
- Multi-node training
- Tensor/pipeline parallelism
- Sequence packing
- Curriculum learning

---

## 4. Architecture Overview

### New file structure

```
finetuning/
  finetune/
    config.py         # Hierarchical config dataclasses + YAML loading
    data.py           # Dataset classes (CPT, SFT, MultipleChoice)
    collator.py       # Unified padding/collation (single source of truth)
    model.py          # Model + tokenizer loading, LoRA, Flash Attention
    trainer.py        # Thin wrapper around transformers.Trainer
    evaluators.py     # Evaluator base + implementations
    callbacks.py      # Evaluation, checkpointing, metrics callbacks
    metrics.py        # Throughput tracker (tokens/sec, MFU, GPU stats)
    logging.py        # Wandb + console + JSON logging facade
  scripts/
    train.py          # CLI entry point (thin: parse config, wire components, train)
    evaluate.py       # CLI entry point for standalone evaluation
    export_data.py    # Dataset export (kept as-is, it's well-designed)
  utils/
    sharpness.py      # Hessian utilities (kept as-is)
```

### Data flow (redesigned)

```
YAML config
  -> config.py parses + validates into typed, hierarchical Config
  -> model.py loads model with Flash Attention 2 / FSDP-ready
  -> data.py creates datasets (same pattern, cleaner interface)
  -> collator.py provides unified collation
  -> trainer.py creates transformers.Trainer with:
      - Custom callbacks for evaluation, metrics, logging
      - Wandb integration
      - Configurable checkpointing
  -> Trainer.train() runs (with optional resume_from_checkpoint)
  -> logging.py streams metrics to wandb + console + JSON
  -> Results saved
```

### Key difference from current design

The current design puts all orchestration logic in `scripts/train.py` (400 lines). The new design moves orchestration into a `run_training(config)` function that can be called from anywhere -- a script, a notebook, or another library. The scripts become thin CLI wrappers.

---

## 5. Module Design

### 5.1 `config.py` — Hierarchical Configuration

Replace the flat `Args` dataclass with grouped, validated config:

```python
@dataclass
class ModelConfig:
    path: str                                     # HF model ID or local path
    revision: Optional[str] = None
    dtype: str = "bfloat16"                       # "float32", "float16", "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    compile: bool = False
    trust_remote_code: bool = True

@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: str = "all-linear"

@dataclass
class DataConfig:
    train_path: str                               # Path to training data
    train_type: str = "sft"                       # "cpt" or "sft"
    train_subset: Optional[int] = None
    max_length: int = 2048
    apply_chat_template: bool = False
    chat_template: Optional[str] = None           # Custom Jinja2 template

@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 32                          # Global (effective) batch size
    device_batch_size: int = 4                    # Per-device micro batch size
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"                  # Any HF scheduler name
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"                # Any HF optimizer name
    seed: Optional[int] = None
    mixed_precision: Optional[str] = None         # "bf16", "fp16", or null (auto from dtype)
    gradient_checkpointing: bool = False
    fsdp: Optional[str] = None                    # FSDP config string or null
    fsdp_config: Optional[dict] = None            # FSDP config dict

@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_strategy: str = "epoch"                  # "steps", "epoch", "no"
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = 3
    resume_from: Optional[str] = None             # Path to resume from
    output_dir: str = "./output"

@dataclass
class EvalTaskConfig:
    name: str
    dataset_path: str
    dataset_type: str                             # "cpt", "sft", "multiple_choice"
    evaluator: str                                # "perplexity", "multiple_choice", etc.
    batch_size: int = 32
    max_length: Optional[int] = None
    subset: Optional[int] = None
    # For intermediate evals:
    eval_every: Optional[int] = None              # Steps between evals
    eval_mode: str = "steps"                      # "steps" or "epochs"
    # Evaluator-specific params:
    params: Optional[dict] = None                 # e.g., {"num_samples": 20} for sharpness

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    log_every_n_steps: int = 1
    log_throughput: bool = True                    # tokens/sec, MFU, etc.
    log_system: bool = True                       # GPU memory, utilization, power

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    intermediate_evals: List[EvalTaskConfig] = field(default_factory=list)
    final_evals: List[EvalTaskConfig] = field(default_factory=list)
    best_model_metric: Optional[str] = None       # e.g., "eval_name:log_perplexity"
```

**Key improvements**:
- Grouped by concern (model, data, training, checkpoint, logging)
- Sensible defaults for everything -- a minimal config only needs `model.path`, `data.train_path`, and `checkpoint.output_dir`
- Validated at load time (e.g., `batch_size % (device_batch_size * num_gpus) == 0`)
- dtype parsed once, used everywhere

**Loading**:
```python
def load_config(path: str) -> Config:
    """Load and validate a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Construct nested dataclasses from nested dict
    config = Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        ...
    )
    _validate(config)
    return config
```

### 5.2 `collator.py` — Unified Collation

One collator to rule them all. Currently, padding logic is duplicated in 4 places.

```python
class LMCollator:
    """Pads/truncates variable-length sequences for causal LM training and evaluation."""

    def __init__(self, pad_token_id: int, max_length: int, padding_side: str = "left"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Input:  list of {"input_ids": [...], "attention_mask": [...], "labels": [...]}
        Output: batched tensors, padded to min(max_length, longest_in_batch)
        """
        batch_max_len = min(self.max_length, max(len(x["input_ids"]) for x in examples))

        input_ids, attention_mask, labels = [], [], []
        for x in examples:
            ids = x["input_ids"][:batch_max_len]
            mask = x.get("attention_mask", [1] * len(ids))[:len(ids)]
            labs = x["labels"][:len(ids)]
            pad_len = batch_max_len - len(ids)

            if self.padding_side == "left":
                ids = [self.pad_token_id] * pad_len + ids
                mask = [0] * pad_len + mask
                labs = [-100] * pad_len + labs
            else:
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
                labs = labs + [-100] * pad_len

            input_ids.append(torch.tensor(ids))
            attention_mask.append(torch.tensor(mask))
            labels.append(torch.tensor(labs))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
```

This replaces:
- `Trainer._create_collator()` in trainer.py
- Inline padding in `PerplexityEvaluator.evaluate()`
- Inline padding in `MultipleChoiceEvaluator.evaluate()`
- `prepare_batches_for_sharpness()` in sharpness.py

### 5.3 `data.py` — Dataset Classes

Keep the current `BaseDataset` / `CPTDataset` / `SFTDataset` / `MultipleChoiceDataset` pattern, but simplify:

**Changes from current**:
- Remove `.pt` file support (rarely used, adds complexity)
- Remove `accelerator` parameter -- tokenization parallelism moves to a simpler approach using `multiprocessing` or HF `datasets.map()`
- `as_training_data()` and `as_evaluation_data()` return HuggingFace `Dataset` objects directly (not raw dicts), so they integrate cleanly with `Trainer`
- Factory function `create_dataset()` stays

```python
class BaseDataset(ABC):
    def __init__(self, path: str, tokenizer, max_length: int,
                 shuffle: bool = False, subset: Optional[int] = None,
                 apply_chat_template: bool = False, seed: Optional[int] = None):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle = shuffle
        self.subset = subset
        self.apply_chat_template = apply_chat_template
        self.seed = seed

        self.raw_data = self._load()
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.raw_data)
        if subset is not None:
            self.raw_data = self.raw_data[:subset]

    @abstractmethod
    def _load(self) -> List[Dict]:
        """Load raw data from path."""
        ...

    @abstractmethod
    def tokenize(self) -> Dataset:
        """Return a HuggingFace Dataset with columns: input_ids, attention_mask, labels."""
        ...

def create_dataset(path, dataset_type, tokenizer, **kwargs) -> BaseDataset:
    types = {"cpt": CPTDataset, "sft": SFTDataset, "multiple_choice": MultipleChoiceDataset}
    return types[dataset_type](path, tokenizer, **kwargs)
```

### 5.4 `model.py` — Model Loading

Simplify the branching logic and add Flash Attention 2 support:

```python
def load_model(config: ModelConfig, lora_config: LoRAConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from config."""
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[config.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        config.path, revision=config.revision, trust_remote_code=config.trust_remote_code
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    # Check if path contains a LoRA adapter to merge
    adapter_path = os.path.join(config.path, "adapter_config.json")
    if os.path.exists(adapter_path):
        peft_config = PeftConfig.from_pretrained(config.path)
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, config.path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)

    # Apply LoRA if configured
    if lora_config.enabled:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        peft_cfg = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
        model = get_peft_model(model, peft_cfg)

    if config.compile:
        model = torch.compile(model)

    return model, tokenizer
```

**Key changes**:
- `attn_implementation="flash_attention_2"` passed to `from_pretrained()`
- No manual device placement (Trainer/accelerate handles this)
- No 8-bit quantization support (use FSDP instead for memory savings)
- Returns tuple instead of dict
- Much less branching

### 5.5 `trainer.py` — Training Engine

Replace `trl.SFTTrainer` with `transformers.Trainer`:

```python
from transformers import Trainer, TrainingArguments

def create_training_args(config: Config, num_processes: int) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from our config."""
    tc = config.training
    cc = config.checkpoint
    lc = config.logging

    micro_batch = tc.device_batch_size * num_processes
    grad_accum = tc.batch_size // micro_batch

    return TrainingArguments(
        output_dir=cc.output_dir,

        # Optimization
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        max_grad_norm=tc.max_grad_norm,
        optim=tc.optimizer,
        lr_scheduler_type=tc.lr_scheduler,
        warmup_ratio=tc.warmup_ratio,
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=tc.gradient_checkpointing,

        # Precision
        bf16=(tc.mixed_precision == "bf16" or config.model.dtype == "bfloat16"),
        fp16=(tc.mixed_precision == "fp16" or config.model.dtype == "float16"),

        # FSDP
        fsdp=tc.fsdp,
        fsdp_config=tc.fsdp_config,

        # Checkpointing
        save_strategy=cc.save_strategy if cc.enabled else "no",
        save_steps=cc.save_steps,
        save_total_limit=cc.save_total_limit,

        # Logging
        logging_strategy="steps",
        logging_steps=lc.log_every_n_steps,
        report_to="wandb" if lc.wandb else "none",
        run_name=lc.wandb_name,

        # Eval (handled by our callbacks, not Trainer's built-in eval)
        eval_strategy="no",

        # Misc
        seed=tc.seed or 42,
        data_seed=tc.seed or 42,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

def run_training(config: Config):
    """Main training function. Can be called from CLI or programmatically."""
    accelerator = Accelerator()

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Load model and tokenizer
    model, tokenizer = load_model(config.model, config.lora)

    # Create dataset
    train_dataset = create_dataset(
        config.data.train_path, config.data.train_type, tokenizer,
        max_length=config.data.max_length,
        shuffle=True, seed=config.training.seed,
        apply_chat_template=config.data.apply_chat_template,
        subset=config.data.train_subset,
    ).tokenize()

    # Create collator
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = LMCollator(pad_token_id, config.data.max_length, padding_side="left")

    # Build callbacks
    callbacks = build_callbacks(config, model, tokenizer)

    # Create Trainer
    training_args = create_training_args(config, accelerator.num_processes)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train (with optional resume)
    trainer.train(resume_from_checkpoint=config.checkpoint.resume_from)

    # Run final evaluations
    run_final_evals(config, model, tokenizer)

    # Save final model
    trainer.save_model(config.checkpoint.output_dir)
```

**Key changes**:
- Uses `transformers.Trainer` directly (not SFTTrainer)
- `run_training(config)` is a callable function, not buried in a script
- FSDP support via `TrainingArguments.fsdp`
- Gradient clipping via `max_grad_norm`
- Checkpointing via `save_strategy`
- Resume via `resume_from_checkpoint`
- wandb via `report_to="wandb"`
- Gradient checkpointing via `gradient_checkpointing`
- Configurable optimizer via `optim`

### 5.6 `evaluators.py` — Evaluation Framework

Keep the current abstract pattern but use the unified collator:

```python
class Evaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Run evaluation and return metrics dict."""
        ...

class PerplexityEvaluator(Evaluator):
    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        collator = LMCollator(
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            max_length=self.max_length,
            padding_side="left",
        )
        eval_data = self.dataset.tokenize()
        dataloader = DataLoader(eval_data, batch_size=self.batch_size, collate_fn=collator)

        total_loss = 0.0
        total_correct = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)

                num_tokens = batch["labels"].ne(-100).sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_correct += (outputs.logits[:, :-1].argmax(-1) == batch["labels"][:, 1:]).float().sum().item()
                total_tokens += num_tokens

        return {
            "log_perplexity": total_loss / total_tokens,
            "per_token_accuracy": total_correct / total_tokens,
        }
```

**Evaluator registry** (replaces the if/elif chain in `create_evaluator`):

```python
EVALUATOR_REGISTRY = {
    "perplexity": PerplexityEvaluator,
    "multiple_choice": MultipleChoiceEvaluator,
    "max_eigenvalue_sharpness": MaxEigenvalueSharpnessEvaluator,
    "sum_eigenvalues_sharpness": SumEigenvaluesSharpnessEvaluator,
    "directional_sharpness": DirectionalSharpnessEvaluator,
    "distance": DistanceEvaluator,
}

def create_evaluator(eval_config: EvalTaskConfig, tokenizer, **kwargs) -> Evaluator:
    cls = EVALUATOR_REGISTRY[eval_config.evaluator]
    dataset = create_dataset(eval_config.dataset_path, eval_config.dataset_type, tokenizer, ...)
    return cls(dataset, eval_config.batch_size, eval_config.max_length, **(eval_config.params or {}))
```

This is extensible -- users can register custom evaluators:
```python
EVALUATOR_REGISTRY["my_evaluator"] = MyEvaluatorClass
```

### 5.7 `callbacks.py` — Callbacks

Keep `EvaluationCallback` and `BestModelCallback`, but add:

```python
class ThroughputCallback(TrainerCallback):
    """Logs tokens/sec, samples/sec, and MFU at each logging step."""

    def __init__(self, model_num_params: int, log_fn: Callable):
        self.model_num_params = model_num_params
        self.log_fn = log_fn
        self._step_start_time = None
        self._step_tokens = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._step_start_time is None:
            return
        elapsed = time.time() - self._step_start_time
        tokens_per_sec = self._step_tokens / max(elapsed, 1e-6)

        # MFU: model_flops_per_token * tokens_per_sec / peak_gpu_flops
        # (approximate: 6 * num_params for transformer forward+backward)
        model_flops = 6 * self.model_num_params * self._step_tokens
        peak_flops = _get_gpu_peak_flops()
        mfu = model_flops / (elapsed * peak_flops) if peak_flops > 0 else 0

        self.log_fn({
            "throughput/tokens_per_sec": tokens_per_sec,
            "throughput/mfu": mfu,
        })


class SystemMetricsCallback(TrainerCallback):
    """Logs GPU memory, utilization, and power at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return

        metrics = {}
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1e9
            mem_total = torch.cuda.get_device_properties(i).total_mem / 1e9
            metrics[f"system/gpu{i}_memory_gb"] = mem_used
            metrics[f"system/gpu{i}_memory_pct"] = mem_used / mem_total * 100

        # nvidia-smi based metrics (utilization, power) via pynvml if available
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                metrics[f"system/gpu{i}_utilization_pct"] = util.gpu
                metrics[f"system/gpu{i}_power_w"] = power
        except Exception:
            pass

        self.log_fn(metrics)
```

### 5.8 `metrics.py` — Metrics Tracking

A thin abstraction for computing and accumulating metrics during training:

```python
@dataclass
class TrainingMetrics:
    """Accumulated training metrics for the current logging interval."""
    loss_sum: float = 0.0
    num_tokens: int = 0
    num_steps: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.num_tokens, 1)

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.num_tokens / max(elapsed, 1e-6)

    def reset(self):
        self.loss_sum = 0.0
        self.num_tokens = 0
        self.num_steps = 0
        self.start_time = time.time()

def estimate_mfu(num_params: int, tokens_per_second: float, gpu_name: str = None) -> float:
    """Estimate Model FLOP Utilization.

    MFU = actual_flops / theoretical_peak_flops
    actual_flops ~= 6 * num_params * tokens_per_second (forward + backward)
    """
    actual_flops = 6 * num_params * tokens_per_second
    peak_flops = get_gpu_peak_flops(gpu_name)
    return actual_flops / peak_flops if peak_flops > 0 else 0.0

GPU_PEAK_FLOPS = {
    "A100": 312e12,       # bf16
    "A100-80GB": 312e12,
    "H100": 989e12,       # bf16
    "A6000": 155e12,      # bf16 (approximate)
    "RTX 4090": 165e12,   # bf16
    "L40": 181e12,        # bf16
}
```

### 5.9 `logging.py` — Logging Facade

```python
class Logger:
    """Unified logger that writes to console, JSON, and optionally wandb."""

    def __init__(self, config: LoggingConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._history = []

        if config.wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                tags=config.wandb_tags,
            )

    def log(self, metrics: Dict[str, float], step: int):
        """Log metrics to all sinks."""
        entry = {"step": step, **metrics}
        self._history.append(entry)

        # Console
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        print(f"[step {step}] {', '.join(parts)}", flush=True)

        # Wandb
        if self.config.wandb:
            import wandb
            wandb.log(metrics, step=step)

    def save(self):
        """Save full history to JSON."""
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self._history, f)

    def finish(self):
        """Finalize logging (close wandb, save JSON)."""
        self.save()
        if self.config.wandb:
            import wandb
            wandb.finish()
```

---

## 6. Configuration System

### Minimal config (SFT with defaults)

```yaml
model:
  path: meta-llama/Llama-3.1-8B-Instruct

data:
  train_path: data/train.jsonl
  train_type: sft
  max_length: 2048

training:
  epochs: 3
  batch_size: 32
  device_batch_size: 4
  learning_rate: 2e-5

checkpoint:
  output_dir: ./output/my_run
```

Everything else uses sensible defaults (cosine scheduler, 0.1 warmup, 1.0 grad clip, no wandb, epoch-based checkpointing).

### Full config (all options)

```yaml
model:
  path: meta-llama/Llama-3.1-8B-Instruct
  revision: main
  dtype: bfloat16
  attn_implementation: flash_attention_2
  compile: false
  trust_remote_code: true

lora:
  enabled: true
  r: 16
  alpha: 32.0
  dropout: 0.05
  target_modules: all-linear

data:
  train_path: data/train.jsonl
  train_type: sft
  train_subset: 10000
  max_length: 4096
  apply_chat_template: true
  chat_template: null  # uses model's default

training:
  epochs: 3
  batch_size: 128
  device_batch_size: 4
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler: cosine
  max_grad_norm: 1.0
  optimizer: adamw_torch
  seed: 42
  gradient_checkpointing: false
  fsdp: "full_shard auto_wrap"
  fsdp_config:
    fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

checkpoint:
  enabled: true
  save_strategy: steps
  save_steps: 500
  save_total_limit: 3
  resume_from: null
  output_dir: ./output/my_run

logging:
  wandb: true
  wandb_project: finetuning
  wandb_name: llama-sft-run1
  wandb_tags: [sft, llama, lora]
  log_every_n_steps: 10
  log_throughput: true
  log_system: true

intermediate_evals:
  - name: val_perplexity
    dataset_path: data/val.jsonl
    dataset_type: sft
    evaluator: perplexity
    batch_size: 32
    eval_every: 200
    eval_mode: steps

  - name: mmlu_accuracy
    dataset_path: data/mmlu.jsonl
    dataset_type: multiple_choice
    evaluator: multiple_choice
    batch_size: 16
    eval_every: 1
    eval_mode: epochs

final_evals:
  - name: test_perplexity
    dataset_path: data/test.jsonl
    dataset_type: sft
    evaluator: perplexity
    batch_size: 32

  - name: test_mc
    dataset_path: data/test_mc.jsonl
    dataset_type: multiple_choice
    evaluator: multiple_choice
    batch_size: 16

best_model_metric: val_perplexity:log_perplexity  # save best by this metric
```

### CPT config

```yaml
model:
  path: meta-llama/Llama-3.1-8B

data:
  train_path: data/corpus.jsonl
  train_type: cpt
  max_length: 2048

training:
  epochs: 1
  batch_size: 64
  device_batch_size: 8
  learning_rate: 5e-5

checkpoint:
  output_dir: ./output/cpt_run
```

---

## 7. Data Pipeline

### Data formats (unchanged from current)

| Format | Type | Schema |
|--------|------|--------|
| CPT | JSONL | `{"text": "raw text..."}` |
| CPT (chat) | JSONL | `{"text": [{"role": "user", "content": "..."}, ...]}` |
| SFT | JSONL | `{"prompt": [...messages...], "completion": [...messages...]}` |
| Multiple Choice | JSONL | `{"prompt": [...], "choices": {"A": [...], "B": [...]}, "label": "A"}` |

These are well-designed formats. No changes needed.

### Processing pipeline

```
JSONL file
  -> _load(): read lines, validate schema
  -> tokenize(): tokenize text, build input_ids/attention_mask/labels
  -> Return HuggingFace Dataset (columnar, efficient, memory-mappable)
```

The key change is returning `datasets.Dataset` objects instead of raw dicts-of-lists. This:
- Integrates natively with `transformers.Trainer`
- Enables memory mapping for large datasets
- Provides built-in shuffling and batching

### Chat template handling

Keep `_apply_chat_template_handling_system()` -- it handles the common case where models don't support system messages gracefully. Keep `_longest_common_prefix_len()` for efficient SFT label construction.

### Data export (`export_data.py`)

**No changes needed.** The export framework is well-designed:
- Clean abstract exporter pattern
- 30+ dataset implementations
- Good CLI interface
- Train/val splitting support

---

## 8. Model Loading

### Flash Attention 2 support

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # NEW
)
```

This is a one-line change that provides significant speedups (2-4x) and memory savings. Falls back gracefully if flash-attn is not installed.

### FSDP compatibility

FSDP requires models to NOT be moved to a device before wrapping. The current code does `model.to(device)` in `load_model()`. The new code removes manual device placement entirely -- Trainer/accelerate handles it.

### Removed features

- **8-bit quantization** (`load_in_8bit`): Replaced by FSDP for memory savings. If users need quantization, they can use `bitsandbytes` config through HF's `BitsAndBytesConfig` -- but this is a separate concern and should not be in the core model loading.
- **Manual device placement**: Trainer handles this.
- **torch.compile key fixup hack**: Fixed properly by saving through Trainer's built-in `save_model()`.

---

## 9. Training Engine

### Why `transformers.Trainer` instead of SFTTrainer

| Feature | SFTTrainer | Trainer |
|---------|-----------|---------|
| Custom data pipeline | Must disable with `skip_prepare_dataset` | Native support via `data_collator` |
| FSDP | Via accelerate config | Native `fsdp` parameter |
| Checkpointing | Full support | Full support |
| Resume | Full support | Full support |
| wandb | Full support | Full support |
| Custom loss | Difficult | Override `compute_loss()` |
| Gradient clipping | Yes | Yes |
| Mixed precision | Yes | Yes |
| Extra dependency | `trl` | None |

We were already bypassing SFTTrainer's data pipeline and disabling its eval/save strategies. Moving to Trainer removes the `trl` dependency and gives us direct access to all features we need.

### Why not a fully custom training loop?

A custom loop would give maximum flexibility, but:
- We'd have to reimplement distributed training coordination
- We'd lose resume-from-checkpoint for free
- We'd lose wandb integration for free
- We'd lose learning rate scheduling for free
- We'd have to handle gradient accumulation manually

`transformers.Trainer` gives us all of this with a clean API. We can always override `compute_loss()` for custom loss functions, and use callbacks for everything else.

### FSDP configuration

FSDP is configured through `TrainingArguments`:

```python
TrainingArguments(
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    },
)
```

Or through an `accelerate` config file for more control.

---

## 10. Evaluation Framework

### Design: evaluators as plugins

The current evaluator design (abstract base + concrete implementations) is good. We keep it, but improve:

1. **Registry pattern**: Replace if/elif chain with a dict lookup
2. **Unified collation**: All evaluators use `LMCollator`
3. **DataLoader**: Use `torch.utils.data.DataLoader` instead of manual batching
4. **Extensibility**: Users can register custom evaluators

### Built-in evaluators

| Evaluator | Metrics | Dataset types |
|-----------|---------|---------------|
| `perplexity` | `log_perplexity`, `per_token_accuracy` | CPT, SFT |
| `multiple_choice` | `log_perplexity`, `relative_log_perplexity`, `accuracy` | MultipleChoice |
| `max_eigenvalue_sharpness` | `max_eigenvalue_sharpness` | CPT, SFT |
| `sum_eigenvalues_sharpness` | `sum_eigenvalue_sharpness` | CPT, SFT |
| `directional_sharpness` | `directional_sharpness` | CPT, SFT |
| `distance` | `parameter_distances`, `model_distance` | Any |

### Adding custom evaluators

```python
class MyCustomEvaluator(Evaluator):
    def __init__(self, dataset, batch_size, max_length, my_param=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.my_param = my_param

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        # ... custom evaluation logic ...
        return {"my_metric": 0.95}

# Register it
EVALUATOR_REGISTRY["my_evaluator"] = MyCustomEvaluator
```

Then in YAML:
```yaml
final_evals:
  - name: my_eval
    evaluator: my_evaluator
    params:
      my_param: 100
```

---

## 11. Metrics and Logging

### What gets logged

Every logging step (configurable via `log_every_n_steps`):

**Training metrics** (from Trainer):
- `train/loss` -- average loss over logging interval
- `train/learning_rate` -- current learning rate
- `train/epoch` -- current epoch (fractional)
- `train/global_step` -- total optimizer steps taken
- `train/grad_norm` -- gradient norm (if `max_grad_norm` is set)

**Throughput metrics** (from `ThroughputCallback`):
- `throughput/tokens_per_sec` -- tokens processed per second
- `throughput/samples_per_sec` -- samples processed per second
- `throughput/mfu` -- Model FLOP Utilization (vs theoretical peak)

**System metrics** (from `SystemMetricsCallback`, optional):
- `system/gpu{i}_memory_gb` -- GPU memory used (GB)
- `system/gpu{i}_memory_pct` -- GPU memory used (%)
- `system/gpu{i}_utilization_pct` -- GPU compute utilization
- `system/gpu{i}_power_w` -- GPU power draw (watts)

**Evaluation metrics** (from `EvaluationCallback`, at eval intervals):
- `eval/{eval_name}/log_perplexity`
- `eval/{eval_name}/per_token_accuracy`
- `eval/{eval_name}/accuracy`
- ... any metrics returned by the evaluator

### Where metrics go

1. **Console**: Printed at each logging step
2. **Wandb**: Logged in real-time (if enabled)
3. **JSON**: `metrics.json` saved at end of training (full history)
4. **HF Trainer log_history**: Available via `trainer.state.log_history`

### Wandb integration

When `logging.wandb: true`:
- Training config logged as `wandb.config`
- All metrics logged in real-time
- Model artifacts optionally saved
- Run tagged and named for easy filtering

---

## 12. Checkpointing

### Checkpoint strategy

| Strategy | Behavior |
|----------|----------|
| `"no"` | No checkpointing (current behavior, not recommended) |
| `"epoch"` | Save at end of each epoch |
| `"steps"` | Save every `save_steps` optimizer steps |

### What gets saved

Each checkpoint (via `transformers.Trainer.save_model()`):
- Model weights (safetensors)
- Tokenizer
- Training state (optimizer, scheduler, step count)
- `trainer_state.json` (for resume)

### Resume from checkpoint

```yaml
checkpoint:
  resume_from: ./output/my_run/checkpoint-500
```

This resumes training from step 500, restoring:
- Model weights
- Optimizer state
- Learning rate schedule position
- Random number generator states

### Best model tracking

When `best_model_metric` is set:
- `BestModelCallback` runs the specified evaluator at each evaluation interval
- Tracks the best score and saves the corresponding checkpoint
- At end of training, the best checkpoint is identified
- Works with `save_total_limit` -- best checkpoint is never deleted

### Disk space management

`save_total_limit: 3` keeps only the 3 most recent checkpoints (plus the best one if tracking). This prevents disk space from growing unboundedly.

---

## 13. Key Design Decisions

### Decision 1: transformers.Trainer over custom loop

**Rationale**: Trainer provides 90% of what we need (distributed training, checkpointing, resume, wandb, gradient accumulation, mixed precision, FSDP) with zero boilerplate. The remaining 10% (custom evaluation, throughput metrics) is cleanly handled by callbacks. A custom loop would require reimplementing all of this.

**Trade-off**: We're coupled to HF's Trainer API. But since we're already coupled to HF for models and tokenizers, this is an acceptable trade-off.

### Decision 2: Hierarchical config with dataclasses (not Hydra/OmegaConf)

**Rationale**: Hydra adds complexity (config groups, multirun, sweeps) that we don't need. Plain dataclasses with `yaml.safe_load()` are simple, type-checked, and debuggable. If config composition is needed later, we can add it without changing the interface.

**Trade-off**: No automatic config sweeps or composition. But the config is simple enough that this isn't needed -- a bash loop over YAML files works fine.

### Decision 3: Keep eager tokenization (for now)

**Rationale**: Lazy tokenization (on-the-fly in the DataLoader) would save memory but complicate the code. For typical finetuning datasets (10K-1M examples), eager tokenization is fast and fits in memory. We can add lazy loading later if needed, behind the same `BaseDataset` interface.

**Trade-off**: Large datasets (>10M examples) may require streaming. But this is a future concern -- the architecture supports adding it.

### Decision 4: Unified collator as single source of truth

**Rationale**: The same padding logic currently exists in 4 places. A single `LMCollator` class is used everywhere -- training, evaluation, and sharpness computation. This eliminates bugs from inconsistent padding behavior.

### Decision 5: Evaluator registry pattern

**Rationale**: The current `create_evaluator()` function uses an if/elif chain that must be modified to add new evaluators. A registry dict allows:
- Adding evaluators without modifying core code
- External evaluators (user-defined)
- Clean YAML-to-class mapping

### Decision 6: Remove experiment infrastructure coupling

**Rationale**: `from experiments.runlib import ...` makes the library unusable without that external package. The `check_exists` functionality (GCS, local) is experiment management, not finetuning. It should be handled by the calling script, not the library.

**Migration**: Users who need `check_exists` behavior can wrap `run_training()`:
```python
if not check_exists(output_path):
    run_training(config)
```

### Decision 7: Optional everything

Every feature beyond "load model, load data, train, save" is optional:
- wandb: off by default
- checkpointing: on by default but can be disabled
- intermediate evals: empty list by default
- final evals: empty list by default
- LoRA: disabled by default
- FSDP: disabled by default
- Flash Attention: enabled by default (but graceful fallback)
- throughput metrics: on by default
- system metrics: on by default

This means the simplest possible config is ~10 lines of YAML.

---

## 14. Revision: Issues and Simplifications

After thorough review of the Step 1 plan against the actual codebase, the following issues were identified and addressed.

### 14.1 Architectural Simplifications

**Remove `logging.py` module.** `transformers.Trainer` already integrates with wandb when `report_to="wandb"` is set. Training metrics (loss, learning rate, grad_norm) are automatically logged. A separate Logger class that also calls `wandb.log()` would cause duplicate entries. Instead:
- Training metrics: handled by Trainer natively
- Evaluation metrics: logged via Trainer's logging inside EvaluationCallback
- Final results: saved as JSON files (same as current)

**Remove `metrics.py` module.** The `TrainingMetrics` dataclass tracks values that Trainer already tracks internally. The `estimate_mfu()` function (~15 lines) and GPU peak FLOPS table belong in `callbacks.py` where they're used.

### 14.2 Bug Fixes in the Step 1 Plan

**ThroughputCallback has no token counting mechanism.** The plan shows `self._step_tokens` but never explains how it gets populated. Fix: use Trainer's `state.num_input_tokens_seen` to compute tokens/sec, or count non-(-100) tokens from the batch's labels tensor in callback hooks.

**BestModelCallback copies full state_dict to CPU RAM.** For a 7B model, this is ~14GB of RAM. For larger models, this will OOM. Fix: save best model to `{output_dir}/best/` directory using `trainer.save_model()`. Track only the best score and step. Load back only when needed for final evaluation.

**SystemMetricsCallback logs all GPUs from every rank.** In distributed training, this creates N×N metrics. Fix: only log from the main process (`state.is_world_process_zero`).

### 14.3 Data Pipeline Simplifications

**Remove accelerator-based parallel tokenization.** The custom `_process_parallel()` using `accelerate.gather_object()` adds complexity for marginal speedup on typical finetuning datasets (10K-1M examples). Replace with simple tqdm loops. For future scaling, HuggingFace `datasets.map(num_proc=...)` is a more standard approach.

**Remove `.pt` file support.** Rarely used and adds branching logic throughout data.py. JSONL is the standard format.

**Keep `tokenize()` returning `Dict[str, List]` instead of HF Dataset.** The Step 1 plan proposed returning HF `Dataset` objects, but MultipleChoiceDataset has a nested structure (per-choice input_ids) that doesn't map to a flat HF Dataset. Keeping dict-of-lists as the return type is simpler and consistent across all dataset types. Convert to HF Dataset only in the training flow: `Dataset.from_dict(dataset.tokenize())`.

### 14.4 Other Changes

- **Keep `debug.py` unchanged** — Small, useful for development, no dependencies on other modules.
- **Keep `utils/sharpness.py` as-is** — Complex but correct. The sharpness evaluators wrap it cleanly.
- **Keep legacy scripts as optional** — `compute_gradient.py`, `compute_tokenized_dataset_loss.py`, `evaluate_sharpness.py`, and `filter_data.py` are not part of the core restructure. They can remain as standalone utilities and be updated later.
- **Add concrete config validation** — Validate: batch_size divisibility, valid dtype strings, valid scheduler names, required fields set, eval_every > 0, etc.

### 14.5 Final File Structure

```
finetuning/
  finetune/
    __init__.py         # Exports: load_config, run_training, create_dataset, etc.
    config.py           # Config dataclasses + YAML loading + validation
    collator.py         # LMCollator (single source of truth for padding)
    data.py             # BaseDataset + CPT/SFT/MultipleChoice + create_dataset()
    model.py            # load_model() with Flash Attention 2 + LoRA
    trainer.py          # run_training() + create_training_args()
    evaluators.py       # Evaluator ABC + 6 implementations + registry
    callbacks.py        # Eval, BestModel, Throughput, SystemMetrics callbacks
    debug.py            # Token visualization (unchanged)
  scripts/
    train.py            # CLI: load config -> run_training()
    evaluate.py         # CLI: load config -> run evaluations
    export_data.py      # Dataset exporters (unchanged)
  utils/
    sharpness.py        # Hessian utilities (unchanged)
```

**8 core files** to implement (config, collator, data, model, trainer, evaluators, callbacks, \_\_init\_\_) + **2 scripts** (train.py, evaluate.py). Removed vs current: `logging.py` (unnecessary), `metrics.py` (folded into callbacks). Renamed: `evaluator.py` → `evaluators.py`.

---

## 15. Implementation Plan

Implementation is split into 7 self-contained steps. Each step produces working, importable code. Steps must be done in order due to dependencies.

---

### Step 1: Foundation — `config.py` + `collator.py`

**Files to create:** `finetune/config.py`, `finetune/collator.py`

**Dependencies:** None. These are leaf modules with no imports from the finetuning package.

#### `config.py`

All dataclasses from Section 5.1. Key constants:

```python
DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

VALID_LR_SCHEDULERS = [
    "linear", "cosine", "cosine_with_restarts", "polynomial",
    "constant", "constant_with_warmup", "inverse_sqrt",
]
```

Dataclasses (all with `@dataclass`):

| Class | Required fields | Key defaults |
|-------|----------------|--------------|
| `ModelConfig` | `path: str` | `dtype="bfloat16"`, `attn_implementation="flash_attention_2"`, `compile=False`, `trust_remote_code=True` |
| `LoRAConfig` | *(none)* | `enabled=False`, `r=8`, `alpha=16.0`, `dropout=0.0`, `target_modules="all-linear"` |
| `DataConfig` | `train_path: str` | `train_type="sft"`, `max_length=2048`, `apply_chat_template=False` |
| `TrainingConfig` | *(none)* | `epochs=1`, `batch_size=32`, `device_batch_size=4`, `learning_rate=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`, `lr_scheduler="cosine"`, `max_grad_norm=1.0`, `optimizer="adamw_torch"`, `gradient_checkpointing=False` |
| `CheckpointConfig` | *(none)* | `enabled=True`, `save_strategy="epoch"`, `save_total_limit=3`, `output_dir="./output"` |
| `EvalTaskConfig` | `name: str`, `dataset_path: str`, `dataset_type: str`, `evaluator: str` | `batch_size=32`, `eval_mode="steps"` |
| `LoggingConfig` | *(none)* | `wandb=False`, `log_every_n_steps=1`, `log_throughput=True`, `log_system=True` |
| `Config` | `model: ModelConfig`, `data: DataConfig` | All others have `field(default_factory=...)` |

Functions:

- **`load_config(path: str) -> Config`**: Read YAML with `yaml.safe_load()`, construct nested dataclasses. Handle missing sections gracefully (use defaults). Parse `intermediate_evals` and `final_evals` as lists of `EvalTaskConfig`.

- **`_validate(config: Config)`**: Raise `ValueError` for:
  - `config.model.dtype not in DTYPE_MAP`
  - `config.data.train_type not in ("cpt", "sft")`
  - `config.training.lr_scheduler not in VALID_LR_SCHEDULERS`
  - `config.training.batch_size <= 0` or `device_batch_size <= 0`
  - `config.training.epochs <= 0`
  - `config.training.learning_rate < 0`
  - `config.checkpoint.save_strategy == "steps"` but `save_steps is None`
  - Any eval task with `eval_mode not in ("steps", "epochs")`

- **`resolve_dtype(dtype_str: str) -> torch.dtype`**: Simple lookup in `DTYPE_MAP`, raise on unknown.

#### `collator.py`

Single class, exactly as specified in Section 5.2:

```python
class LMCollator:
    def __init__(self, pad_token_id: int, max_length: int, padding_side: str = "left"): ...
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]: ...
```

Logic: Pad each example to `min(max_length, max_batch_len)`. For left padding: prepend `pad_token_id` to input_ids, `0` to attention_mask, `-100` to labels. Stack into tensors.

#### Verification

- `from finetune.config import load_config, Config, resolve_dtype` imports cleanly
- `from finetune.collator import LMCollator` imports cleanly
- A sample YAML loads into a fully populated `Config` object
- `LMCollator` correctly pads a list of variable-length examples into uniform tensors

---

### Step 2: Data Pipeline — `data.py`

**Files to create:** `finetune/data.py` (complete rewrite)

**Dependencies:** Step 1 (config.py for type hints only — no runtime import needed)

#### Implementation

Keep the class hierarchy but simplify significantly:

```python
class BaseDataset(ABC):
    def __init__(self, path: str, tokenizer, max_length: int,
                 shuffle: bool = False, subset: Optional[int] = None,
                 apply_chat_template: bool = False, seed: Optional[int] = None):
        self.raw_data = self._load()
        # shuffle with seeded RNG, then subset

    @abstractmethod
    def _load(self) -> List[Dict]: ...

    @abstractmethod
    def tokenize(self) -> Dict[str, List]: ...
```

**CPTDataset:**
- `_load()`: Read JSONL, validate each entry has exactly `{"text"}` key
- `tokenize()`:
  - If `apply_chat_template`: use `_apply_chat_template_handling_system()` for each example
  - Else: `tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]`
  - `labels = input_ids` (full sequence is target for causal LM)
  - `attention_mask = [1] * len(ids)` for each example
  - Return `{"input_ids": [...], "attention_mask": [...], "labels": [...]}`

**SFTDataset:**
- `_load()`: Read JSONL, validate `{"prompt", "completion"}` keys
- `tokenize()`:
  - Tokenize prompts and completions separately (with or without chat template)
  - Use `_longest_common_prefix_len()` to find prompt/completion boundary
  - Warn on heavy truncation via `_warn_on_heavy_prompt_truncation()`
  - `input_ids = completion_ids` (full sequence)
  - `labels = [-100] * prefix_len + completion_ids[prefix_len:]` (mask prompt tokens)
  - Return `{"input_ids": [...], "attention_mask": [...], "labels": [...]}`

**MultipleChoiceDataset:**
- `_load()`: Read JSONL, validate `{"prompt", "choices", "label"}` keys
- `tokenize()`: Return nested structure:
  - `input_ids`: `List[Dict[str, List[int]]]` — per-example dict mapping choice keys to token sequences
  - `attention_mask`: same nesting
  - `labels`: same nesting (prompt tokens masked with -100)
  - `answer_labels`: `List[str]` — correct choice key per example

**Helper functions** (keep from current code):
- `_apply_chat_template_handling_system(tokenizer, messages, ...)` — tries chat template, removes system messages if unsupported
- `_longest_common_prefix_len(a, b)` — for SFT prompt masking
- `_warn_on_heavy_prompt_truncation(prefix_lens, prompt_lens, ...)` — warn if >20% truncated

**Factory:**
```python
def create_dataset(path, dataset_type, tokenizer, **kwargs) -> BaseDataset:
    types = {"cpt": CPTDataset, "sft": SFTDataset, "multiple_choice": MultipleChoiceDataset}
    return types[dataset_type](path, tokenizer, **kwargs)
```

#### Key changes from current code

- **Removed:** `accelerator` parameter, `_process_parallel()`, `_apply_chat_template_parallel()`, `.pt` file support, `dataset_type` property, `as_training_data()` / `as_evaluation_data()` split
- **Added:** Single `tokenize()` method replacing both `as_training_data()` and `as_evaluation_data()`
- **Simplified:** Tokenization uses simple `tqdm` list comprehensions instead of distributed processing

#### Verification

- `CPTDataset("test.jsonl", tokenizer, 2048).tokenize()` returns correct format
- `SFTDataset` properly masks prompt tokens with -100 in labels
- `MultipleChoiceDataset` returns nested per-choice structure
- `create_dataset()` factory dispatches correctly

---

### Step 3: Model Loading — `model.py`

**Files to create:** `finetune/model.py` (rewrite)

**Dependencies:** Step 1 (imports `ModelConfig`, `LoRAConfig`, `resolve_dtype` from config.py)

#### Implementation

Single function:

```python
def load_model(config: ModelConfig, lora_config: LoRAConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
```

Logic:
1. `dtype = resolve_dtype(config.dtype)`
2. Load tokenizer: `AutoTokenizer.from_pretrained(config.path, revision=config.revision, trust_remote_code=config.trust_remote_code)`
3. Build `model_kwargs = {"torch_dtype": dtype, "trust_remote_code": ..., "revision": ...}`
4. If `config.attn_implementation` is set: add `"attn_implementation"` to kwargs
5. If `adapter_config.json` exists at `config.path`: load base model from PeftConfig, merge adapter with `merge_and_unload()`
6. Else: `AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)`
7. If `lora_config.enabled`: apply LoRA via `peft.get_peft_model()` with `LoraConfig(r=..., lora_alpha=..., lora_dropout=..., target_modules=...)`
8. If `config.compile`: `model = torch.compile(model)`
9. Return `(model, tokenizer)`

#### Key differences from current

- Takes config dataclasses, not loose parameters
- Returns `(model, tokenizer)` tuple, not dict
- **No** `model.to(device)` — Trainer/accelerate handles device placement
- **No** `load_in_8bit` — use FSDP for memory savings instead
- **No** device parameter
- `attn_implementation` passed to `from_pretrained()` for Flash Attention 2

#### Verification

- Can load a model and tokenizer from a HuggingFace path
- Model stays on CPU (no `.to()` calls)
- LoRA wrapping works when `lora_config.enabled=True`
- Flash Attention kwarg is passed through correctly

---

### Step 4: Evaluators — `evaluators.py`

**Files to create:** `finetune/evaluators.py` (new file, replaces `evaluator.py`)

**Dependencies:** Steps 1-2 (imports `LMCollator` from collator.py, `BaseDataset`/`create_dataset` from data.py, `EvalTaskConfig` from config.py). Also imports from `utils/sharpness.py` for sharpness evaluators.

#### Implementation

**Registry pattern:**
```python
EVALUATOR_REGISTRY = {}

def register_evaluator(name):
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator
```

**Base class:**
```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]: ...
```

**Built-in evaluators** (each decorated with `@register_evaluator("name")`):

| Name | Class | Init params | Returns |
|------|-------|-------------|---------|
| `"perplexity"` | `PerplexityEvaluator` | `dataset, batch_size, max_length` | `{log_perplexity, per_token_accuracy}` |
| `"multiple_choice"` | `MultipleChoiceEvaluator` | `dataset, batch_size, max_length` | `{log_perplexity, relative_log_perplexity, accuracy}` |
| `"max_eigenvalue_sharpness"` | `MaxEigenvalueSharpnessEvaluator` | `dataset, batch_size, max_length` | `{max_eigenvalue_sharpness}` |
| `"sum_eigenvalues_sharpness"` | `SumEigenvaluesSharpnessEvaluator` | `dataset, batch_size, max_length, num_samples=20` | `{sum_eigenvalue_sharpness}` |
| `"directional_sharpness"` | `DirectionalSharpnessEvaluator` | `dataset, batch_size, max_length, direction` | `{directional_sharpness}` |
| `"distance"` | `DistanceEvaluator` | `dataset, batch_size, max_length, initial_model_state, normalize=False` | `{parameter_distances, model_distance}` |

**Key improvement: Use `LMCollator` + `DataLoader` for batching in PerplexityEvaluator and MultipleChoiceEvaluator** instead of inline padding logic. This eliminates the 4-place padding duplication.

For `PerplexityEvaluator`:
```python
collator = LMCollator(pad_token_id, max_length, padding_side="left")
eval_dataset = Dataset.from_dict(self.dataset.tokenize())
dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, collate_fn=collator)
# iterate batches, compute loss + accuracy
```

For `MultipleChoiceEvaluator`: Flatten all choices into a sequential list, batch with LMCollator, compute per-choice losses, split back by number of choices per example. Same math as current implementation.

Sharpness and distance evaluators: Keep same logic as current, using `utils/sharpness.py` utilities.

**Factory function:**
```python
def create_evaluator(eval_config: EvalTaskConfig, tokenizer, **kwargs) -> Evaluator:
    cls = EVALUATOR_REGISTRY[eval_config.evaluator]
    dataset = create_dataset(
        eval_config.dataset_path, eval_config.dataset_type, tokenizer,
        max_length=eval_config.max_length or 2048, subset=eval_config.subset
    )
    return cls(dataset=dataset, batch_size=eval_config.batch_size,
               max_length=eval_config.max_length, **(eval_config.params or {}), **kwargs)
```

#### Verification

- All 6 evaluators are in `EVALUATOR_REGISTRY`
- `PerplexityEvaluator` produces valid loss values using `LMCollator`
- `MultipleChoiceEvaluator` produces accuracy between 0 and 1
- `create_evaluator()` correctly instantiates from `EvalTaskConfig`
- Custom evaluators can be registered with `@register_evaluator("name")`

---

### Step 5: Callbacks — `callbacks.py`

**Files to create:** `finetune/callbacks.py` (rewrite)

**Dependencies:** Steps 1, 4 (imports from config.py, evaluators.py)

#### Implementation

**IntervalCallback** (keep current pattern, clean up):
```python
class IntervalCallback(ABC, TrainerCallback):
    def __init__(self, eval_every: int, eval_mode: str = "steps"):
        # eval_mode in ("steps", "epochs")
    def on_step_end(...): if eval_mode == "steps" and step % eval_every == 0 -> run()
    def on_epoch_end(...): if eval_mode == "epochs" -> run()
    def on_train_begin(...): run(step=0)
    def on_train_end(...): run(step=global_step)
    @abstractmethod
    def run(self, step: int, **kwargs): ...
```

**EvaluationCallback(IntervalCallback):**
- `__init__(self, name, model, tokenizer, evaluator, eval_every, eval_mode)`
- `run()`: evaluate, store results, print, log to Trainer via `kwargs["control"]` or by accessing the trainer instance
- Attributes: `results: List[Dict]`, `steps: List[int]`
- Logging to wandb: in `on_log` or by manually calling the trainer's log method with `{f"eval/{self.name}/{k}": v for k, v in result.items()}`

**BestModelCallback(TrainerCallback):**
- `__init__(self, model, tokenizer, evaluator, metric_key, output_dir, mode="min")`
- `_update()`: evaluate, compare against best, if better: save model to `{output_dir}/best/` using the trainer's `save_model()` method
- Attributes: `best_score`, `best_step`, `best_dir`
- Runs at `on_epoch_end` and `on_train_begin`
- **Does NOT hold state_dict in memory** — saves to disk immediately

**ThroughputCallback(TrainerCallback):**
- `__init__(self, model_num_params: int)`
- Tracks step start time in `on_step_begin`
- In `on_log`: compute `tokens_per_sec` from trainer state, compute `mfu` via `estimate_mfu()`, add to logs dict
- Token counting: use `state.num_input_tokens_seen` if available, or accumulate from batches

**SystemMetricsCallback(TrainerCallback):**
- In `on_log`: only if `state.is_world_process_zero`
- Log GPU memory via `torch.cuda.memory_allocated()`
- Log GPU utilization and power via `pynvml` (with try/except for graceful degradation)

**MFU helper** (in this file):
```python
GPU_PEAK_FLOPS_BF16 = {
    "NVIDIA A100-SXM4-80GB": 312e12,
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA H100": 989e12,
    "NVIDIA A6000": 155e12,
    "NVIDIA L40": 181e12,
}

def estimate_mfu(num_params, tokens_per_sec, num_gpus=1):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    peak = GPU_PEAK_FLOPS_BF16.get(gpu_name, 0)
    actual = 6 * num_params * tokens_per_sec
    return actual / (peak * num_gpus) if peak > 0 else 0.0
```

#### Verification

- `EvaluationCallback` correctly runs evaluator at specified intervals and stores results
- `BestModelCallback` saves to `{output_dir}/best/` on improvement (NOT to RAM)
- `ThroughputCallback` produces reasonable tokens/sec values
- `SystemMetricsCallback` only runs on main process

---

### Step 6: Training Engine — `trainer.py`

**Files to create:** `finetune/trainer.py` (rewrite)

**Dependencies:** All previous steps (imports from config, collator, data, model, evaluators, callbacks)

#### Implementation

**`create_training_args(config: Config, num_processes: int) -> TrainingArguments`:**

Map our config to HuggingFace `TrainingArguments`:
- `output_dir` = `config.checkpoint.output_dir`
- Optimization: `learning_rate`, `weight_decay`, `max_grad_norm`, `optim`, `lr_scheduler_type`, `warmup_ratio`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps` (computed from `batch_size / (device_batch_size * num_processes)`), `gradient_checkpointing`
- Precision: `bf16` and `fp16` from `config.training.mixed_precision` or inferred from `config.model.dtype`
- FSDP: `fsdp`, `fsdp_config` (pass through from config)
- Checkpointing: `save_strategy`, `save_steps`, `save_total_limit` (respect `config.checkpoint.enabled`)
- Logging: `logging_strategy="steps"`, `logging_steps`, `report_to="wandb"` or `"none"`
- Eval: `eval_strategy="no"` (we handle eval via callbacks)
- Misc: `seed`, `data_seed`, `remove_unused_columns=False`, `dataloader_pin_memory=True`

**`build_callbacks(config: Config, model, tokenizer) -> List[TrainerCallback]`:**

1. If `config.logging.log_throughput`: add `ThroughputCallback(num_params)`
2. If `config.logging.log_system`: add `SystemMetricsCallback()`
3. For each `config.intermediate_evals`: create evaluator, add `EvaluationCallback`
4. If `config.best_model_metric`: parse `"eval_name:metric_key"`, find eval config, add `BestModelCallback`

**`run_training(config: Config) -> Dict`:**

Main orchestration function — can be called from CLI or programmatically:

1. **Init accelerator**: `Accelerator()` for rank info
2. **Seed**: `set_seed(config.training.seed)` if set
3. **wandb**: init on main process if `config.logging.wandb` (pass full config as `wandb.config`)
4. **Load model + tokenizer**: `load_model(config.model, config.lora)`, set `tokenizer.padding_side = "left"`
5. **Create dataset**: `create_dataset(...)` then `Dataset.from_dict(dataset.tokenize())`
6. **Create collator**: `LMCollator(pad_token_id, max_length)`
7. **Build callbacks**: `build_callbacks(config, model, tokenizer)`
8. **Create TrainingArguments**: `create_training_args(config, accelerator.num_processes)`
9. **Create Trainer**: `Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator, callbacks=callbacks)`
10. **Train**: `trainer.train(resume_from_checkpoint=config.checkpoint.resume_from)`
11. **Final evaluations**: iterate `config.final_evals`, create evaluators, run, collect results
12. **Save** (main process only): model via `trainer.save_model()`, training log as JSON, eval results as JSON, config as YAML
13. **wandb finish**
14. Return results dict

#### Key design points

- Uses `transformers.Trainer`, NOT `trl.SFTTrainer`
- `run_training(config)` is a library function, not a script
- FSDP, checkpointing, resume, wandb all handled via `TrainingArguments`
- No manual device placement or dtype handling — Trainer does it
- No `torch.compile` safetensors key fixup hack — `trainer.save_model()` handles it correctly

#### Verification

- `run_training(config)` executes end-to-end with a small model and dataset
- Uses `transformers.Trainer` (verify: no `trl` import)
- Checkpoints saved at configured intervals
- wandb receives metrics when enabled
- Results files saved to `output_dir`

---

### Step 7: CLI Scripts + Integration + Cleanup

**Files to modify/create:**
- `scripts/train.py` (rewrite — thin wrapper)
- `scripts/evaluate.py` (rewrite — thin wrapper)
- `finetune/__init__.py` (update exports)
- `setup.py` or `pyproject.toml` (update dependencies)

#### `scripts/train.py`

```python
import sys
import torch
from finetuning.finetune.config import load_config
from finetuning.finetune.trainer import run_training

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)
    torch.set_float32_matmul_precision("high")
    config = load_config(sys.argv[1])
    run_training(config)
```

That's it. ~10 lines.

#### `scripts/evaluate.py`

```python
import sys, json, torch
from finetuning.finetune.config import load_config
from finetuning.finetune.model import load_model
from finetuning.finetune.evaluators import create_evaluator

if __name__ == "__main__":
    config = load_config(sys.argv[1])
    model, tokenizer = load_model(config.model, config.lora)
    tokenizer.padding_side = "left"
    model.eval()
    results = {}
    for eval_task in config.final_evals:
        evaluator = create_evaluator(eval_task, tokenizer)
        results[eval_task.name] = evaluator.evaluate(model, tokenizer)
    with open(sys.argv[2] if len(sys.argv) > 2 else "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
```

#### `finetune/__init__.py`

```python
from .config import Config, load_config
from .data import BaseDataset, create_dataset
from .model import load_model
from .trainer import run_training
from .evaluators import Evaluator, EVALUATOR_REGISTRY, create_evaluator, register_evaluator
from .collator import LMCollator
```

#### Dependency updates

- **Remove**: `trl` (no longer needed)
- **Keep**: `transformers>=4.36.0`, `torch>=2.0.0`, `peft`, `accelerate`, `datasets`, `safetensors`, `scipy`, `numpy`, `tqdm`, `pyyaml`, `jinja2`
- **Add**: `wandb` (optional), `pynvml` (optional, for GPU metrics)

#### Cleanup

- Delete old `finetune/evaluator.py` (replaced by `evaluators.py`)
- The old trainer.py and data.py are replaced by the new versions
- Keep `scripts/export_data.py` unchanged
- Keep `finetune/debug.py` unchanged
- Keep `utils/sharpness.py` unchanged
- Keep standalone scripts (`compute_gradient.py`, etc.) — they don't import from the restructured modules

#### Final verification

- `python scripts/train.py examples/sft_config.yaml` runs end-to-end
- `python scripts/evaluate.py examples/eval_config.yaml` runs evaluations
- `from finetuning.finetune import run_training, load_config` works
- No `trl` import anywhere in the codebase
- All evaluation types still work (perplexity, multiple_choice, sharpness, distance)
- Create 2-3 example YAML configs: minimal SFT, full SFT with wandb, CPT config
