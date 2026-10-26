"""Hierarchical configuration system for finetuning.

Grouped dataclasses, loaded from YAML via ``load_config``. Validated on
construction so that callers can trust a ``Config`` object is internally
consistent before training begins.

Design notes
============
* ``warmup_ratio`` is gone. Warmup is always expressed in absolute
  ``warmup_steps`` — callers (or the stage layer) compute the step count
  from a ratio if they want one.
* ``lr_scheduler`` is one of ``constant | cosine | wsd``. WSD takes
  explicit ``wsd_stable_steps`` / ``wsd_decay_steps``. Cosine takes
  ``wsd_decay_steps`` as the length of its decay phase.
* ``CheckpointConfig`` has no ``save_strategy`` / ``save_steps`` /
  ``save_total_limit`` / ``resume_from``. Saving is driven by
  ``save_at_epochs`` / ``save_at_steps`` / ``save_final`` plus an
  explicit toggle for ``optimizer.pt`` / ``trainer_state.pt`` — a
  ``SavePolicyCallback`` in the loop owns dispatch. Cross-phase reloads
  (trunk → branch) are expressed via ``base_model_checkpoint`` + the
  ``reload`` selector dict.
* ``parallelism`` is ``single | ddp | fsdp``. The FSDP path uses FSDP2
  (``fully_shard``); there is no FSDP1 config knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import yaml


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

VALID_LR_SCHEDULERS = ("constant", "cosine", "wsd")
VALID_PARALLELISM = ("single", "ddp", "fsdp")
VALID_RELOAD_KEYS = ("model", "optimizer", "scheduler", "ema", "trainer_state")


@dataclass
class ModelConfig:
    path: str = ""
    tokenizer_path: Optional[str] = None
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    compile: bool = False
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: str = "all-linear"
    layers_to_transform: Optional[List[int]] = None


@dataclass
class TrainableConfig:
    """Selective parameter trainability — sets ``requires_grad`` per-param.

    When ``enabled``, each entry in ``modules`` is resolved against the
    loaded model. The lookup tries ``model.get_submodule(name)`` first
    (selecting every parameter under that module, recursively); failing
    that, ``model.get_parameter(name)`` (selecting that one parameter).
    Names that resolve to neither raise at apply time.

    Modes:
        ``"include"``: freeze every parameter, then unfreeze those resolved.
        ``"exclude"``: train every parameter, then freeze those resolved.

    Mutually exclusive with ``lora.enabled=True`` — both manipulate
    ``requires_grad`` and combining them is intentionally forbidden.
    """
    enabled: bool = False
    mode: Literal["include", "exclude"] = "include"
    modules: List[str] = field(default_factory=list)


@dataclass
class DataStageConfig:
    """A single dataset stage in a multi-stage training run.

    Stages run sequentially under one ``run_training()`` call sharing
    one optimizer + one continuous LR scheduler whose total horizon is
    the sum of every stage's ``epochs * steps_per_epoch``.
    """
    train_path: str = ""
    epochs: int = 1
    train_subset: Optional[int] = None
    train_subset_seed: Optional[int] = None


@dataclass
class DataConfig:
    # Multi-stage training spec. When non-empty, the trainer iterates
    # these stages in order; the legacy ``train_path`` /
    # ``training.epochs`` / legacy subset fields are forbidden. When
    # empty, ``_validate`` synthesizes a single-element ``stages`` list
    # from the legacy fields for backward compat.
    stages: List[DataStageConfig] = field(default_factory=list)

    train_path: str = ""
    train_type: str = "sft"
    train_subset: Optional[int] = None
    # Seed for the deterministic random subset selection. Pinned per
    # experiment, NOT per training run — changing ``training.seed``
    # should not change which N examples we train on. Falls back to a
    # deterministic file-order prefix when ``train_subset`` is set
    # without a seed (legacy behavior).
    train_subset_seed: Optional[int] = None
    max_length: int = 2048
    apply_chat_template: bool = False
    chat_template: Optional[str] = None


@dataclass
class TrainingConfig:
    # Duration. One of (epochs, max_steps) is authoritative; if max_steps
    # is set it overrides epochs for loop termination.
    epochs: int = 1
    max_steps: Optional[int] = None

    # Batch shape.  ``device_batch_size`` may be ``"auto"`` to probe GPU
    # memory at startup and select the largest feasible value.
    batch_size: int = 32
    device_batch_size: Union[int, str] = 4
    dataloader_drop_last: bool = True

    # Optimizer.
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Mixed precision. None = fp32 training.
    mixed_precision: Optional[str] = "bf16"
    gradient_checkpointing: bool = False
    max_memory_fraction: float = 0.95
    seed: Optional[int] = 42

    # Scheduler. See module docstring for the WSD / cosine / constant
    # conventions. ``warmup_steps`` is always in absolute optimizer steps.
    # ``warmup_ratio`` is a convenience fallback: if ``warmup_steps`` is 0
    # and ``warmup_ratio`` is set, the trainer converts
    # ``warmup_steps = round(warmup_ratio * max_steps)`` at scheduler
    # build time. This exists solely so legacy experiment YAMLs that only
    # know about ``warmup_ratio`` keep training with the intended warmup.
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    wsd_stable_steps: Optional[int] = None
    wsd_decay_steps: Optional[int] = None
    wsd_min_lr_fraction: float = 0.1

    # Parallelism. ``num_gpus`` is consulted by ``finetuning.scripts.train``
    # when it auto-relaunches under ``torchrun --standalone`` for multi-GPU
    # parallelism. ``None`` = autodetect from ``CUDA_VISIBLE_DEVICES`` or
    # ``torch.cuda.device_count()``; ignored when already running under
    # torchrun (the parent ``torchrun`` invocation owns the world size).
    parallelism: str = "single"
    num_gpus: Optional[int] = None


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    update_every: int = 1
    device: str = "cpu"  # "cpu" or "gpu"


@dataclass
class CheckpointConfig:
    output_dir: str = "./output"

    # Save schedule. If both lists are empty and save_final is False, no
    # checkpoints will be written — validation will flag this.
    save_at_epochs: List[int] = field(default_factory=list)
    save_at_steps: List[int] = field(default_factory=list)
    save_final: bool = True
    save_optimizer_and_trainer_state: bool = True

    # On-disk format for ``save_at_epochs`` / ``save_at_steps`` snapshots.
    # ``"raw"`` (default): ``model.pt`` / ``optimizer.pt`` / etc. — used to
    # resume training. ``"hf"``: a self-contained HuggingFace directory
    # (``config.json`` + ``model.safetensors`` + tokenizer files) per
    # snapshot — directly loadable by vLLM. ``"hf"`` is incompatible with
    # ``save_optimizer_and_trainer_state=True`` (no optimizer state is
    # produced); ``_validate`` flags that.
    save_at_epochs_format: Literal["raw", "hf"] = "raw"

    # Cross-phase reload: point at a consolidated checkpoint directory
    # (containing model.pt, optimizer.pt, ema.pt, etc.). The ``reload``
    # dict mirrors ``state.Reload``'s fields; unknown keys raise.
    base_model_checkpoint: Optional[str] = None
    reload: Dict[str, bool] = field(default_factory=dict)


@dataclass
class EvalTaskConfig:
    name: str = ""
    dataset_path: str = ""
    dataset_type: str = ""
    evaluator: str = ""
    batch_size: int = 32
    max_length: Optional[int] = None
    subset: Optional[int] = None
    eval_every: Optional[int] = None
    eval_mode: str = "steps"
    log_samples: int = 128
    dataset_dir: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    log_every_n_steps: int = 1
    log_throughput: bool = True
    log_system: bool = True
    profiling: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    trainable: TrainableConfig = field(default_factory=TrainableConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    intermediate_evals: List[EvalTaskConfig] = field(default_factory=list)
    final_evals: List[EvalTaskConfig] = field(default_factory=list)
    best_model_metric: Optional[str] = None
    best_model_mode: str = "min"
    float32_matmul_precision: str = "high"
    # Distance tracking — L2 distance from the initial (base) model.
    distance_eval_every: Optional[int] = None
    distance_eval_mode: str = "steps"
    distance_device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype: {dtype_str!r}. Must be one of: {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[dtype_str]


def load_config(path: str) -> Config:
    """Load, deserialize, and validate a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    raw_data = dict(raw.get("data", {}))
    raw_stages = raw_data.pop("stages", None)
    data_cfg = DataConfig(**raw_data)
    if raw_stages:
        data_cfg.stages = [DataStageConfig(**s) for s in raw_stages]

    config = Config(
        model=ModelConfig(**raw.get("model", {})),
        data=data_cfg,
        training=TrainingConfig(**raw.get("training", {})),
        checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        lora=LoRAConfig(**raw.get("lora", {})),
        trainable=TrainableConfig(**raw.get("trainable", {})),
        ema=EMAConfig(**raw.get("ema", {})),
        intermediate_evals=[EvalTaskConfig(**e) for e in raw.get("intermediate_evals", [])],
        final_evals=[EvalTaskConfig(**e) for e in raw.get("final_evals", [])],
        best_model_metric=raw.get("best_model_metric"),
        best_model_mode=raw.get("best_model_mode", "min"),
        float32_matmul_precision=raw.get("float32_matmul_precision", "high"),
        distance_eval_every=raw.get("distance_eval_every"),
        distance_eval_mode=raw.get("distance_eval_mode", "steps"),
        distance_device=raw.get("distance_device", "cpu"),
    )
    _validate(config)
    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(config: Config) -> None:
    # Model
    if not config.model.path:
        raise ValueError("model.path is required")
    if config.model.dtype not in DTYPE_MAP:
        raise ValueError(
            f"model.dtype must be one of {list(DTYPE_MAP.keys())}, got {config.model.dtype!r}"
        )

    # Data
    if config.data.stages:
        if config.data.train_path:
            raise ValueError(
                "data.train_path must be unset when data.stages is given"
            )
        if config.data.train_subset is not None or config.data.train_subset_seed is not None:
            raise ValueError(
                "data.train_subset / data.train_subset_seed must be unset "
                "when data.stages is given — set them per stage"
            )
        for i, s in enumerate(config.data.stages):
            if not s.train_path:
                raise ValueError(f"data.stages[{i}].train_path is required")
            if s.epochs <= 0:
                raise ValueError(
                    f"data.stages[{i}].epochs must be positive, got {s.epochs}"
                )
    else:
        if not config.data.train_path:
            raise ValueError("data.train_path or data.stages is required")
        # Normalize legacy single-stage fields into a one-element stages
        # list so the trainer always reads from ``data.stages``. Clear
        # the legacy fields so re-validation is idempotent.
        config.data.stages = [DataStageConfig(
            train_path=config.data.train_path,
            epochs=config.training.epochs,
            train_subset=config.data.train_subset,
            train_subset_seed=config.data.train_subset_seed,
        )]
        config.data.train_path = ""
        config.data.train_subset = None
        config.data.train_subset_seed = None
    if config.data.train_type not in ("cpt", "sft", "pretokenized"):
        raise ValueError(
            f"data.train_type must be 'cpt', 'sft', or 'pretokenized', "
            f"got {config.data.train_type!r}"
        )
    if config.data.max_length <= 0:
        raise ValueError("data.max_length must be positive")

    # Training
    tc = config.training
    if tc.epochs <= 0 and not tc.max_steps:
        raise ValueError("training.epochs must be positive (or set training.max_steps)")
    if tc.max_steps is not None and tc.max_steps <= 0:
        raise ValueError("training.max_steps must be positive when set")
    if tc.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if isinstance(tc.device_batch_size, str):
        if tc.device_batch_size != "auto":
            raise ValueError(
                f"training.device_batch_size must be a positive int or 'auto', "
                f"got {tc.device_batch_size!r}"
            )
    elif tc.device_batch_size <= 0:
        raise ValueError("training.device_batch_size must be positive")
    if not (0.0 < tc.max_memory_fraction <= 1.0):
        raise ValueError(
            f"training.max_memory_fraction must be in (0, 1], "
            f"got {tc.max_memory_fraction}"
        )
    if tc.learning_rate < 0:
        raise ValueError("training.learning_rate must be non-negative")
    if tc.lr_scheduler not in VALID_LR_SCHEDULERS:
        raise ValueError(
            f"training.lr_scheduler must be one of {VALID_LR_SCHEDULERS}, got {tc.lr_scheduler!r}"
        )
    if tc.parallelism not in VALID_PARALLELISM:
        raise ValueError(
            f"training.parallelism must be one of {VALID_PARALLELISM}, got {tc.parallelism!r}"
        )
    if tc.num_gpus is not None and tc.num_gpus < 1:
        raise ValueError(
            f"training.num_gpus must be >= 1 when set, got {tc.num_gpus}"
        )
    if tc.warmup_steps < 0:
        raise ValueError("training.warmup_steps must be non-negative")
    if tc.lr_scheduler == "wsd":
        if tc.wsd_stable_steps is None or tc.wsd_decay_steps is None:
            raise ValueError(
                "training.wsd_stable_steps and training.wsd_decay_steps are "
                "required when lr_scheduler='wsd'"
            )
        if tc.wsd_stable_steps < 0 or tc.wsd_decay_steps < 0:
            raise ValueError("WSD stable/decay step counts must be non-negative")
    if tc.mixed_precision is not None and tc.mixed_precision not in ("bf16", "fp16"):
        raise ValueError(
            f"training.mixed_precision must be 'bf16', 'fp16', or null, "
            f"got {tc.mixed_precision!r}"
        )

    # Checkpoint
    cc = config.checkpoint
    if not cc.save_at_epochs and not cc.save_at_steps and not cc.save_final:
        raise ValueError(
            "At least one of checkpoint.{save_at_epochs, save_at_steps, save_final} "
            "must be set — otherwise the run produces no outputs."
        )
    if cc.save_at_epochs_format not in ("raw", "hf"):
        raise ValueError(
            f"checkpoint.save_at_epochs_format must be 'raw' or 'hf', "
            f"got {cc.save_at_epochs_format!r}"
        )
    if cc.save_at_epochs_format == "hf" and not cc.save_at_epochs:
        raise ValueError(
            "checkpoint.save_at_epochs_format='hf' requires "
            "checkpoint.save_at_epochs to be non-empty."
        )
    for k in cc.reload:
        if k not in VALID_RELOAD_KEYS:
            raise ValueError(
                f"checkpoint.reload has unknown key {k!r}; allowed: {VALID_RELOAD_KEYS}"
            )
    if cc.base_model_checkpoint is not None and not cc.reload:
        raise ValueError(
            "checkpoint.base_model_checkpoint is set but checkpoint.reload is empty; "
            "specify which pieces to restore (at minimum {model: true})."
        )
    # If ema is requested, model must also be requested (see checkpoint.load_checkpoint).
    if cc.reload.get("ema") and not cc.reload.get("model"):
        raise ValueError(
            "checkpoint.reload.ema=true requires checkpoint.reload.model=true"
        )

    # Eval tasks
    for eval_task in config.intermediate_evals + config.final_evals:
        if not eval_task.name:
            raise ValueError("Each eval task must have a 'name'")
        if not eval_task.evaluator:
            raise ValueError(f"Eval task {eval_task.name!r}: evaluator is required")
        if eval_task.eval_mode not in ("steps", "epochs"):
            raise ValueError(
                f"Eval task {eval_task.name!r}: eval_mode must be 'steps' or 'epochs', "
                f"got {eval_task.eval_mode!r}"
            )
    for eval_task in config.intermediate_evals:
        if eval_task.eval_every is None or eval_task.eval_every <= 0:
            raise ValueError(
                f"Intermediate eval {eval_task.name!r}: eval_every must be a positive integer"
            )

    # Best model
    if config.best_model_mode not in ("min", "max"):
        raise ValueError(
            f"best_model_mode must be 'min' or 'max', got {config.best_model_mode!r}"
        )

    # Distance eval
    if config.distance_eval_every is not None and config.distance_eval_every <= 0:
        raise ValueError("distance_eval_every must be a positive integer")
    if config.distance_eval_mode not in ("steps", "epochs"):
        raise ValueError(
            f"distance_eval_mode must be 'steps' or 'epochs', got {config.distance_eval_mode!r}"
        )
    if config.distance_device not in ("cpu", "gpu"):
        raise ValueError(
            f"distance_device must be 'cpu' or 'gpu', got {config.distance_device!r}"
        )

    # Trainable
    tr = config.trainable
    if tr.mode not in ("include", "exclude"):
        raise ValueError(
            f"trainable.mode must be 'include' or 'exclude', got {tr.mode!r}"
        )
    if tr.enabled:
        if config.lora.enabled:
            raise ValueError(
                "trainable.enabled and lora.enabled cannot both be true — "
                "they both manipulate requires_grad"
            )
        if not isinstance(tr.modules, list):
            raise ValueError("trainable.modules must be a list of strings")
        for i, name in enumerate(tr.modules):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"trainable.modules[{i}] must be a non-empty string, "
                    f"got {name!r}"
                )

    # EMA
    if config.ema.enabled:
        if not (0.0 <= config.ema.decay < 1.0):
            raise ValueError(f"ema.decay must be in [0, 1), got {config.ema.decay}")
        if config.ema.update_every < 1:
            raise ValueError(f"ema.update_every must be >= 1, got {config.ema.update_every}")
        if config.ema.device not in ("cpu", "gpu"):
            raise ValueError(f"ema.device must be 'cpu' or 'gpu', got {config.ema.device!r}")

    # Misc
    if config.float32_matmul_precision not in ("highest", "high", "medium"):
        raise ValueError(
            f"float32_matmul_precision must be 'highest', 'high', or 'medium', "
            f"got {config.float32_matmul_precision!r}"
        )
