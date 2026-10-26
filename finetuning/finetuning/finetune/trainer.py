"""Training orchestrator: glues config → model → loop together.

``run_training(config)`` is the single public entrypoint. It owns the
sequence:

  1. initialize distributed
  2. load raw HF model + tokenizer
  3. move to device, wrap with FSDP2 (if requested)
  4. build optimizer, EMA, scheduler
  5. load cross-phase checkpoint (if ``base_model_checkpoint`` is set)
  6. build dataset / dataloader / callbacks
  7. run the training loop
  8. save a final HF model directory (if ``save_final=True``)
  9. cleanup distributed

Does not use ``transformers.Trainer``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch._dynamo  # noqa: F401 — used conditionally for compile config
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .callbacks import (
    Callback,
    ConsoleLoggerCallback,
    ProfilingCallback,
    SaveByEpochCallback,
    SaveByStepCallback,
    SystemMetricsCallback,
    ThroughputCallback,
    WandbLoggerCallback,
)
from .checkpoint import load_checkpoint, save_hf_model, save_partial_model
from .collator import LMCollator
from .config import Config
from .data import create_dataset
from .distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    rank,
    world_size,
)
from .ema import ModelEMA
from .loop import train as train_loop
from .model import load_model
from .optim import build_optimizer
from .schedulers import build_scheduler
from .state import Reload, TrainerState


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class _TrainingDataset(Dataset):
    """``torch.utils.data.Dataset`` adapter around a tokenized data dict.

    Accepts the dict returned by ``BaseDataset.tokenize()`` directly so
    callers can decide whether to tokenize once-per-rank, once-globally
    (rank 0 + broadcast), or load from a shared cache. The values may
    be either Python lists (JSONL datasets) or numpy arrays
    (pretokenized binary memmaps); both shapes are handed to the
    collator as plain Python lists.
    """

    def __init__(self, data):
        self._ids = data["input_ids"]
        self._mask = data["attention_mask"]
        self._labels = data["labels"]
        self._n = len(self._ids)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ii = self._ids[idx]
        am = self._mask[idx]
        lb = self._labels[idx]
        if hasattr(ii, "tolist"):
            ii = ii.tolist()
        if hasattr(am, "tolist"):
            am = am.tolist()
        if hasattr(lb, "tolist"):
            lb = lb.tolist()
        return {"input_ids": ii, "attention_mask": am, "labels": lb}


def _tokenize_distributed(
    cfg: Config,
    tokenizer,
    *,
    train_path: str,
    train_subset: Optional[int],
    train_subset_seed: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    """Split tokenization across ranks; each rank takes a contiguous slice.

    Default per-rank behavior would run ``BaseDataset.tokenize()`` over
    the whole dataset on every rank — N× wasted work where N is
    ``world_size``. Instead, each rank loads the same raw JSONL (cheap
    parse), keeps only its row slice (rank ``r`` gets rows
    ``[r * N // ws, (r + 1) * N // ws)``), runs ``tokenize()`` on that
    slice, and ``all_gather_object``s the per-rank tokenized dicts.
    Every rank ends up with the full tokenized dataset in the original
    row order, so the downstream ``DistributedSampler`` partitioning
    semantics are unchanged.

    For ``world_size == 1`` the function degenerates to a plain
    ``source.tokenize()``.

    The all_gather payload is bounded by the per-rank slice's token
    count (under our current configs ≈ 15-30 MB per rank for an 8K-row
    chat-template SFT dataset, gathered into N copies on every rank)
    — well within NCCL's pickle-over-collective limits.
    """
    ws = world_size()
    r = rank()

    source = create_dataset(
        train_path,
        cfg.data.train_type,
        tokenizer,
        max_length=cfg.data.max_length,
        shuffle=False,  # DistributedSampler owns shuffling
        seed=seed,
        apply_chat_template=cfg.data.apply_chat_template,
        subset=train_subset,
        subset_seed=train_subset_seed,
    )

    if ws == 1:
        return source.tokenize()

    # PreTokenizedDataset stores a fixed-shape memmap (no Python
    # tokenize loop) — splitting it across ranks gains nothing and
    # the ``raw_data`` attr doesn't even exist. Fall back to "every
    # rank tokenizes its own memmap view" for that case.
    if not hasattr(source, "raw_data"):
        return source.tokenize()

    n = len(source.raw_data)
    start = r * n // ws
    end = (r + 1) * n // ws

    # Tokenize only this rank's slice. We mutate ``raw_data`` in place
    # to keep the tokenize() implementation slice-agnostic; the
    # ``finally`` restores it so anything downstream that touches the
    # source still sees the full data (none of our paths do, but it's
    # cheap insurance).
    full_raw = source.raw_data
    source.raw_data = full_raw[start:end]
    try:
        my_tokenized: Dict[str, Any] = source.tokenize()
    finally:
        source.raw_data = full_raw

    # Gather every rank's tokenized slice, in rank order.
    gathered: List[Optional[Dict[str, Any]]] = [None] * ws

    import pickle

    payload = pickle.dumps(my_tokenized)
    print(
        f"[rank {r}] rows={end-start}, "
        f"keys={list(my_tokenized.keys())}, "
        f"pickle_bytes={len(payload)}",
        flush=True,
    )
    
    torch.distributed.all_gather_object(gathered, my_tokenized)

    keys = list(my_tokenized.keys())
    merged: Dict[str, List[Any]] = {k: [] for k in keys}
    for chunk in gathered:
        assert chunk is not None
        for k in keys:
            merged[k].extend(chunk[k])
    return merged


def _build_dataloader(
    cfg: Config,
    tokenizer,
    *,
    train_path: str,
    train_subset: Optional[int],
    train_subset_seed: Optional[int],
    seed: int,
) -> DataLoader:
    tokenized = _tokenize_distributed(
        cfg, tokenizer,
        train_path=train_path,
        train_subset=train_subset,
        train_subset_seed=train_subset_seed,
        seed=seed,
    )
    dataset = _TrainingDataset(tokenized)

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collator = LMCollator(pad_id, cfg.data.max_length, padding_side="left")

    ws = world_size()
    sampler: Optional[DistributedSampler] = None
    if ws > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ws,
            rank=rank(),
            shuffle=True,
            seed=seed,
            drop_last=cfg.training.dataloader_drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.training.device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=cfg.training.dataloader_drop_last,
        persistent_workers=True,
    )


# ---------------------------------------------------------------------------
# Callback assembly
# ---------------------------------------------------------------------------


def _build_callbacks(cfg: Config, num_params: int, tokenizer=None) -> List[Callback]:
    callbacks: List[Callback] = []

    if cfg.checkpoint.save_at_epochs:
        callbacks.append(
            SaveByEpochCallback(
                output_dir=Path(cfg.checkpoint.output_dir),
                save_at_epochs=cfg.checkpoint.save_at_epochs,
                save_optimizer_and_trainer_state=(
                    cfg.checkpoint.save_optimizer_and_trainer_state
                ),
                save_format=cfg.checkpoint.save_at_epochs_format,
                tokenizer=tokenizer,
            )
        )
    if cfg.checkpoint.save_at_steps:
        callbacks.append(
            SaveByStepCallback(
                output_dir=Path(cfg.checkpoint.output_dir),
                save_at_steps=cfg.checkpoint.save_at_steps,
                save_optimizer_and_trainer_state=(
                    cfg.checkpoint.save_optimizer_and_trainer_state
                ),
            )
        )

    callbacks.append(ConsoleLoggerCallback(log_every_n_steps=cfg.logging.log_every_n_steps))

    if cfg.logging.wandb:
        callbacks.append(WandbLoggerCallback(log_every_n_steps=cfg.logging.log_every_n_steps))

    if cfg.logging.log_throughput:
        callbacks.append(
            ThroughputCallback(
                num_params=num_params,
                log_every_n_steps=cfg.logging.log_every_n_steps,
                num_gpus=max(1, world_size()),
            )
        )

    if cfg.logging.log_system:
        callbacks.append(
            SystemMetricsCallback(log_every_n_steps=cfg.logging.log_every_n_steps)
        )

    if cfg.logging.profiling and cfg.logging.wandb:
        callbacks.append(
            ProfilingCallback(log_every_n_steps=cfg.logging.log_every_n_steps)
        )

    return callbacks


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------


def _wandb_init(cfg: Config) -> None:
    if not cfg.logging.wandb or not is_main_process():
        return
    try:
        import wandb
    except ImportError:
        return
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_name,
        tags=cfg.logging.wandb_tags,
        config={
            "model": vars(cfg.model),
            "data": vars(cfg.data),
            "training": vars(cfg.training),
            "checkpoint": vars(cfg.checkpoint),
            "ema": vars(cfg.ema),
            "lora": vars(cfg.lora),
        },
    )


def _wandb_finish() -> None:
    if not is_main_process():
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------


def run_training(config: Config) -> Dict[str, Any]:
    """End-to-end training orchestrator."""
    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    r, ws, local_rank = init_distributed()
    device = (
        torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    )

    if config.training.parallelism == "fsdp" and ws == 1:
        warnings.warn(
            "FSDP is selected but only 1 GPU is available. "
            "Switching to DDP automatically.",
            stacklevel=2,
        )
        config.training.parallelism = "ddp"

    if config.training.seed is not None:
        # Same seed on every rank — FSDP broadcasts rely on identical RNG
        # state for collective init, and the ``DistributedSampler`` uses
        # its own (rank-aware) seed for data shuffling.
        _set_seed(config.training.seed)

    _wandb_init(config)

    # 1. Load raw HF model + tokenizer.
    #    Weights stream straight onto ``device`` (no CPU staging) via
    #    ``device_map`` inside ``load_model``. For multi-GPU runs each
    #    rank sees its own ``cuda:{local_rank}``, so FSDP/DDP wrapping
    #    below operates on a model already resident on the right GPU.
    model, tokenizer = load_model(config.model, config.lora, device=device)
    tokenizer.padding_side = "left"
    if config.data.chat_template is not None:
        tokenizer.chat_template = config.data.chat_template

    # 1b. Apply selective trainability config (if any). Sets
    #     ``requires_grad`` on a per-parameter basis before FSDP wrap and
    #     optimizer build. The existing ``_param_groups`` in ``optim.py``
    #     filters by ``requires_grad``, so the optimizer naturally trains
    #     only the selected subset.
    from .trainable import apply_trainable_config
    apply_trainable_config(model, config.trainable)

    # 2. Count params on the raw model for MFU accounting. Counting here
    #    (pre-wrap) keeps the code simple: on a plain ``nn.Module`` every
    #    parameter is a regular tensor and ``p.numel()`` gives the real
    #    element count. Post-FSDP2 the params are ``DTensor``s whose
    #    ``.numel()`` still returns the *global* logical size (not the
    #    per-rank shard), so counting post-wrap would also work — but
    #    pre-wrap is unambiguous and avoids depending on DTensor semantics.
    num_params = sum(p.numel() for p in model.parameters())

    # 3. Wrap. ``load_model`` already placed the model on ``device``;
    #    ``.to(device)`` is left in as a cheap safety net for any
    #    parameters/buffers that slipped past ``device_map`` (e.g. extra
    #    tensors registered post-init by custom model code).
    model.to(device)
    if config.training.parallelism == "fsdp" and ws > 1:
        from .distributed import wrap_fsdp

        model = wrap_fsdp(model, config.training)
    elif config.training.parallelism == "ddp" and ws > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # single-process: no wrapping needed

    # 3. Optimizer + EMA.
    optimizer = build_optimizer(model, config.training)
    ema: Optional[ModelEMA] = None
    if config.ema.enabled:
        ema = ModelEMA(
            model,
            decay=config.ema.decay,
            update_every=config.ema.update_every,
            device="cpu" if config.ema.device == "cpu" else None,
        )

    # 3b. Auto-detect device_batch_size if requested. Done *before* compile
    #     so the probe varies the batch dimension on an eager model — if we
    #     compiled first, Dynamo would see multiple batch sizes across the
    #     bs=1/bs=2/backoff/grow probes and mark the batch dim dynamic, which
    #     is incompatible with FSDP2's root-level DTensor params (symbolic
    #     shapes on a Shard() DTensor hit a numel() assertion inside the
    #     DTensor mul dispatch). Compiling after pinning keeps the batch dim
    #     static in the traced graph.
    if config.training.device_batch_size == "auto":
        from .auto_batch_size import auto_find_device_batch_size

        resolved_bs = auto_find_device_batch_size(
            model=model,
            cfg=config,
            device=device,
            tokenizer=tokenizer,
            max_memory_fraction=config.training.max_memory_fraction,
            fwd_model=model,
            optimizer=optimizer,
        )
        config.training.device_batch_size = resolved_bs
        # Log the resolved value to wandb so the config panel shows the
        # actual int rather than "auto".
        if is_main_process():
            try:
                import wandb
                if wandb.run is not None:
                    wandb.config.update({"training/device_batch_size": resolved_bs})
            except ImportError:
                pass

    assert isinstance(config.training.device_batch_size, int), (
        f"device_batch_size must be resolved to int before building "
        f"dataloader, got {config.training.device_batch_size!r}"
    )

    # 3c. Compile model if requested. Deferred until after auto-bs so the
    #     first traced forward sees a single, fixed batch size and Dynamo
    #     compiles a static-shape graph.
    if config.model.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        fwd_model = torch.compile(model)
    else:
        fwd_model = model

    # 4. Dataloaders — one per stage. Built here so we can sum per-stage
    #    step budgets into the total horizon the scheduler needs upfront.
    seed = config.training.seed or 42
    stage_dataloaders = [
        _build_dataloader(
            config, tokenizer,
            train_path=stage.train_path,
            train_subset=stage.train_subset,
            train_subset_seed=stage.train_subset_seed,
            seed=seed,
        )
        for stage in config.data.stages
    ]

    # 5. Scheduler. The schedule is *continuous* across all stages — one
    #    warmup at step 0, one cosine decay to ``total_max_steps``. For
    #    "wsd" the caller is expected to set the phase lengths
    #    explicitly. ``warmup_ratio`` is the legacy fallback.
    _micro = config.training.device_batch_size * max(1, ws)
    _grad_accum = max(1, config.training.batch_size // _micro)
    stage_max_steps = [
        stage.epochs * max(1, len(dl) // _grad_accum)
        for stage, dl in zip(config.data.stages, stage_dataloaders)
    ]
    effective_max_steps = config.training.max_steps or sum(stage_max_steps)
    warmup_steps = config.training.warmup_steps
    if warmup_steps == 0 and config.training.warmup_ratio > 0:
        warmup_steps = int(round(config.training.warmup_ratio * effective_max_steps))
    decay_steps = config.training.wsd_decay_steps or 0
    if config.training.lr_scheduler == "cosine" and decay_steps == 0:
        decay_steps = max(0, effective_max_steps - warmup_steps)

    scheduler = build_scheduler(
        optimizer,
        kind=config.training.lr_scheduler,
        warmup_steps=warmup_steps,
        stable_steps=config.training.wsd_stable_steps or 0,
        decay_steps=decay_steps,
        min_lr_fraction=config.training.wsd_min_lr_fraction,
    )

    state = TrainerState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        config=config,
    )

    # 6. Cross-phase checkpoint load (trunk → branch).
    if config.checkpoint.base_model_checkpoint is not None:
        reload_selector = Reload(
            **{k: bool(v) for k, v in config.checkpoint.reload.items()}
        )
        if is_main_process():
            print(
                f"[ckpt] Loading {config.checkpoint.base_model_checkpoint} "
                f"with {reload_selector}",
                flush=True,
            )
        load_checkpoint(
            config.checkpoint.base_model_checkpoint, state, reload_selector
        )

    # 7. Callbacks.
    callbacks = _build_callbacks(config, num_params, tokenizer=tokenizer)

    # 8. Train.
    train_loop(state, stage_dataloaders, callbacks, config, fwd_model=fwd_model)

    # 9. Final save (branches only — trunks set save_final=False).
    #    Dispatch on ``config.trainable.enabled``: partial-finetune runs
    #    write a peft-style sparse directory (``partial_config.json`` +
    #    ``partial_model.safetensors``) carrying ONLY the parameters with
    #    ``requires_grad=True``; full / LoRA runs go through the standard
    #    ``save_hf_model`` path.
    if config.checkpoint.save_final:
        output_dir = Path(config.checkpoint.output_dir)
        if state.ema is not None:
            state.ema.swap_in(state.model)
        try:
            if config.trainable.enabled:
                save_partial_model(
                    state.model,
                    tokenizer,
                    output_dir,
                    base_model_name_or_path=config.model.path,
                    trainable_mode=config.trainable.mode,
                    trainable_modules=list(config.trainable.modules),
                )
            else:
                save_hf_model(state.model, tokenizer, output_dir)
        finally:
            if state.ema is not None:
                state.ema.swap_out(state.model)
        if is_main_process():
            log_path = output_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(state.log_history, f, indent=2)

    _wandb_finish()
    cleanup_distributed()

    return {
        "log_history": state.log_history,
        "num_params": num_params,
    }


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
