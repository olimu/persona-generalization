"""Consolidated (non-sharded) checkpoint I/O for finetuning.

One on-disk format, one save entrypoint, one load entrypoint. All
checkpoints are *full* state dicts (rank 0 only writes), so they can be
reloaded onto a different world size and re-sharded by FSDP2.

Directory layout::

    ckpt/
      model.pt          — full model state_dict (rank 0)
      optimizer.pt      — full optimizer state_dict (rank 0)
      scheduler.pt      — scheduler.state_dict()   (rank 0)
      ema.pt            — full EMA state_dict      (rank 0, optional)
      trainer_state.pt  — step / epoch / tokens / rng (rank 0)

Saving and loading use the modern
``torch.distributed.checkpoint.state_dict`` APIs
(``get_model_state_dict`` / ``set_model_state_dict`` etc.) which handle
DTensor gather + scatter transparently and are world-size agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from .distributed import barrier, is_main_process
from .state import Reload, TrainerState


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sd_save_options():
    """StateDictOptions for gathering a full state dict at save time.

    ``cpu_offload=True`` keeps the gathered tensors on rank-0 CPU so they
    can be serialized without exhausting GPU memory.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def _sd_load_options():
    """StateDictOptions for scattering a full state dict at load time.

    Each rank passes its own (identical) loaded dict; DCP shards it
    across ranks. We deliberately do NOT use ``broadcast_from_rank0=True``
    — the cross-version behavior there is murky and "every rank loads
    from the shared filesystem" is simpler and explicit.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def _get_model_state_dict(model: nn.Module) -> Dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    return get_model_state_dict(model, options=_sd_save_options())


def _set_model_state_dict(model: nn.Module, sd: Dict[str, Any]) -> None:
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    set_model_state_dict(model, sd, options=_sd_load_options())


def _get_optim_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    return get_optimizer_state_dict(model, optimizer, options=_sd_save_options())


def _set_optim_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sd: Dict[str, Any],
) -> None:
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=sd,
        options=_sd_load_options(),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_checkpoint(
    state: TrainerState,
    path: PathLike,
    *,
    save_optimizer_and_trainer_state: bool = True,
) -> None:
    """Save a full-state-dict checkpoint at ``path``.

    All ranks participate in the gathers; only rank 0 writes to disk. A
    ``dist.barrier`` at the end ensures all ranks exit this function in
    sync before training resumes. When
    ``save_optimizer_and_trainer_state=False``, the checkpoint omits
    ``optimizer.pt`` and ``trainer_state.pt`` but still writes the model,
    scheduler, and optional EMA state.
    """
    path = Path(path)
    if is_main_process():
        path.mkdir(parents=True, exist_ok=True)

    # 0. Save HF config.json so the checkpoint dir is self-describing
    #    (needed by AutoModelForCausalLM.from_pretrained at eval time).
    if is_main_process():
        raw = getattr(state.model, "_orig_mod", state.model)
        if hasattr(raw, "module"):  # unwrap DDP
            raw = raw.module
        if hasattr(raw, "config"):
            raw.config.to_json_file(str(path / "config.json"))

    # 1. Model (raw weights).
    model_sd = _get_model_state_dict(state.model)
    if is_main_process():
        torch.save(model_sd, path / "model.pt")

    # 2. Optimizer.
    if save_optimizer_and_trainer_state and state.optimizer is not None:
        opt_sd = _get_optim_state_dict(state.model, state.optimizer)
        if is_main_process():
            torch.save(opt_sd, path / "optimizer.pt")

    # 3. Scheduler (tiny; save on rank 0 only).
    if is_main_process() and state.scheduler is not None:
        torch.save(state.scheduler.state_dict(), path / "scheduler.pt")

    # 4. EMA — swap shadow into model, capture full state dict, swap back.
    #    This leverages FSDP2's DTensor-aware state dict machinery instead
    #    of requiring a separate world-size-aware EMA gather.
    if state.ema is not None:
        state.ema.swap_in(state.model)
        try:
            ema_sd = _get_model_state_dict(state.model)
        finally:
            state.ema.swap_out(state.model)
        if is_main_process():
            torch.save(ema_sd, path / "ema.pt")

    # 5. Trainer state.
    if save_optimizer_and_trainer_state and is_main_process():
        trainer_state: Dict[str, Any] = {
            "step": state.step,
            "epoch": state.epoch,
            "tokens_seen": state.tokens_seen,
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            trainer_state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(trainer_state, path / "trainer_state.pt")

    barrier()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


@torch.no_grad()
def load_checkpoint(
    path: PathLike,
    state: TrainerState,
    reload: Reload,
) -> None:
    """Load selected pieces from a checkpoint directory into ``state``.

    Must be called *after* model/optimizer/ema/scheduler are all
    constructed on each rank. Cross-world-size loading works because every
    on-disk artifact is a full (non-sharded) state dict.

    Load order is important:

      1. EMA (if requested) is scattered first, captured into the shadow,
         and then the raw model weights are scattered on top — leaving
         the model in its raw-weight state.
      2. Optimizer state is scattered after the model state.
      3. Scheduler / trainer_state are restored last.

    Requires ``reload.model=True`` whenever ``reload.ema=True``, since the
    EMA-load flow relies on a subsequent model scatter to restore the
    raw weights after the temporary swap.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    if reload.ema and not reload.model:
        raise ValueError(
            "reload.ema=True requires reload.model=True so the raw model "
            "weights can be restored after the EMA capture."
        )

    # Every rank loads each file from disk (it's a shared filesystem and
    # the files are a few GB at most for 1.7B models). This avoids any
    # ambiguity around ``broadcast_from_rank0`` semantics across torch
    # versions: DCP's ``set_*_state_dict`` with ``full_state_dict=True``
    # then has an identical dict on every rank and shards it cleanly.

    # 1. EMA: scatter ema.pt into the model, capture to shadow.
    ema_path = path / "ema.pt"
    if reload.ema and state.ema is not None and ema_path.exists():
        ema_sd = torch.load(ema_path, map_location="cpu", weights_only=True)
        _set_model_state_dict(state.model, ema_sd)
        state.ema.capture_from_model(state.model)
        # EMA shadow is now the loaded EMA. Model still holds EMA weights;
        # step 2 (reload.model) will overwrite them with the raw weights.
        del ema_sd

    # 2. Model (raw weights).
    if reload.model:
        model_sd = torch.load(
            path / "model.pt", map_location="cpu", weights_only=True
        )
        _set_model_state_dict(state.model, model_sd)
        del model_sd

        # If EMA exists but wasn't reloaded, resync the shadow with the
        # freshly loaded raw weights so it doesn't lag the model.
        if not reload.ema and state.ema is not None:
            state.ema.capture_from_model(state.model)

    # 3. Optimizer.
    if reload.optimizer and state.optimizer is not None:
        opt_path = path / "optimizer.pt"
        if not opt_path.exists():
            raise FileNotFoundError(
                f"Missing optimizer checkpoint: {opt_path}. "
                "Was checkpoint.save_optimizer_and_trainer_state disabled?"
            )
        opt_sd = torch.load(opt_path, map_location="cpu", weights_only=False)
        _set_optim_state_dict(state.model, state.optimizer, opt_sd)
        del opt_sd

    # 4. Scheduler (tiny; all ranks load independently).
    if reload.scheduler and state.scheduler is not None:
        sched_path = path / "scheduler.pt"
        if sched_path.exists():
            # weights_only=False because LambdaLR state_dict includes a list
            # of ``_lr_lambdas`` placeholders (``None`` entries for function
            # lambdas) that the weights_only allowlist rejects.
            sched_sd = torch.load(sched_path, map_location="cpu", weights_only=False)
            state.scheduler.load_state_dict(sched_sd)

    # 5. Trainer state.
    if reload.trainer_state:
        ts_path = path / "trainer_state.pt"
        if not ts_path.exists():
            raise FileNotFoundError(
                f"Missing trainer-state checkpoint: {ts_path}. "
                "Was checkpoint.save_optimizer_and_trainer_state disabled?"
            )
        ts = torch.load(ts_path, map_location="cpu", weights_only=False)
        state.step = int(ts.get("step", state.step))
        state.epoch = float(ts.get("epoch", state.epoch))
        state.tokens_seen = int(ts.get("tokens_seen", state.tokens_seen))
        if "torch_rng" in ts:
            torch.set_rng_state(ts["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in ts:
            torch.cuda.set_rng_state_all(ts["cuda_rng"])

    barrier()


# ---------------------------------------------------------------------------
# HF-format final save (for downstream CodeEvaluation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_hf_model(
    model: nn.Module,
    tokenizer,
    output_dir: PathLike,
) -> None:
    """Save a standard HuggingFace model directory (``config.json`` +
    safetensors + tokenizer files) on rank 0 for downstream eval.

    Uses the DCP state dict API to gather the full model weights to rank 0,
    then calls the unwrapped module's ``save_pretrained``.
    """
    output_dir = Path(output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    sd = _get_model_state_dict(model)

    if is_main_process():
        # Unwrap ``torch.compile``'s ``OptimizedModule`` if present. FSDP2's
        # ``fully_shard`` is applied in-place and does *not* nest a module;
        # the wrapped object still ``isinstance(PreTrainedModel)`` and its
        # ``save_pretrained`` will accept our override ``state_dict``.
        raw = getattr(model, "_orig_mod", model)
        # Unwrap DistributedDataParallel too, for the ``parallelism=ddp`` path.
        ddp_cls = torch.nn.parallel.DistributedDataParallel
        if isinstance(raw, ddp_cls):
            raw = raw.module
        raw.save_pretrained(output_dir, state_dict=sd, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

    barrier()


# ---------------------------------------------------------------------------
# Partial-finetune save (subset of parameters)
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_partial_model(
    model: nn.Module,
    tokenizer,
    output_dir: PathLike,
    *,
    base_model_name_or_path: str,
    trainable_mode: str,
    trainable_modules: List[str],
) -> None:
    """Save only the parameters whose ``requires_grad=True``.

    Modeled on PEFT's adapter directory layout but with distinct file
    names so the format is unambiguous:

        partial_config.json       — probe + base-model pointer + metadata
        partial_model.safetensors — filtered state_dict (only trainable params)
        tokenizer.json / special_tokens_map.json / tokenizer_config.json

    The on-disk weight set is determined by ``requires_grad`` on the
    LIVE model (not the configured ``trainable_modules`` list) — that
    way the format stays correct even if a future change to
    ``apply_trainable_config`` interprets module names differently.
    ``trainable_modules`` is still written into ``partial_config.json``
    as informational metadata for downstream readers.
    """
    import safetensors.torch

    output_dir = Path(output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # All ranks participate in the DCP gather — same path as save_hf_model.
    sd = _get_model_state_dict(model)

    if is_main_process():
        raw = getattr(model, "_orig_mod", model)
        ddp_cls = torch.nn.parallel.DistributedDataParallel
        if isinstance(raw, ddp_cls):
            raw = raw.module

        trainable_param_names = {
            n for n, p in raw.named_parameters() if p.requires_grad
        }
        partial_sd = {k: v for k, v in sd.items() if k in trainable_param_names}

        saved_param_count = sum(int(t.numel()) for t in partial_sd.values())
        total_param_count = sum(int(p.numel()) for p in raw.parameters())

        partial_config = {
            "format_version": 1,
            "base_model_name_or_path": base_model_name_or_path,
            "trainable_mode": trainable_mode,
            "trainable_modules": list(trainable_modules),
            "saved_param_names": sorted(partial_sd.keys()),
            "saved_param_count": saved_param_count,
            "total_param_count": total_param_count,
            "model_class": type(raw).__name__,
            "dtype": str(next(iter(partial_sd.values())).dtype) if partial_sd else None,
        }
        with open(output_dir / "partial_config.json", "w") as f:
            json.dump(partial_config, f, indent=2)

        safetensors.torch.save_file(
            partial_sd, str(output_dir / "partial_model.safetensors")
        )
        tokenizer.save_pretrained(output_dir)

    barrier()
