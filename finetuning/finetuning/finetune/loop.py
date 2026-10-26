"""Core training loop.

Owns the forward/backward/optim-step cycle explicitly. Does not use
``transformers.Trainer``. Manages:

* gradient accumulation (configurable ``device_batch_size`` ↔ ``batch_size``),
* FSDP2 per-micro-batch grad-sync suppression,
* mixed-precision autocast for the non-FSDP path,
* scheduler step, EMA update, token accounting,
* callback dispatch at train/epoch/step boundaries.
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import torch
import torch.nn as nn

from .config import Config
from .distributed import is_main_process, world_size
from .state import TrainerState


# ---------------------------------------------------------------------------
# Callback dispatch
# ---------------------------------------------------------------------------


def _dispatch(callbacks: Sequence, hook: str, *args, **kwargs) -> None:
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is None:
            continue
        fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _autocast_ctx(cfg: Config):
    """Return an autocast context manager matching the config.

    Under FSDP2, ``MixedPrecisionPolicy`` already casts params to bf16/fp16
    at the parameter level, so this is effectively a no-op. For the
    non-FSDP path it's how we get mixed-precision forward.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    mp = cfg.training.mixed_precision
    if mp == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if mp == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _compute_loop_sizes(cfg: Config, dataloader_len: int) -> Dict[str, int]:
    """Derive grad_accum, steps_per_epoch, max_steps from config + data size."""
    tc = cfg.training
    world = world_size()
    micro = tc.device_batch_size * world
    if tc.batch_size % micro != 0:
        raise ValueError(
            f"batch_size ({tc.batch_size}) must be a multiple of "
            f"device_batch_size * world_size ({tc.device_batch_size} * {world} = {micro})"
        )
    grad_accum = tc.batch_size // micro
    steps_per_epoch = dataloader_len // grad_accum
    if steps_per_epoch == 0:
        raise ValueError(
            f"Dataloader has only {dataloader_len} batches but grad_accum is "
            f"{grad_accum}; at least {grad_accum} batches are needed per step."
        )
    if tc.max_steps is not None and tc.max_steps > 0:
        max_steps = tc.max_steps
    else:
        max_steps = tc.epochs * steps_per_epoch
    return {
        "grad_accum": grad_accum,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
    }


def _set_sampler_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


# ---------------------------------------------------------------------------
# One optimizer step
# ---------------------------------------------------------------------------


def _do_optimizer_step(
    state: TrainerState,
    fwd_model: nn.Module,
    data_iter: Iterator[Dict[str, torch.Tensor]],
    grad_accum: int,
    cfg: Config,
    autocast_ctx,
    device: torch.device,
    profiling: bool = False,
) -> Dict[str, Any]:
    """Consume ``grad_accum`` micro-batches and take one optimizer step.

    Returns a ``logs`` dict with training metrics for the step. Raises
    ``StopIteration`` if the data iterator is exhausted before the full
    accumulation is complete.
    """
    _prof = profiling and torch.cuda.is_available()

    if _prof:
        torch.cuda.synchronize(device)
        t_step_start = time.monotonic()
        t_data = 0.0
        t_forward = 0.0
        t_backward = 0.0

    state.optimizer.zero_grad(set_to_none=True)

    # Accumulate loss and token counts as GPU tensors to avoid per-micro-batch
    # .item() syncs that stall the pipeline between backward and the next
    # forward.  We extract scalars once after the optimizer step, when the
    # GPU work is already complete.
    _ws = max(1, world_size())
    total_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens_t = torch.zeros((), device=device, dtype=torch.long)
    max_seq_len = 0

    # ``state.model`` is the raw (FSDP-wrapped, uncompiled) module — used for
    # parameter iteration, grad-sync control, and EMA updates. ``fwd_model``
    # may be a ``torch.compile`` wrapper of ``state.model`` used only for
    # the forward pass.
    raw_model = state.model

    for micro_i in range(grad_accum):
        if _prof:
            torch.cuda.synchronize(device)
            t0 = time.monotonic()

        batch = next(data_iter)  # may raise StopIteration → caller handles
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        max_seq_len = max(max_seq_len, batch["input_ids"].shape[1])

        # FSDP2: avoid reducing grads on every micro-batch; sync only on the
        # last one. On non-FSDP models, this attribute is absent and we fall
        # through to ordinary gradient accumulation (which still syncs every
        # backward in DDP — acceptable for grad_accum=1 which is our case).
        is_last = (micro_i == grad_accum - 1)
        if hasattr(raw_model, "set_requires_gradient_sync"):
            raw_model.set_requires_gradient_sync(is_last)

        if _prof:
            torch.cuda.synchronize(device)
            t1 = time.monotonic()
            t_data += t1 - t0

        with autocast_ctx:
            outputs = fwd_model(**batch, use_cache=False)
            loss = outputs.loss / grad_accum

        if _prof:
            torch.cuda.synchronize(device)
            t2 = time.monotonic()
            t_forward += t2 - t1

        loss.backward()
        # loss was divided by grad_accum inside the micro-loop, so the
        # sum over micro-batches equals the mean over the full mini-batch.
        total_loss_t += loss.detach().float()
        if "attention_mask" in batch:
            # Report *global* (cross-rank) tokens for a meaningful
            # throughput number; assumes each rank sees an equally-shaped
            # batch, which holds for our DistributedSampler setup.
            total_tokens_t += batch["attention_mask"].sum() * _ws
        elif "input_ids" in batch:
            # Dense batch (no padding) — all tokens are real.
            total_tokens_t += batch["input_ids"].numel() * _ws

        if _prof:
            torch.cuda.synchronize(device)
            t3 = time.monotonic()
            t_backward += t3 - t2

    # Gradient clipping (on full, accumulated grad).
    if _prof:
        torch.cuda.synchronize(device)
        t_clip_start = time.monotonic()

    gn = None
    if cfg.training.max_grad_norm and cfg.training.max_grad_norm > 0:
        gn = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), cfg.training.max_grad_norm
        )

    if _prof:
        torch.cuda.synchronize(device)
        t_optim_start = time.monotonic()

    state.optimizer.step()
    state.scheduler.step()
    state.step += 1

    # Extract scalars after optimizer step — GPU work is complete so these
    # .item() calls won't stall the pipeline.
    total_loss = total_loss_t.item()
    total_tokens = int(total_tokens_t.item())
    grad_norm = gn.item() if gn is not None else 0.0
    state.tokens_seen += total_tokens

    if _prof:
        torch.cuda.synchronize(device)
        t_ema_start = time.monotonic()

    if state.ema is not None:
        state.ema.update(raw_model)

    logs: Dict[str, Any] = {
        "loss": total_loss,
        "lr": state.scheduler.get_last_lr()[0],
        "grad_norm": grad_norm,
        "tokens": total_tokens,
        "max_seq_len": max_seq_len,
    }

    if _prof:
        torch.cuda.synchronize(device)
        t_step_end = time.monotonic()
        t_total = max(t_step_end - t_step_start, 1e-9)
        logs["_profiling"] = {
            "data_load": t_data,
            "forward": t_forward,
            "backward": t_backward,
            "grad_clip": t_optim_start - t_clip_start,
            "optimizer_step": t_ema_start - t_optim_start,
            "ema_update": t_step_end - t_ema_start,
            "step_total": t_total,
        }

    return logs


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def train(
    state: TrainerState,
    dataloaders: Sequence[Iterable],
    callbacks: Sequence,
    cfg: Config,
    fwd_model: nn.Module | None = None,
) -> Dict[str, Any]:
    """Run the multi-stage training loop until ``total_max_steps`` is reached.

    ``dataloaders`` is a list with the same length as ``cfg.data.stages``,
    one DataLoader per stage. Stages run sequentially under a single
    optimizer + scheduler whose horizon was already sized to span all
    stages by ``run_training`` — i.e. warmup happens once at the very
    start and the cosine decay (if any) spans the full multi-stage
    duration.

    Mutates ``state`` in place. Returns a dict containing ``log_history``
    (a list of per-step log records) for the caller to propagate.

    If *fwd_model* is provided (already compiled by the caller) it is
    used directly; otherwise compilation is done here as a fallback.
    """
    stages = cfg.data.stages
    if len(dataloaders) != len(stages):
        raise ValueError(
            f"train() expected {len(stages)} dataloaders to match "
            f"cfg.data.stages, got {len(dataloaders)}"
        )

    ws = max(1, world_size())
    micro = cfg.training.device_batch_size * ws
    if cfg.training.batch_size % micro != 0:
        raise ValueError(
            f"batch_size ({cfg.training.batch_size}) must be a multiple of "
            f"device_batch_size * world_size ({cfg.training.device_batch_size} "
            f"* {ws} = {micro})"
        )
    grad_accum = cfg.training.batch_size // micro
    per_stage_steps_per_epoch = [
        max(1, len(dl) // grad_accum) for dl in dataloaders
    ]
    per_stage_max_steps = [
        s.epochs * spe for s, spe in zip(stages, per_stage_steps_per_epoch)
    ]
    if cfg.training.max_steps is not None and cfg.training.max_steps > 0:
        total_max_steps = cfg.training.max_steps
    else:
        total_max_steps = sum(per_stage_max_steps)

    state.max_steps = total_max_steps

    autocast_ctx = _autocast_ctx(cfg)
    device = next(state.model.parameters()).device

    # Use the caller-provided compiled model, or build one here as a
    # fallback for callers that don't pass fwd_model.
    if fwd_model is None:
        if cfg.model.compile:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            fwd_model = torch.compile(state.model)
        else:
            fwd_model = state.model

    profiling = cfg.logging.profiling

    _dispatch(callbacks, "on_train_begin", state)

    # Walk the stages from wherever ``state.step`` lands; this lets resume
    # skip past completed stages while still firing per-stage hooks for
    # the in-progress one.
    cumulative = 0
    stage_starts: list[int] = []
    for sm in per_stage_max_steps:
        stage_starts.append(cumulative)
        cumulative += sm

    epoch_idx = 0  # global epoch counter across all stages

    for stage_idx, (stage, dataloader) in enumerate(zip(stages, dataloaders)):
        stage_start = stage_starts[stage_idx]
        stage_end = stage_start + per_stage_max_steps[stage_idx]
        if state.step >= stage_end:
            # Resume past this stage — already done.
            continue

        steps_per_epoch = per_stage_steps_per_epoch[stage_idx]
        stage_target = min(stage_end, total_max_steps)

        _dispatch(callbacks, "on_stage_begin", state,
                  stage_idx=stage_idx, stage=stage)

        while state.step < stage_target:
            _set_sampler_epoch(dataloader, epoch_idx)
            _dispatch(callbacks, "on_epoch_begin", state)

            data_iter = iter(dataloader)
            budget = min(steps_per_epoch, stage_target - state.step)
            epoch_completed = True
            for _ in range(budget):
                try:
                    logs = _do_optimizer_step(
                        state, fwd_model, data_iter, grad_accum, cfg,
                        autocast_ctx, device, profiling=profiling,
                    )
                except StopIteration:
                    # Dataloader exhausted before we planned — drop the
                    # partial step. Happens when dataloader_len is not an
                    # exact multiple of grad_accum.
                    epoch_completed = False
                    break

                state.epoch = state.step / max(1, steps_per_epoch)
                logs["step"] = state.step
                logs["epoch"] = state.epoch
                logs["stage"] = stage_idx
                state.log_history.append(logs)
                _dispatch(callbacks, "on_step_end", state, logs=logs)

                if state.step >= total_max_steps:
                    break

            epoch_idx += 1
            # Fire on_epoch_end only when the current epoch has truly
            # completed: both the full per-epoch budget ran AND the
            # dataloader wasn't exhausted early.
            epoch_fully_done = (budget == steps_per_epoch) and epoch_completed
            if epoch_fully_done:
                _dispatch(callbacks, "on_epoch_end", state)

            if state.step >= total_max_steps:
                break

        _dispatch(callbacks, "on_stage_end", state,
                  stage_idx=stage_idx, stage=stage)

        if state.step >= total_max_steps:
            break

    _dispatch(callbacks, "on_train_end", state)

    return {"log_history": state.log_history}
