"""Auto-detect the largest device_batch_size that fits in GPU memory.

When ``device_batch_size`` is set to ``"auto"`` in the training config, this
module runs a full faux training step (forward + backward + optimizer) to
allocate all lazy state — cuBLAS handles, FSDP collective workspaces,
NCCL buffers, dynamo caches, and optimizer momentum/velocity buffers — then
probes forward+backward at batch sizes 1 and 2 to measure the per-sample
activation memory and extrapolates to the memory cap. The extrapolation is
only a rough guess, so the estimate is then validated with a backoff loop
(shrinking until it fits) followed by a grow loop (expanding to the largest
feasible value) so that the final result is the true maximum
device_batch_size subject to the global batch divisibility constraint.

Because the warm-up step allocates optimizer state in-place, subsequent
probes see the real memory baseline and no analytical correction is needed.
RNG states are saved and restored, and optimizer buffers are zeroed after
probing, so the faux step does not affect training reproducibility.
"""

from __future__ import annotations

import gc
import random
import warnings
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .config import Config
from .distributed import is_main_process, world_size
from .loop import _autocast_ctx


# ---------------------------------------------------------------------------
# RNG state save / restore
# ---------------------------------------------------------------------------


def _save_rng_states(device: torch.device) -> Dict[str, Any]:
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if device.type == "cuda":
        states["torch_cuda"] = torch.cuda.get_rng_state(device)
    return states


def _restore_rng_states(states: Dict[str, Any], device: torch.device) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in states:
        torch.cuda.set_rng_state(states["torch_cuda"], device)


# ---------------------------------------------------------------------------
# Dummy batch creation
# ---------------------------------------------------------------------------


def _create_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a worst-case dummy batch for memory probing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Memory probing
# ---------------------------------------------------------------------------


def _reset_cuda_memory(model: nn.Module, device: torch.device) -> None:
    """Release all cached GPU memory before a fresh probe.

    Zeros gradients, drops Python-level cycles via ``gc.collect`` (the
    allocator cannot reclaim blocks still pinned by a Python reference),
    then releases cached blocks back to the driver. Finally, synchronizes
    the device so prior kernels have finished touching the memory.
    """
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _probe_memory(
    model: nn.Module,
    fwd_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> float:
    """Run forward+backward and return peak memory in bytes.

    Uses *fwd_model* (which may be a ``torch.compile`` wrapper) for the
    forward pass, matching the real training code path. *model* is the
    raw (FSDP-wrapped) module used only for gradient cleanup.

    Returns ``float('inf')`` if OOM occurs. Regardless of outcome, the
    autograd graph, gradients, and cached allocator blocks are released
    before returning so that the next probe starts from a clean slate.
    """
    outputs = None
    loss = None
    _reset_cuda_memory(model, device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with _autocast_ctx(cfg):
            # ``use_cache=False`` matches the real training loop
            # (see ``loop._do_optimizer_step``) — HF defaults to ``True``,
            # which keeps ``past_key_values`` live on the output and
            # inflates the measured peak relative to actual training.
            outputs = fwd_model(**batch, use_cache=False)
            loss = outputs.loss
        loss.backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        return float(peak)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return float("inf")
        raise
    finally:
        # Drop references to tensors that keep the autograd graph and
        # activation memory alive, then run the cleanup sequence so the
        # blocks are actually reclaimed before we return. This matters for
        # both paths: on OOM the partial graph must go; on success the
        # activations and gradients must go before the next retry.
        del outputs, loss
        _reset_cuda_memory(model, device)


def _all_ranks_ok(device: torch.device) -> bool:
    """Check that no rank signalled OOM (used to avoid collective hangs)."""
    if not dist.is_initialized():
        return True
    ok = torch.ones(1, device=device, dtype=torch.long)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


# ---------------------------------------------------------------------------
# Divisibility constraint
# ---------------------------------------------------------------------------


def _find_largest_valid_bs(max_bs: int, global_batch_size: int, ws: int) -> int:
    """Find the largest ``device_batch_size <= max_bs`` satisfying the
    divisibility constraint ``global_batch_size % (bs * ws) == 0``.
    """
    if global_batch_size % ws != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by "
            f"world_size ({ws})"
        )
    per_gpu = global_batch_size // ws
    for bs in range(min(max(max_bs, 1), per_gpu), 0, -1):
        if per_gpu % bs == 0:
            return bs
    return 1


def _next_valid_bs_above(current: int, global_batch_size: int, ws: int) -> int | None:
    """Smallest ``device_batch_size > current`` satisfying divisibility, or
    ``None`` if ``current`` is already the per-GPU maximum.
    """
    if global_batch_size % ws != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by "
            f"world_size ({ws})"
        )
    per_gpu = global_batch_size // ws
    for bs in range(current + 1, per_gpu + 1):
        if per_gpu % bs == 0:
            return bs
    return None


# ---------------------------------------------------------------------------
# Safe probe with multi-GPU OOM coordination
# ---------------------------------------------------------------------------


def _warmup_full_step(
    model: nn.Module,
    fwd_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    vocab_size: int,
    cfg: Config,
    device: torch.device,
) -> None:
    """Run a full training step (fwd + bwd + optimizer) to allocate all lazy state.

    After this call the optimizer's momentum/velocity buffers, cuBLAS
    handles, FSDP collective workspaces, dynamo caches, and NCCL buffers
    are all resident in GPU memory. Subsequent memory probes therefore
    measure the true baseline and don't need analytical corrections.
    """
    _reset_cuda_memory(model, device)
    batch = _create_dummy_batch(1, seq_len, vocab_size, device)
    outputs = None
    loss = None
    try:
        with _autocast_ctx(cfg):
            outputs = fwd_model(**batch, use_cache=False)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                "[auto_bs] OOM during warm-up step. Enable "
                "gradient_checkpointing, use FSDP, reduce max_length, "
                "or use a smaller model."
            ) from e
        raise
    finally:
        del outputs, loss, batch
        _reset_cuda_memory(model, device)


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Zero optimizer state buffers so training starts fresh.

    Keeps the tensors allocated (the memory is already accounted for) but
    resets values so the warm-up step doesn't contaminate real training.
    """
    for param_states in optimizer.state.values():
        for k, v in param_states.items():
            if isinstance(v, torch.Tensor):
                v.zero_()


def _safe_probe(
    model: nn.Module,
    fwd_model: nn.Module,
    bs: int,
    seq_len: int,
    vocab_size: int,
    cfg: Config,
    device: torch.device,
) -> float:
    """Probe at a given batch size and coordinate OOM across ranks.

    If any rank OOMs, all ranks return ``float('inf')`` to avoid
    collective hangs from FSDP reduce-scatter.
    """
    ws = world_size()
    batch = _create_dummy_batch(bs, seq_len, vocab_size, device)
    try:
        mem = _probe_memory(model, fwd_model, batch, cfg, device)
    finally:
        # Drop the dummy batch tensors and reclaim their blocks before the
        # next probe (or before callers see control flow return).
        del batch
        _reset_cuda_memory(model, device)

    # Coordinate: if any rank OOM'd, all ranks must treat this as OOM.
    if ws > 1 and dist.is_initialized():
        ok_flag = torch.tensor(
            [0 if mem == float("inf") else 1], device=device, dtype=torch.long
        )
        dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)
        if ok_flag.item() == 0:
            return float("inf")

    return mem


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def auto_find_device_batch_size(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    tokenizer: Any,
    max_memory_fraction: float = 0.95,
    *,
    fwd_model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Probe GPU memory to find the largest feasible ``device_batch_size``.

    Must be called **after** the model is on-device and wrapped (FSDP/DDP),
    the optimizer is constructed, and ``torch.compile`` has been applied (if
    enabled).  By running a full faux training step (forward + backward +
    optimizer step) during warm-up, all lazy state — cuBLAS handles, FSDP
    collective workspaces, NCCL buffers, dynamo caches, and optimizer
    momentum/velocity buffers — is allocated before the calibration probes,
    so no analytical correction is needed.

    Args:
        model: The raw (FSDP-wrapped, uncompiled) model on *device*.
        cfg: Full training config.
        device: CUDA device to probe.
        tokenizer: HF tokenizer (used for vocab_size).
        max_memory_fraction: Fraction of total GPU memory to target.
        fwd_model: The forward model (may be a ``torch.compile`` wrapper).
            Falls back to *model* if not provided.
        optimizer: The optimizer. When provided, the warm-up phase runs a
            full training step so optimizer state memory is measured rather
            than estimated analytically.

    Returns:
        The resolved ``device_batch_size`` as a positive int.
    """
    if fwd_model is None:
        fwd_model = model

    if not torch.cuda.is_available():
        warnings.warn("[auto_bs] CUDA not available; falling back to device_batch_size=1")
        return 1

    ws = world_size()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    memory_cap = total_mem * max_memory_fraction

    # Use model's vocab_size (may differ from tokenizer after embedding resize).
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size or 32000
    seq_len = cfg.data.max_length

    rng_states = _save_rng_states(device)

    if is_main_process():
        print(
            f"[auto_bs] Probing GPU memory "
            f"(ws={ws}, "
            f"total={total_mem / 1e9:.1f} GB, "
            f"cap={memory_cap / 1e9:.1f} GB)",
            flush=True,
        )

    # --- Warm-up step ---
    #
    # Run a full faux training step (forward + backward + optimizer) to
    # allocate all lazy state: cuBLAS/cuDNN handles, FSDP collective
    # workspaces, NCCL buffers, the dynamo cache, and — critically —
    # optimizer momentum/velocity buffers.  Previous versions only ran
    # forward+backward and accounted for optimizer state analytically,
    # which under-estimated for compiled / FSDP2 models and produced
    # batch sizes that were too small.  After this step every persistent
    # allocation is resident, so the calibration probes see the true
    # memory baseline without needing any analytical corrections.
    if optimizer is not None:
        _warmup_full_step(
            model, fwd_model, optimizer, seq_len, vocab_size, cfg, device,
        )
    else:
        # Legacy fallback: no optimizer provided — plain fwd+bwd warm-up.
        _ = _safe_probe(model, fwd_model, 1, seq_len, vocab_size, cfg, device)

    # --- Probe at bs=1 ---
    mem_1 = _safe_probe(model, fwd_model, 1, seq_len, vocab_size, cfg, device)

    if mem_1 == float("inf"):
        _restore_rng_states(rng_states, device)
        raise RuntimeError(
            "[auto_bs] Model does not fit in GPU memory even at "
            "device_batch_size=1. Enable gradient_checkpointing, use FSDP, "
            "reduce max_length, or use a smaller model."
        )

    # --- Probe at bs=2 ---
    mem_2 = _safe_probe(model, fwd_model, 2, seq_len, vocab_size, cfg, device)

    if mem_2 == float("inf"):
        warnings.warn("[auto_bs] batch_size=2 OOMs; falling back to device_batch_size=1")
        _restore_rng_states(rng_states, device)
        return _find_largest_valid_bs(1, cfg.training.batch_size, ws)

    # --- Extrapolate ---
    mem_per_sample = mem_2 - mem_1
    if mem_per_sample <= 0:
        warnings.warn(
            f"[auto_bs] Non-positive per-sample memory delta "
            f"({mem_per_sample / 1e6:.1f} MB); falling back to device_batch_size=1"
        )
        _restore_rng_states(rng_states, device)
        return _find_largest_valid_bs(1, cfg.training.batch_size, ws)

    mem_fixed = mem_1 - mem_per_sample
    estimated_max = int((memory_cap - mem_fixed) / mem_per_sample)
    estimated_max = max(1, estimated_max)

    if is_main_process():
        print(
            f"[auto_bs] Probe results: "
            f"mem@1={mem_1 / 1e9:.2f} GB, "
            f"mem@2={mem_2 / 1e9:.2f} GB, "
            f"per_sample={mem_per_sample / 1e6:.0f} MB, "
            f"raw_estimate={estimated_max}",
            flush=True,
        )

    # --- Multi-GPU synchronization: take the minimum across ranks ---
    if ws > 1 and dist.is_initialized():
        bs_tensor = torch.tensor([estimated_max], device=device, dtype=torch.long)
        dist.all_reduce(bs_tensor, op=dist.ReduceOp.MIN)
        estimated_max = int(bs_tensor.item())

    # --- Apply divisibility constraint ---
    estimated_max = _find_largest_valid_bs(
        estimated_max, cfg.training.batch_size, ws
    )

    # --- Validation probe with backoff loop ---
    #
    # The analytic extrapolation is only a rough guess, so we first back off
    # until we find a batch size that fits, then try to grow past the initial
    # estimate to make sure we're actually using the maximum feasible value.
    def _fits(bs: int) -> tuple[bool, float]:
        mem = _safe_probe(model, fwd_model, bs, seq_len, vocab_size, cfg, device)
        ok = mem != float("inf") and mem <= memory_cap
        # Synchronize the fit decision across ranks: any rank that doesn't
        # fit means all ranks must treat this bs as infeasible.
        if ws > 1 and dist.is_initialized():
            ok_flag = torch.tensor(
                [1 if ok else 0], device=device, dtype=torch.long
            )
            dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)
            ok = bool(ok_flag.item())
        return ok, mem

    max_backoff_attempts = 6
    backed_off = False
    fit_bs = None
    for attempt in range(max_backoff_attempts):
        ok, mem_v = _fits(estimated_max)
        if ok:
            fit_bs = estimated_max
            break

        # Back off to the next smaller valid batch size.
        prev = estimated_max
        estimated_max = _find_largest_valid_bs(
            estimated_max - 1, cfg.training.batch_size, ws
        )
        if is_main_process():
            reason = "OOM" if mem_v == float("inf") else f"peak={mem_v / 1e9:.2f} GB > cap"
            print(
                f"[auto_bs] Validation failed ({reason}) at bs={prev}; "
                f"backing off to {estimated_max} (attempt {attempt + 1}/{max_backoff_attempts})",
                flush=True,
            )
        backed_off = True
        if estimated_max >= prev:
            # Can't go lower — stuck at the floor.
            fit_bs = estimated_max
            break

    if fit_bs is None:
        # Exhausted backoff attempts without finding a fit; use the floor.
        fit_bs = estimated_max

    # --- Grow phase ---
    #
    # If the initial estimate fit on the first try, we likely under-estimated.
    # Try successively larger valid batch sizes until one fails or we hit the
    # per-GPU ceiling. If we backed off, we already know the next larger size
    # didn't fit, so growing would just re-probe a known-bad value.
    if not backed_off:
        max_grow_attempts = 8
        for _ in range(max_grow_attempts):
            next_bs = _next_valid_bs_above(fit_bs, cfg.training.batch_size, ws)
            if next_bs is None:
                break  # Already at per-GPU maximum.
            ok, mem_v = _fits(next_bs)
            if ok:
                if is_main_process():
                    print(
                        f"[auto_bs] bs={next_bs} fits "
                        f"(peak={mem_v / 1e9:.2f} GB); growing",
                        flush=True,
                    )
                fit_bs = next_bs
            else:
                if is_main_process():
                    reason = (
                        "OOM" if mem_v == float("inf")
                        else f"peak={mem_v / 1e9:.2f} GB > cap"
                    )
                    print(
                        f"[auto_bs] bs={next_bs} doesn't fit ({reason}); "
                        f"stopping grow at bs={fit_bs}",
                        flush=True,
                    )
                break

    estimated_max = fit_bs

    # --- Cleanup ---
    # Reset optimizer state values so the warm-up step doesn't contaminate
    # real training. The buffers stay allocated (preserving memory layout).
    if optimizer is not None:
        _reset_optimizer_state(optimizer)
    _reset_cuda_memory(model, device)
    _restore_rng_states(rng_states, device)

    if is_main_process():
        print(f"[auto_bs] Final device_batch_size = {estimated_max}", flush=True)

    return max(1, estimated_max)
