"""Optimizer construction for finetuning.

Single entrypoint: ``build_optimizer(model, cfg)`` returns a fused AdamW
with weight decay split (no WD on biases or norm layers).
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from .config import TrainingConfig


def _param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Split params into decay / no-decay groups.

    No weight decay for:
    * 1-dimensional parameters (biases, norm scales, embedding tables'
      broadcast channels) — standard practice for LLM finetuning.
    * Params whose name contains ``"bias"`` or ``"norm"`` — belt and
      suspenders in case dimensionality heuristic misses anything.

    Only trainable params (``requires_grad=True``) are included, so LoRA
    finetuning naturally trains only the adapter.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.dim() <= 1
            or name.endswith(".bias")
            or "bias" in name.split(".")
            or "norm" in name.lower()
        )
        if is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Create a fused AdamW for the given model and training config.

    Under FSDP2 the parameters are DTensors; ``torch.optim.AdamW`` with
    ``fused=True`` handles DTensor parameters transparently for current
    torch versions.
    """
    groups = _param_groups(model, cfg.weight_decay)
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
