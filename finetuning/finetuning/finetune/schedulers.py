"""Unified learning-rate scheduler factory.

Replaces HF's ``get_scheduler`` string dispatch. Produces a ``LambdaLR``
whose lambda covers constant / cosine / warmup-stable-decay cases via a
single arithmetic expression.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


# Sentinel used for "constant after warmup" — we treat stable_steps as
# effectively infinite so the step counter never crosses into decay.
_STABLE_INFINITY = 10 ** 18


def _lr_factor(
    step: int,
    *,
    warmup: int,
    stable: int,
    decay: int,
    min_frac: float,
) -> float:
    """Return the LR multiplier for a given step.

    Layout:
      [0, warmup)                    linear ramp 0 → 1
      [warmup, warmup+stable)        flat at 1
      [warmup+stable, w+s+decay)     cosine 1 → min_frac
      [w+s+decay, ∞)                 flat at min_frac

    Edge cases:
    * warmup == 0 → no ramp; LR starts at 1.
    * decay == 0  → no decay; LR stays at 1 forever after the stable phase
                    (no division by zero).
    """
    if step < warmup:
        # Use ``(step + 1) / warmup`` so the very first training step
        # receives a non-zero LR (1/warmup) instead of exactly zero.
        # At ``step == warmup - 1`` this yields 1.0, flush against the
        # stable phase.
        if warmup <= 0:
            return 1.0
        return float(step + 1) / float(warmup)
    if step < warmup + stable:
        return 1.0
    if decay <= 0:
        return 1.0
    decay_step = step - warmup - stable
    if decay_step >= decay:
        return min_frac
    progress = float(decay_step) / float(decay)  # in [0, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
    return min_frac + (1.0 - min_frac) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: Literal["constant", "cosine", "wsd"],
    *,
    warmup_steps: int,
    stable_steps: int = 0,
    decay_steps: int = 0,
    min_lr_fraction: float = 0.1,
) -> LambdaLR:
    """Build a LambdaLR for the given schedule kind.

    ``constant``:  linear warmup → flat at the configured LR forever.
    ``cosine``:    linear warmup → cosine decay to ``min_lr_fraction``.
                   ``decay_steps`` specifies the length of the decay phase.
                   ``stable_steps`` is ignored.
    ``wsd``:       warmup → stable plateau (``stable_steps``) → cosine decay
                   (``decay_steps``) to ``min_lr_fraction``.

    The returned scheduler's state_dict is pickle-safe (LambdaLR omits the
    actual lambda). To restore state, re-call ``build_scheduler`` then call
    ``scheduler.load_state_dict(saved_state)``.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if stable_steps < 0:
        raise ValueError(f"stable_steps must be >= 0, got {stable_steps}")
    if decay_steps < 0:
        raise ValueError(f"decay_steps must be >= 0, got {decay_steps}")
    if not 0.0 <= min_lr_fraction <= 1.0:
        raise ValueError(f"min_lr_fraction must be in [0, 1], got {min_lr_fraction}")

    if kind == "constant":
        stable = _STABLE_INFINITY
        decay = 0
    elif kind == "cosine":
        stable = 0
        decay = decay_steps
    elif kind == "wsd":
        stable = stable_steps
        decay = decay_steps
    else:
        raise ValueError(f"Unknown scheduler kind: {kind!r}")

    # Closure captures integers and a float — trivially copyable.
    def lr_lambda(step: int) -> float:
        return _lr_factor(
            step,
            warmup=warmup_steps,
            stable=stable,
            decay=decay,
            min_frac=min_lr_fraction,
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
