"""TrainerState and Reload dataclasses.

``TrainerState`` is the single mutable bag of training state shared
across the training loop, callbacks, and checkpoint I/O. ``Reload``
is an immutable selector describing which pieces of a saved checkpoint
should be restored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .config import Config
    from .ema import ModelEMA


@dataclass
class TrainerState:
    """Live training state. Mutated by the loop; read by callbacks."""

    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ema: Optional["ModelEMA"] = None

    step: int = 0              # number of completed optimizer steps
    epoch: float = 0.0         # fractional epoch count
    tokens_seen: int = 0       # non-padding tokens forwarded so far
    max_steps: int = 0         # planned total optimizer steps

    log_history: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional["Config"] = None


@dataclass(frozen=True)
class Reload:
    """Which pieces of a saved checkpoint to restore into a TrainerState."""

    model: bool = True
    optimizer: bool = False
    scheduler: bool = False
    ema: bool = False
    trainer_state: bool = False  # step + epoch + tokens_seen + RNG

    @classmethod
    def resume(cls) -> "Reload":
        """Mid-run resume: restore *everything*."""
        return cls(
            model=True,
            optimizer=True,
            scheduler=True,
            ema=True,
            trainer_state=True,
        )

    @classmethod
    def branch(cls) -> "Reload":
        """Trunk → branch handoff: weights + optimizer + EMA, fresh scheduler and step counter."""
        return cls(
            model=True,
            optimizer=True,
            scheduler=False,
            ema=True,
            trainer_state=False,
        )

    def without(self, **overrides: bool) -> "Reload":
        """Return a copy with specified fields overridden."""
        return replace(self, **overrides)
