"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of trainable parameters updated with the EMA formula:

    shadow = decay * shadow + (1 - decay) * param

Shadow parameters are stored in fp32 on a configurable device (CPU by default).
Under FSDP2 each rank operates on its *local shard*: since EMA is a linear
operation, EMA-of-shards equals shard-of-EMA, so per-rank shadows are a
valid per-rank view of the full EMA. Cross-world-size checkpoint reload
uses the model state dict (via ``capture_from_model``) rather than a raw
per-rank shadow dump.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return the rank-local shard of ``t`` if it's a DTensor, else ``t`` itself.

    FSDP2 stores parameters as DTensors; ``.to_local()`` yields a reference
    to the underlying local tensor. In-place writes to the returned tensor
    propagate back through the DTensor wrapper.
    """
    to_local = getattr(t, "to_local", None)
    if to_local is None:
        return t
    return to_local()


class ModelEMA:
    """Exponential moving average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor in [0, 1). Higher = smoother.
        update_every: Apply EMA update every N calls to update().
        device: Device for shadow parameters ("cpu" or None to keep on model device).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_every: int = 1,
        device: Optional[str] = None,
    ):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")

        self.decay = decay
        self.update_every = update_every
        self.num_updates = 0
        self.device = device

        # Initialize shadow params from current model state (fp32).
        # Under FSDP2 we extract the rank-local shard so the shadow is a
        # plain tensor (not a DTensor), which simplifies CPU offload and
        # in-place EMA updates.
        target_device = torch.device(device) if device else None
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                t = _local(param.data).detach().clone().float()
                if target_device is not None:
                    t = t.to(target_device)
                self.shadow[name] = t

        self._saved: Dict[str, torch.Tensor] = {}
        self._verified = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA of current model parameters.

        Should be called once per optimizer step (not per micro-batch).
        Respects update_every — only applies the EMA formula every N calls.
        """
        self.num_updates += 1
        if self.num_updates % self.update_every != 0:
            return

        matched = 0
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            matched += 1
            dst = self.shadow[name]
            src = _local(param.data).detach().float()
            if dst.device != src.device:
                src = src.to(dst.device, non_blocking=True)
            dst.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

        if not self._verified:
            self._verified = True
            if matched == 0 and len(self.shadow) > 0:
                shadow_key = next(iter(self.shadow))
                model_key = next((n for n, _ in model.named_parameters()), None)
                raise RuntimeError(
                    f"EMA update matched 0 of {len(self.shadow)} shadow params. "
                    f"Parameter names changed between init and update. "
                    f"First shadow key: {shadow_key!r}, "
                    f"first model key: {model_key!r}"
                )

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> None:
        """Replace model parameters with EMA values, saving originals.

        Call ``swap_out()`` to restore the original parameters. Under FSDP2
        this operates on each rank's local shard; the swap is per-rank and
        does not need a collective.
        """
        self._saved = {}
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            local = _local(param.data)
            self._saved[name] = local.detach().clone()
            ema_val = self.shadow[name].to(dtype=local.dtype, device=local.device)
            local.copy_(ema_val)

    @torch.no_grad()
    def swap_out(self, model: nn.Module) -> None:
        """Restore original parameters after a ``swap_in()``."""
        for name, param in model.named_parameters():
            if name not in self._saved:
                continue
            _local(param.data).copy_(self._saved[name])
        self._saved = {}

    @torch.no_grad()
    def capture_from_model(self, model: nn.Module) -> None:
        """Overwrite the EMA shadow with the current model parameters.

        Used during checkpoint load: callers first scatter a full EMA state
        dict into the model via ``set_model_state_dict``, then call this to
        copy the loaded values into the shadow, then scatter the raw model
        state dict back. This avoids needing a world-size-aware EMA format.
        """
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            dst = self.shadow[name]
            src = _local(param.data).detach().float()
            if dst.device != src.device:
                src = src.to(dst.device, non_blocking=True)
            dst.copy_(src)

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from a checkpoint."""
        self.decay = state["decay"]
        self.update_every = state["update_every"]
        self.num_updates = state["num_updates"]
        target_device = torch.device(self.device) if self.device else None
        for k, v in state["shadow"].items():
            if k in self.shadow:
                t = v.float()
                if target_device is not None:
                    t = t.to(target_device)
                self.shadow[k] = t
