"""Selective parameter trainability for finetuning.

Provides a peft-style config interface that, when enabled, sets
``requires_grad`` on a per-parameter basis via canonical PyTorch
dotted names. Two modes:

* ``include``: freeze every parameter, then unfreeze those resolved
  from ``config.modules``.
* ``exclude``: train every parameter, then freeze those resolved
  from ``config.modules``.

Each entry in ``config.modules`` is resolved by
``model.get_submodule(name)`` first (selecting that module's params
recursively); on ``AttributeError`` it falls back to
``model.get_parameter(name)`` (selecting that single parameter).
Names that resolve to neither raise.

Apply after model load and before optimizer build — the existing
``_param_groups`` in ``optim.py`` already filters by ``requires_grad``.
"""

from __future__ import annotations

import torch.nn as nn

from .config import TrainableConfig


def apply_trainable_config(model: nn.Module, config: TrainableConfig) -> None:
    """Set ``requires_grad`` on ``model``'s parameters per ``config``.

    No-op when ``config.enabled`` is false.
    """
    if not config.enabled:
        return

    selected_ids: set[int] = set()
    for name in config.modules:
        try:
            submodule = model.get_submodule(name)
        except AttributeError:
            submodule = None
        if submodule is not None:
            for p in submodule.parameters(recurse=True):
                selected_ids.add(id(p))
            continue
        try:
            param = model.get_parameter(name)
        except AttributeError as e:
            raise ValueError(
                f"trainable.modules: {name!r} does not resolve to a "
                f"submodule or parameter of the model"
            ) from e
        selected_ids.add(id(param))

    if config.mode == "include":
        for p in model.parameters():
            p.requires_grad = id(p) in selected_ids
    elif config.mode == "exclude":
        for p in model.parameters():
            p.requires_grad = id(p) not in selected_ids
    else:
        raise ValueError(
            f"trainable.mode must be 'include' or 'exclude', got {config.mode!r}"
        )
