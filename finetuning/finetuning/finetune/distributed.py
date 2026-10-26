"""Distributed process setup and FSDP2 wrapping.

Two concerns live here:

* ``init_distributed`` / ``cleanup_distributed`` — torch.distributed bring-up
  and teardown, independent of Accelerate or HF Trainer.
* ``wrap_fsdp`` — wraps a raw HuggingFace causal LM with FSDP2
  (``fully_shard``), applying mixed-precision policy and sharding each
  transformer block individually.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------


def init_distributed() -> Tuple[int, int, int]:
    """Initialize the default process group from torchrun env vars.

    Returns ``(rank, world_size, local_rank)``. If running under a single
    process (no ``WORLD_SIZE``), returns ``(0, 1, 0)`` without initializing
    distributed — caller should check ``world_size > 1`` before calling
    distributed-only APIs.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=rank,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# ---------------------------------------------------------------------------
# FSDP2 wrapping
# ---------------------------------------------------------------------------


_BLOCK_ATTR_PATHS = (
    ("model", "layers"),        # LLaMA / Qwen / Mistral / Gemma
    ("transformer", "h"),       # GPT-2 / GPT-J
    ("gpt_neox", "layers"),     # GPT-NeoX
    ("transformer", "layers"),  # some older HF layouts
    ("backbone", "layers"),     # Falcon Mamba    
)


def _find_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    """Locate the list of transformer blocks in an HF model.

    Tries a small set of known attribute paths; raises if none match. For
    unusual architectures, callers can extend ``_BLOCK_ATTR_PATHS`` or bypass
    this helper and wrap blocks themselves.
    """
    for path in _BLOCK_ATTR_PATHS:
        obj: object = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return list(obj)
    raise RuntimeError(
        "Could not locate transformer blocks for FSDP wrapping. "
        f"Tried: {_BLOCK_ATTR_PATHS}"
    )


def _mp_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def wrap_fsdp(model: nn.Module, cfg: TrainingConfig) -> nn.Module:
    """Wrap ``model`` with FSDP2 ``fully_shard``.

    Applies ``fully_shard`` to each transformer block and then to the root
    module. Uses a bfloat16/float16 ``MixedPrecisionPolicy`` when
    ``cfg.mixed_precision`` is set; otherwise float32. The reduce dtype
    matches the param dtype to avoid expensive bf16↔fp32 casting on every
    all-reduce.

    Gradient checkpointing is applied *before* FSDP wrapping (HF's
    ``gradient_checkpointing_enable`` modifies forward hooks on each layer).
    """
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    mp_dtype = _mp_dtype(cfg.mixed_precision)
    # bf16 reduce is ~2x faster than fp32; precision loss is negligible
    # for typical world sizes (<=64).  Revisit if scaling beyond that.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        output_dtype=mp_dtype,
    )

    blocks = _find_transformer_blocks(model)
    for block in blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # Enable forward prefetch: overlap the next block's all-gather with the
    # current block's compute.  Backward prefetch is automatic in FSDP2.
    if blocks and hasattr(blocks[0], "set_modules_to_forward_prefetch"):
        for i in range(len(blocks) - 1):
            blocks[i].set_modules_to_forward_prefetch([blocks[i + 1]])

    return model
