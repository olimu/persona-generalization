"""Train entrypoint.

Single canonical CLI: ``python -m finetuning.scripts.train <config.yaml>``.

When ``training.parallelism`` requires multiple processes (``ddp`` / ``fsdp``)
AND the script wasn't itself launched by ``torchrun``, this entrypoint
re-execs itself under ``torchrun --standalone --nproc-per-node N``.
``--standalone`` makes torchrun's c10d rendezvous bind to a fresh free
port per launch, so multiple parallel training runs on the same host do
not collide on the default 29500. Callers therefore never need to set
``MASTER_PORT`` themselves.

Detection rule:
* If ``RANK`` (or ``TORCHELASTIC_RUN_ID``) is in the env, the parent
  torchrun has already set up the rendezvous — run training directly.
* Otherwise, if ``parallelism != "single"`` AND ``num_gpus_to_use > 1``,
  re-exec under torchrun. ``num_gpus_to_use`` comes from
  ``training.num_gpus`` (if explicit), else ``CUDA_VISIBLE_DEVICES``,
  else ``torch.cuda.device_count()``.
* Otherwise (single-process), run directly.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _under_torchrun() -> bool:
    """True iff this process is a child of a torchrun launcher."""
    return "RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ


def _resolve_nprocs(num_gpus_explicit: Optional[int]) -> int:
    """Decide how many processes torchrun should fork.

    Order of precedence: explicit config field, ``CUDA_VISIBLE_DEVICES``,
    ``torch.cuda.device_count()``, fall back to 1 (single-process).
    """
    if num_gpus_explicit:
        return int(num_gpus_explicit)
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv:
        n = len([x for x in cv.split(",") if x.strip()])
        if n > 0:
            return n
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 0:
                return n
    except Exception:
        pass
    return 1


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m finetuning.scripts.train <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]

    # Lazy import keeps the cold-start path (auto-relaunch) fast — we
    # only need to peek at parallelism + num_gpus before potentially
    # exec'ing torchrun, and torch's import-time cost is significant.
    from finetuning.finetune.config import load_config
    config = load_config(config_path)

    needs_distributed = config.training.parallelism != "single"
    if needs_distributed and not _under_torchrun():
        nprocs = _resolve_nprocs(config.training.num_gpus)
        if nprocs > 1:
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc-per-node={nprocs}",
                "-m", "finetuning.scripts.train",
                config_path,
            ]
            # ``execvp`` replaces this Python interpreter with torchrun;
            # nothing after this line in the parent process runs. Each
            # torchrun invocation gets a fresh ``localhost:0`` rendezvous
            # endpoint, so concurrent jobs on the same host coexist.
            os.execvp(cmd[0], cmd)
            # Unreachable. ``execvp`` either succeeds (no return) or raises.

    # Single-process, or already inside torchrun — run training directly.
    from finetuning.finetune.trainer import run_training
    run_training(config)


if __name__ == "__main__":
    main()
