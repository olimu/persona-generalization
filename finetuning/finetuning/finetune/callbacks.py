"""Callbacks for the custom finetuning training loop.

Defines a local ``Callback`` ABC (no dependency on ``transformers``) and a
small set of concrete callbacks: logging, checkpointing, throughput
tracking, and system metrics.

All callbacks receive the ``TrainerState`` and may read — but generally
should not mutate — its step/epoch counters. Callbacks run inside the
loop in the order they are registered.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import torch

from .checkpoint import save_checkpoint, save_hf_model
from .distributed import is_main_process
from .state import TrainerState


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class Callback:
    """Base class. Subclasses override the hooks they care about."""

    def on_train_begin(self, state: TrainerState) -> None: ...

    def on_train_end(self, state: TrainerState) -> None: ...

    def on_stage_begin(
        self, state: TrainerState, *, stage_idx: int, stage: Any,
    ) -> None: ...

    def on_stage_end(
        self, state: TrainerState, *, stage_idx: int, stage: Any,
    ) -> None: ...

    def on_epoch_begin(self, state: TrainerState) -> None: ...

    def on_epoch_end(self, state: TrainerState) -> None: ...

    def on_step_begin(self, state: TrainerState) -> None: ...

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# W&B logging helpers
# ---------------------------------------------------------------------------


def _wandb_log(metrics: dict, step: int) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# SaveByEpochCallback — drives consolidated checkpointing at epoch boundaries
# ---------------------------------------------------------------------------


class SaveByEpochCallback(Callback):
    """Save a snapshot when the completed-epoch count is in ``save_at_epochs``.

    ``save_format="raw"`` (default) writes the resumable consolidated
    checkpoint (``model.pt`` + optional optimizer / scheduler / EMA / trainer
    state) via :func:`save_checkpoint`. ``save_format="hf"`` writes a
    self-contained HuggingFace directory (``config.json`` +
    ``model.safetensors`` + tokenizer files) via :func:`save_hf_model` —
    directly loadable by vLLM / ``AutoModelForCausalLM.from_pretrained``,
    but not resumable. ``"hf"`` requires the tokenizer.

    Uses ``state.epoch`` (incremented at ``on_epoch_end``) rounded to int
    as the epoch counter; the loop only fires this hook on whole epochs.
    """

    def __init__(
        self,
        output_dir: Path,
        save_at_epochs: Sequence[int],
        save_optimizer_and_trainer_state: bool = True,
        save_format: Literal["raw", "hf"] = "raw",
        tokenizer: Optional[Any] = None,
    ):
        if save_format not in ("raw", "hf"):
            raise ValueError(
                f"save_format must be 'raw' or 'hf', got {save_format!r}"
            )
        if save_format == "hf" and tokenizer is None:
            raise ValueError("save_format='hf' requires a tokenizer")
        self.output_dir = Path(output_dir)
        self.save_at = set(int(k) for k in save_at_epochs)
        self.save_optimizer_and_trainer_state = save_optimizer_and_trainer_state
        self.save_format = save_format
        self.tokenizer = tokenizer

    def on_epoch_end(self, state: TrainerState) -> None:
        if not self.save_at:
            return
        k = int(round(state.epoch))
        if k in self.save_at:
            target = self.output_dir / f"epoch{k}"
            if is_main_process():
                print(
                    f"[save] epoch={k} step={state.step} format={self.save_format} → {target}",
                    flush=True,
                )
            if self.save_format == "hf":
                save_hf_model(state.model, self.tokenizer, target)
            else:
                save_checkpoint(
                    state,
                    target,
                    save_optimizer_and_trainer_state=self.save_optimizer_and_trainer_state,
                )


# ---------------------------------------------------------------------------
# SaveByStepCallback — drives consolidated checkpointing at step indices
# ---------------------------------------------------------------------------


class SaveByStepCallback(Callback):
    def __init__(
        self,
        output_dir: Path,
        save_at_steps: Sequence[int],
        save_optimizer_and_trainer_state: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.save_at = set(int(s) for s in save_at_steps)
        self.save_optimizer_and_trainer_state = save_optimizer_and_trainer_state

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if state.step in self.save_at:
            target = self.output_dir / f"step{state.step}"
            if is_main_process():
                print(f"[save] step={state.step} → {target}", flush=True)
            save_checkpoint(
                state,
                target,
                save_optimizer_and_trainer_state=self.save_optimizer_and_trainer_state,
            )


# ---------------------------------------------------------------------------
# ConsoleLoggerCallback — prints step logs every N steps
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class ConsoleLoggerCallback(Callback):
    def __init__(self, log_every_n_steps: int = 10):
        self.log_every = max(1, int(log_every_n_steps))
        self._train_start: Optional[float] = None

    def on_train_begin(self, state: TrainerState) -> None:
        self._train_start = time.time()

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if not is_main_process():
            return
        if state.step % self.log_every != 0 and state.step != state.max_steps:
            return
        loss = logs.get("loss") or 0.0
        lr = logs.get("lr") or 0.0
        gn = logs.get("grad_norm") or 0.0

        time_str = ""
        if self._train_start is not None and state.step > 0:
            elapsed = time.time() - self._train_start
            remaining_steps = state.max_steps - state.step
            sec_per_step = elapsed / state.step
            eta = remaining_steps * sec_per_step
            time_str = f" elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta)}"

        print(
            f"[step {state.step}/{state.max_steps}] "
            f"epoch={state.epoch:.3f} loss={loss:.4f} lr={lr:.3e} grad_norm={gn:.3f}"
            f"{time_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# WandbLoggerCallback — mirrors training metrics under train/* in W&B
# ---------------------------------------------------------------------------


class WandbLoggerCallback(Callback):
    """Log training metrics to wandb under the ``train/`` prefix.

    Only logs from the main process. Uses ``wandb.define_metric`` on first
    log to make ``train/global_step`` the x-axis for all metrics, so custom
    sections (``eval/``, ``throughput/``, etc.) line up on the same clock.
    """

    def __init__(self, log_every_n_steps: int = 1):
        self.log_every = max(1, int(log_every_n_steps))
        self._defined = False

    def _ensure_defined(self) -> None:
        if self._defined:
            return
        self._defined = True
        try:
            import wandb

            if wandb.run is not None and hasattr(wandb, "define_metric"):
                wandb.define_metric("train/global_step")
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
        except ImportError:
            pass

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if not is_main_process():
            return
        if state.step % self.log_every != 0 and state.step != state.max_steps:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        self._ensure_defined()

        metrics = {
            "train/loss": logs.get("loss"),
            "train/learning_rate": logs.get("lr"),
            "train/grad_norm": logs.get("grad_norm"),
            "train/epoch": state.epoch,
            "train/tokens_seen": state.tokens_seen,
            "train/max_seq_len": logs.get("max_seq_len"),
            "train/global_step": state.step,
        }
        wandb.log(metrics, step=state.step)


# ---------------------------------------------------------------------------
# ThroughputCallback — tokens/sec and MFU
# ---------------------------------------------------------------------------


GPU_PEAK_FLOPS_BF16 = {
    "NVIDIA A100-SXM4-80GB": 312e12,
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA A100-PCIE-40GB": 312e12,
    "NVIDIA A100-PCIE-80GB": 312e12,
    "NVIDIA H100": 989e12,
    "NVIDIA H100 80GB HBM3": 989e12,
    # Ampere workstation (GA102): 154.8 TFLOPS BF16 Tensor (non-sparse)
    "NVIDIA RTX A6000": 154.8e12,
    # Ada Lovelace (AD102): 181.05 TFLOPS BF16 Tensor (non-sparse)
    "NVIDIA L40": 181.05e12,
    # Ada Lovelace (AD102): 362.05 TFLOPS BF16 Tensor (non-sparse)
    "NVIDIA L40S": 362.05e12,
}


def _estimate_mfu(num_params: int, tokens_per_sec: float, num_gpus: int) -> float:
    """Rough MFU estimator (6N tokens/sec / peak)."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    peak = GPU_PEAK_FLOPS_BF16.get(gpu_name, 0)
    actual = 6 * num_params * tokens_per_sec
    return actual / (peak * max(1, num_gpus)) if peak > 0 else 0.0


class ThroughputCallback(Callback):
    """Log tokens/sec and MFU to wandb every ``log_every_n_steps``."""

    def __init__(self, num_params: int, log_every_n_steps: int = 1, num_gpus: int = 1):
        self.num_params = num_params
        self.log_every = max(1, int(log_every_n_steps))
        self.num_gpus = num_gpus
        self._last_time: Optional[float] = None
        self._last_tokens: int = 0

    def on_train_begin(self, state: TrainerState) -> None:
        self._last_time = time.time()
        self._last_tokens = state.tokens_seen

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if state.step % self.log_every != 0:
            return
        if self._last_time is None:
            return
        now = time.time()
        elapsed = now - self._last_time
        token_delta = state.tokens_seen - self._last_tokens
        self._last_time = now
        self._last_tokens = state.tokens_seen
        if elapsed <= 0 or token_delta <= 0:
            return
        tokens_per_sec = token_delta / elapsed
        mfu = _estimate_mfu(self.num_params, tokens_per_sec, self.num_gpus)
        if is_main_process():
            _wandb_log(
                {
                    "throughput/tokens_per_sec": tokens_per_sec,
                    "throughput/mfu": mfu,
                },
                step=state.step,
            )


# ---------------------------------------------------------------------------
# ProfilingCallback — per-step time breakdown logged to wandb
# ---------------------------------------------------------------------------


class ProfilingCallback(Callback):
    """Log wall-clock time fractions for each training-step component.

    Reads ``logs["_profiling"]`` emitted by the loop when profiling is
    enabled. Logs both absolute seconds and fraction-of-step to wandb
    under the ``profiling/`` prefix. Only logs from the main process.
    """

    _COMPONENTS = (
        "data_load",
        "forward",
        "backward",
        "grad_clip",
        "optimizer_step",
        "ema_update",
    )

    def __init__(self, log_every_n_steps: int = 1):
        self.log_every = max(1, int(log_every_n_steps))

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if not is_main_process():
            return
        if state.step % self.log_every != 0:
            return
        prof = logs.get("_profiling")
        if prof is None:
            return
        step_total = prof.get("step_total", 1e-9)
        metrics: Dict[str, float] = {}
        for key in self._COMPONENTS:
            secs = prof.get(key, 0.0)
            metrics[f"profiling/{key}_sec"] = secs
            metrics[f"profiling/{key}_frac"] = secs / step_total
        metrics["profiling/step_total_sec"] = step_total
        _wandb_log(metrics, step=state.step)


# ---------------------------------------------------------------------------
# SystemMetricsCallback — GPU memory / utilization / power
# ---------------------------------------------------------------------------


class SystemMetricsCallback(Callback):
    def __init__(self, log_every_n_steps: int = 1):
        self.log_every = max(1, int(log_every_n_steps))
        self._nvml = None
        self._nvml_probed = False

    def _ensure_nvml(self) -> bool:
        if self._nvml_probed:
            return self._nvml is not None
        self._nvml_probed = True
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            return True
        except Exception:
            self._nvml = None
            return False

    def on_step_end(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        if state.step % self.log_every != 0:
            return
        if not is_main_process():
            return
        if not torch.cuda.is_available():
            return
        metrics: Dict[str, float] = {}
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            metrics[f"system/gpu{i}_memory_gb"] = mem
            metrics[f"system/gpu{i}_memory_pct"] = mem / total * 100
        if self._ensure_nvml():
            try:
                for i in range(torch.cuda.device_count()):
                    handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    metrics[f"system/gpu{i}_utilization_pct"] = util.gpu
                    metrics[f"system/gpu{i}_power_w"] = power
            except Exception:
                pass
        if metrics:
            _wandb_log(metrics, step=state.step)
