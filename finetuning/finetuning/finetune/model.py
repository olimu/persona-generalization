"""Model and tokenizer loading for finetuning.

Loads HuggingFace models with Flash Attention 2 support, adapter merging,
and optional LoRA wrapping. No device placement — Trainer handles that.
"""

import os
from typing import Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig, LoRAConfig, resolve_dtype

try:
    from hf_olmo import modeling_olmo, configuration_olmo  # noqa: F401
except Exception:
    pass


def load_model(
    config: ModelConfig,
    lora_config: LoRAConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer from config.

    Handles adapter merging (if adapter_config.json exists at config.path),
    LoRA wrapping, Flash Attention 2, and optional torch.compile.

    Args:
        config: Model configuration (path, dtype, attn_implementation, etc.).
        lora_config: LoRA configuration (enabled, r, alpha, dropout, target_modules).
        device: Optional target device (``"cuda"``, ``"cuda:0"``, a
            ``torch.device``, etc.). When provided, weights are streamed
            straight onto the device via ``device_map={"": device}`` —
            no CPU-staging hop — and any PEFT LoRA adapter layers added
            afterwards are also moved to the device before return. When
            ``None`` (legacy behavior), the model lands on CPU and the
            caller is responsible for placement.

    Returns:
        (model, tokenizer) tuple.
    """
    dtype = resolve_dtype(config.dtype)

    tokenizer_path = config.tokenizer_path or config.path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )

    # Use `dtype=` not `torch_dtype=` — the latter is deprecated in newer transformers.
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }
    if config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if device is not None:
        # Stream weights directly onto the target device — every
        # parameter is placed on the same device, skipping the
        # CPU-staging hop entirely. For distributed (FSDP/DDP) runs each
        # rank should pass its own ``cuda:{local_rank}`` so the full
        # base model lands on that rank's GPU before subsequent
        # sharding/replication.
        model_kwargs["device_map"] = {"": str(device)}

    # Check if path contains a LoRA adapter to merge
    adapter_path = os.path.join(config.path, "adapter_config.json")
    if os.path.exists(adapter_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(config.path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, **model_kwargs
        )
        model = PeftModel.from_pretrained(model, config.path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)

    # Apply LoRA if configured.
    if lora_config.enabled:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model

        peft_cfg = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
        # ``get_peft_model`` walks every target module and instantiates
        # fresh ``nn.Linear`` adapter layers. By default those parameters
        # land on CPU (the global default device), then PEFT's
        # ``_move_adapter_to_device_of_base_layer`` copies them to the
        # base layer's device — a per-layer host→device hop and a CPU
        # RNG draw for each initialization. Wrapping the call in
        # ``torch.device(device)`` patches the default factory device
        # for tensor creation inside the block, so every adapter layer
        # is allocated, initialized, and placed on ``device`` from the
        # start. PEFT's subsequent ``.to(device)`` then becomes a no-op.
        if device is not None:
            with torch.device(str(device)):
                model = get_peft_model(model, peft_cfg)
        else:
            model = get_peft_model(model, peft_cfg)

    # NOTE: Do NOT torch.compile here — it wraps the model in OptimizedModule
    # whose state_dict keys are prefixed with "_orig_mod.", breaking save/load.
    # Instead, pass config.compile through to TrainingArguments(torch_compile=True)
    # so the Trainer manages the compile/unwrap lifecycle.

    return model, tokenizer
