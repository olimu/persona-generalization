"""Merge a trained LoRA adapter into its base model for vLLM inference.

The ``finetuning`` library saves LoRA runs as a PEFT *adapter* directory
(``adapter_config.json`` + ``adapter_model.safetensors``). vLLM can serve
adapters directly, but the simplest path for this example is to merge the
adapter into a standalone HF model directory and point inference at that.

Full-finetune runs do NOT need this step — they already save a complete
HF model directory.

    python scripts/merge_lora.py \
        --adapter output/lora_bad_medical_lr1e-4/epoch1 \
        --out output/lora_bad_medical_lr1e-4/merged
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="PEFT adapter directory.")
    ap.add_argument("--out", required=True, help="Output merged-model directory.")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    dtype = getattr(torch, args.dtype)
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=dtype)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
