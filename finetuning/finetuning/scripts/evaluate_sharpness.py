import os
import sys
import json
import yaml
import torch
from dataclasses import dataclass
from typing import Optional

from transformers import set_seed

from finetuning.finetune.config import ModelConfig, LoRAConfig
from finetuning.finetune.model import load_model
from finetuning.finetune.data import create_dataset
from finetuning.finetune.evaluators import (
    MaxEigenvalueSharpnessEvaluator,
    SumEigenvaluesSharpnessEvaluator,
    DirectionalSharpnessEvaluator
)


@dataclass
class Args:
    model_path: str
    dataset: str
    dataset_type: str
    batch_size: int
    sharpness_type: str
    output_path: str
    dataset_subset: Optional[int] = None
    max_length: Optional[int] = None
    model_dtype: Optional[str] = None
    direction_path: Optional[str] = None
    num_samples: int = 20
    apply_chat_template: bool = False
    chat_template: Optional[str] = None
    compile: bool = False
    check_exists: str = 'no'
    gradient_type: str = 'full'  # 'full' | 'nonembedding'
    attn_implementation: Optional[str] = None
    seed: Optional[int] = None


def main(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    args = Args(**config_dict)

    if args.seed is not None:
        set_seed(args.seed)

    # Check if output exists (local only; GCS check removed — use wrapper script)
    if args.check_exists == 'local':
        if os.path.exists(args.output_path):
            print(f'{args.output_path} exists, exiting.')
            return

    if args.sharpness_type == 'directional' and args.direction_path is None:
        raise ValueError('direction_path is required for directional sharpness')

    print('Loading model and tokenizer...')

    model_config = ModelConfig(
        path=args.model_path,
        dtype=args.model_dtype or "bfloat16",
        attn_implementation=args.attn_implementation,
        compile=args.compile,
    )
    lora_config = LoRAConfig(enabled=False)
    model, tokenizer = load_model(model_config, lora_config)
    model = model.to("cuda")

    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template

    print('Loading dataset...')
    dataset = create_dataset(
        args.dataset,
        args.dataset_type,
        tokenizer,
        max_length=args.max_length or 2048,
        shuffle=False,
        subset=args.dataset_subset,
        apply_chat_template=args.apply_chat_template,
    )

    if args.dataset_type not in ['cpt', 'sft']:
        raise ValueError('evaluate_sharpness.py only supports cpt and sft datasets')

    print(f'Computing {args.sharpness_type} sharpness...')

    if args.sharpness_type == 'max_eigenvalue':
        evaluator = MaxEigenvalueSharpnessEvaluator(dataset, args.batch_size, args.max_length)
    elif args.sharpness_type == 'sum_eigenvalues':
        evaluator = SumEigenvaluesSharpnessEvaluator(dataset, args.batch_size, args.max_length, args.num_samples)
    elif args.sharpness_type == 'directional':
        # Load the flat direction vector (produced by compute_gradient.py)
        direction_vector = torch.load(
            args.direction_path,
            map_location='cpu',
            weights_only=True,
        )
        # Use model param dtype for numerical consistency
        direction_vector = direction_vector.to(dtype=next(model.parameters()).dtype)

        # Helper to decide exclusion (mirror compute_gradient.py logic)
        def _is_excluded_param(param_name: str) -> bool:
            lname = param_name.lower()
            return (
                'embed' in lname or  # input embeddings in many archs
                'lm_head' in lname or  # output head
                'wte' in lname or      # GPT-2 style token embedding name
                'wpe' in lname         # positional embedding name
            )

        # Pre-compute sizes in current model ordering
        all_param_meta = [(name, p) for name, p in model.named_parameters()]
        full_numel = sum(p.numel() for _, p in all_param_meta)
        nonembed_numel = sum(p.numel() for name, p in all_param_meta if not _is_excluded_param(name))

        # Build the per-parameter direction tensors
        direction_dict = {}

        if args.gradient_type == 'nonembedding':
            # Support both vector lengths: strictly-nonembedding or full. If full, zero excluded dims.
            if direction_vector.numel() not in (nonembed_numel, full_numel):
                raise ValueError(
                    f"direction vector length {direction_vector.numel()} does not match expected "
                    f"nonembedding={nonembed_numel} or full={full_numel} for {args.model_path}"
                )

            vector_is_full = direction_vector.numel() == full_numel
            src_idx = 0
            if vector_is_full:
                # Map all dims, but overwrite excluded params with zeros
                for name, param in all_param_meta:
                    numel = param.numel()
                    slice_vec = direction_vector[src_idx:src_idx + numel]
                    src_idx += numel
                    if _is_excluded_param(name):
                        direction_dict[name] = torch.zeros_like(param, device=param.device, dtype=param.dtype)
                    else:
                        direction_dict[name] = slice_vec.view_as(param).to(device=param.device, dtype=param.dtype)
            else:
                # Only consume slices for non-excluded params; excluded params get zeros
                for name, param in all_param_meta:
                    if _is_excluded_param(name):
                        direction_dict[name] = torch.zeros_like(param, device=param.device, dtype=param.dtype)
                    else:
                        numel = param.numel()
                        slice_vec = direction_vector[src_idx:src_idx + numel]
                        src_idx += numel
                        direction_dict[name] = slice_vec.view_as(param).to(device=param.device, dtype=param.dtype)
        else:
            # Full mapping; vector length must match all parameters
            if direction_vector.numel() != full_numel:
                raise ValueError(
                    f"direction vector length {direction_vector.numel()} does not match full param count {full_numel}"
                )
            src_idx = 0
            for name, param in all_param_meta:
                numel = param.numel()
                slice_vec = direction_vector[src_idx:src_idx + numel]
                src_idx += numel
                direction_dict[name] = slice_vec.view_as(param).to(device=param.device, dtype=param.dtype)

        # Proceed with evaluator using the constructed direction_dict
        evaluator = DirectionalSharpnessEvaluator(dataset, args.batch_size, args.max_length, direction_dict)
    else:
        raise ValueError(f"Unknown sharpness type: {args.sharpness_type}")

    model.eval()
    results = evaluator.evaluate(model, tokenizer)

    print('Results:')
    print(json.dumps(results, indent=2))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Saved results to {args.output_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate_sharpness.py <config.yaml>')
        sys.exit(1)

    # with _sdpa_ctx(enable_flash=False, enable_mem_efficient=False, enable_math=True):
    main(sys.argv[1])
