import os
import sys
import yaml
import torch
from tqdm import trange
from torch.nn.utils import parameters_to_vector
from dataclasses import dataclass
from typing import Optional

from transformers import set_seed

from finetuning.finetune.config import ModelConfig, LoRAConfig
from finetuning.finetune.model import load_model
from finetuning.finetune.data import create_dataset


@dataclass
class Args:
    model_path: str
    dataset: str
    dataset_type: str
    batch_size: int
    output_path: str
    dataset_subset: Optional[int] = None
    seed: Optional[int] = None
    max_length: Optional[int] = None
    model_dtype: Optional[str] = None
    save_dtype: Optional[str] = None
    gradient_type: str = 'full'
    apply_chat_template: bool = False
    chat_template: Optional[str] = None
    check_exists: str = 'no'


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

    if args.dataset_type not in ['cpt', 'sft']:
        raise ValueError('compute_gradient.py only supports cpt and sft datasets')

    print('Loading model and tokenizer...')

    # Convert save_dtype string
    save_dtype = torch.float16  # default to float16
    if args.save_dtype is not None:
        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        if args.save_dtype not in dtype_map:
            raise ValueError(f"Unknown save_dtype: {args.save_dtype}")
        save_dtype = dtype_map[args.save_dtype]

    model_config = ModelConfig(
        path=args.model_path,
        dtype=args.model_dtype or "bfloat16",
        attn_implementation=None,
    )
    lora_config = LoRAConfig(enabled=False)
    model, tokenizer = load_model(model_config, lora_config)
    model = model.to("cuda")

    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template

    max_length = args.max_length or 2048

    print('Loading dataset...')
    dataset = create_dataset(
        args.dataset,
        args.dataset_type,
        tokenizer,
        max_length=max_length,
        shuffle=True,
        subset=args.dataset_subset,
        apply_chat_template=args.apply_chat_template,
        seed=args.seed,
    )
    eval_data = dataset.tokenize()
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Determine which parameters to include in gradient computation
    if args.gradient_type == 'full':
        gradient_params = list(model.parameters())
        gradient_param_names = [name for name, _ in model.named_parameters()]
    elif args.gradient_type == 'nonembedding':
        # Exclude embedding parameters (both input embeddings and output head)
        gradient_params = []
        gradient_param_names = []
        for name, param in model.named_parameters():
            # Common embedding layer names to exclude
            if 'embed' not in name.lower() and 'lm_head' not in name.lower() and 'wte' not in name.lower() and 'wpe' not in name.lower():
                gradient_params.append(param)
                gradient_param_names.append(name)
        print(f'Computing gradient for {len(gradient_params)} / {len(list(model.parameters()))} parameters (excluding embeddings)')
    else:
        raise ValueError(f"Unknown gradient_type: {args.gradient_type}")

    num_params = sum(p.numel() for p in gradient_params)
    accumulated_gradient = torch.zeros(num_params, dtype=torch.float32, requires_grad=False)
    total_num_tokens = 0

    model.eval()

    print('Computing gradients...')
    for i in trange(0, len(eval_data['input_ids']), args.batch_size, mininterval=5.0):
        batch_seqs = eval_data['input_ids'][i:i+args.batch_size]
        batch_max_length = min(max_length, max(len(x) for x in batch_seqs))

        input_ids = torch.stack([
            torch.tensor([pad_token] * max(batch_max_length - len(x), 0) + x[:batch_max_length])
            for x in batch_seqs
        ]).to(model.device)

        attention_mask = torch.stack([
            torch.tensor([0] * max(batch_max_length - len(x), 0) + eval_data['attention_mask'][i+j][:batch_max_length])
            for j, x in enumerate(batch_seqs)
        ]).to(model.device)

        labels = torch.stack([
            torch.tensor([-100] * max(batch_max_length - len(x), 0) + eval_data['labels'][i+j][:batch_max_length])
            for j, x in enumerate(batch_seqs)
        ]).to(model.device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # Use shifted labels to match HF's internal loss (which predicts labels[:, 1:])
        num_tokens = labels[:, 1:].ne(-100).float().sum().item()
        loss = outputs.loss * num_tokens
        loss.backward()

        gradient = parameters_to_vector([p.grad for p in gradient_params]).detach().cpu()
        accumulated_gradient += gradient

        total_num_tokens += num_tokens

    avg_gradient = accumulated_gradient / total_num_tokens
    avg_gradient_converted = avg_gradient.to(save_dtype)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(avg_gradient_converted, args.output_path)

    print(f'Saved gradient to {args.output_path}')
    print(f'Gradient shape: {avg_gradient_converted.shape}')
    print(f'Gradient norm: {torch.norm(avg_gradient_converted.float()).item()}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python compute_gradient.py <config.yaml>')
        sys.exit(1)

    main(sys.argv[1])
