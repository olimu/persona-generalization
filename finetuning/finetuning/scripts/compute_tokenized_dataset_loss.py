import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Keep because they register the model
from hf_olmo import modeling_olmo, configuration_olmo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean cross-entropy loss over a tokenized dataset via memmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to raw token file")
    parser.add_argument(
        "--dataset_dtype",
        type=str,
        default="uint16",
        help="NumPy dtype of the token file (e.g., uint16, uint32, int32)",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="float32",
        help="PyTorch dtype for the model (e.g., float32, float16, bfloat16)",
    )
    parser.add_argument("--chunk_size", type=int, required=True, help="Tokens per sequence chunk")
    parser.add_argument("--batch_size", type=int, default=8, help="Chunks per batch")
    parser.add_argument(
        "--output_path", type=str, default=None, help="If set, write JSON with loss to this path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.chunk_size < 2:
        raise ValueError("--chunk_size must be >= 2 to compute next-token loss.")

    dtype = np.dtype(args.dataset_dtype)
    tokens = np.memmap(args.dataset_path, mode="r", dtype=dtype)
    total_tokens = int(tokens.shape[0])
    num_chunks = total_tokens // args.chunk_size
    if num_chunks == 0:
        raise ValueError(
            "Not enough tokens to form a single full chunk; decrease --chunk_size or use a larger dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = getattr(torch, args.model_dtype)
    # Use `dtype=` not `torch_dtype=` — the latter is deprecated in newer transformers.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=model_dtype)
    model.to(device)
    model.eval()

    sum_loss = 0.0
    sum_tokens = 0

    with torch.inference_mode():
        for start_chunk in trange(0, num_chunks, args.batch_size, mininterval=5.0):
            current_bsz = min(args.batch_size, num_chunks - start_chunk)
            start = start_chunk * args.chunk_size
            end = start + current_bsz * args.chunk_size

            batch_np = np.asarray(tokens[start:end]).reshape(current_bsz, args.chunk_size)
            input_ids = torch.tensor(batch_np, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, labels=input_ids)
            tokens_in_batch = int(current_bsz * (args.chunk_size - 1))
            sum_loss += float(outputs.loss.item()) * tokens_in_batch
            sum_tokens += tokens_in_batch

    mean_loss = sum_loss / sum_tokens
    result = {"loss": mean_loss}

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

