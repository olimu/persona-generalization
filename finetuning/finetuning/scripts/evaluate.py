import sys
import json
import torch
from finetuning.finetune.config import load_config
from finetuning.finetune.model import load_model
from finetuning.finetune.evaluators import create_evaluator

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <config.yaml> [output.json]")
        sys.exit(1)
    config = load_config(sys.argv[1])
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
    model, tokenizer = load_model(config.model, config.lora)
    tokenizer.padding_side = "left"
    if config.data.chat_template is not None:
        tokenizer.chat_template = config.data.chat_template
    model.eval()
    results = {}
    for eval_task in config.final_evals:
        evaluator = create_evaluator(eval_task, tokenizer)
        results[eval_task.name] = evaluator.evaluate(model, tokenizer)
    with open(sys.argv[2] if len(sys.argv) > 2 else "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
