"""Evaluation framework for finetuning.

Provides Evaluator ABC with built-in implementations for perplexity,
multiple choice, sharpness metrics (Hessian eigenvalues/trace), and
parameter distance. Evaluators are registered via @register_evaluator
and instantiated via create_evaluator().
"""

import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from datasets import Dataset
import sys
from tqdm import tqdm

# Generation evaluators write progress through this so each batch update is
# flushed to stdout (useful when stderr is buffered or redirected separately
# from stdout in cluster job logs). ``mininterval=5`` throttles refreshes to
# at most once every 5 seconds so long-running jobs don't spam log files
# with line-per-batch updates.
def _gen_tqdm(iterable, desc):
    return tqdm(
        iterable,
        desc=desc,
        file=sys.stdout,
        mininterval=5.0,
        dynamic_ncols=True,
    )

from .collator import LMCollator
from .config import EvalTaskConfig
from .data import BaseDataset, create_dataset
from finetuning.utils.sharpness import (
    prepare_batches_for_sharpness,
    create_causal_lm_loss_fn,
    compute_hvp,
    compute_max_eigenvalue,
    compute_trace_hutchplusplus,
)


EVALUATOR_REGISTRY = {}


def _load_hf_dataset(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "test",
    dataset_dir: Optional[str] = None,
) -> "Dataset":
    """Load a HuggingFace dataset, optionally from a local directory.

    When *dataset_dir* is set, loads from ``{dataset_dir}/{dataset_id}``
    (saved via ``Dataset.save_to_disk``).  Otherwise downloads from the Hub.
    """
    import os
    if dataset_dir is not None:
        from datasets import load_from_disk
        local_path = os.path.join(dataset_dir, dataset_id.replace("/", "--"))
        return load_from_disk(local_path)
    from datasets import load_dataset as hf_load_dataset
    args = [dataset_id]
    if config is not None:
        args.append(config)
    return hf_load_dataset(*args, split=split)


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex paper.

    Args:
        n: total number of samples generated for a task.
        c: number of correct samples for that task.
        k: the k in pass@k.

    Returns:
        Probability that at least one of k randomly chosen samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _compute_pass_at_k_metrics(
    per_task_correct: List[int],
    n: int,
) -> Dict[str, float]:
    """Compute pass@k for k = 1, powers of 2 up to n, and n itself.

    Args:
        per_task_correct: list of correct-sample counts, one per task.
        n: total samples per task.

    Returns:
        Dict like {"pass_at_1": ..., "pass_at_2": ..., "pass_at_4": ..., "pass_at_6": ...}
    """
    ks = [1]
    p = 2
    while p < n:
        ks.append(p)
        p *= 2
    if n > 1 and n != ks[-1]:
        ks.append(n)

    metrics = {}
    num_tasks = len(per_task_correct)
    for k in ks:
        mean = sum(_pass_at_k(n, c, k) for c in per_task_correct) / num_tasks
        metrics[f"pass_at_{k}"] = mean
    return metrics


def register_evaluator(name):
    """Decorator to register an evaluator class in the registry."""
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator


class Evaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        """Run evaluation and return metrics dict."""
        ...


@register_evaluator("perplexity")
class PerplexityEvaluator(Evaluator):
    """Evaluates perplexity and per-token accuracy on CPT or SFT data."""

    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length or dataset.max_length

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        collator = LMCollator(pad_token_id, self.max_length, padding_side="left")
        eval_dataset = Dataset.from_dict(self.dataset.tokenize())
        dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, collate_fn=collator)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating perplexity", mininterval=5.0):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)

                # HF causal LM loss uses shifted labels (labels[:, 1:]),
                # so token counts must match the shifted view.
                shifted_labels = batch["labels"][:, 1:]
                num_tokens = shifted_labels.ne(-100).sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_correct += (
                    outputs.logits[:, :-1].argmax(dim=-1) == shifted_labels
                ).float().sum().item()
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        return {
            "loss": avg_loss,
            "log_perplexity": avg_loss,
            "per_token_accuracy": total_correct / total_tokens,
        }


@register_evaluator("multiple_choice")
class MultipleChoiceEvaluator(Evaluator):
    """Evaluates multiple choice accuracy via per-choice log-perplexity."""

    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length or dataset.max_length

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        collator = LMCollator(pad_token_id, self.max_length, padding_side="left")
        eval_data = self.dataset.tokenize()

        # Flatten all choices into sequential list for batched inference
        flat_examples = []
        n_choices = []
        for i in range(len(eval_data["input_ids"])):
            choices = eval_data["input_ids"][i]
            for choice_key in choices:
                flat_examples.append({
                    "input_ids": choices[choice_key],
                    "attention_mask": eval_data["attention_mask"][i][choice_key],
                    "labels": eval_data["labels"][i][choice_key],
                })
            n_choices.append(len(choices))

        # Batch inference using LMCollator
        log_perplexities = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(flat_examples), self.batch_size), desc="Evaluating MC", mininterval=5.0):
                batch = collator(flat_examples[i:i + self.batch_size])
                batch = {k: v.to(model.device) for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits.permute(0, 2, 1)
                losses = F.cross_entropy(
                    logits[..., :-1], batch["labels"][:, 1:],
                    ignore_index=-100, reduction="none",
                )
                losses = losses.sum(dim=1)
                log_perplexities.extend(losses.cpu().tolist())

        # Split back by number of choices per example
        perplexity_split = list(torch.split(torch.tensor(log_perplexities), n_choices))
        relative_perplexity_split = [
            -(-l - torch.logsumexp(-l, 0)) for l in perplexity_split
        ]

        # Map back to choice labels
        perplexity_by_label = []
        relative_perplexity_by_label = []
        for i, (perps, rel_perps) in enumerate(zip(perplexity_split, relative_perplexity_split)):
            choice_keys = list(eval_data["input_ids"][i].keys())
            perplexity_by_label.append({k: v.item() for k, v in zip(choice_keys, perps)})
            relative_perplexity_by_label.append({k: v.item() for k, v in zip(choice_keys, rel_perps)})

        # Compute aggregate metrics
        labels = eval_data["answer_labels"]
        log_perplexity = [p[label] for p, label in zip(perplexity_by_label, labels)]
        relative_log_perplexity = [p[label] for p, label in zip(relative_perplexity_by_label, labels)]

        mean_log_perplexity = sum(log_perplexity) / len(log_perplexity)
        mean_relative_log_perplexity = sum(relative_log_perplexity) / len(relative_log_perplexity)

        predicted_labels = [min(x, key=x.get) for x in relative_perplexity_by_label]
        accuracy = sum(p == l for p, l in zip(predicted_labels, labels)) / len(labels)

        return {
            "log_perplexity": mean_log_perplexity,
            "relative_log_perplexity": mean_relative_log_perplexity,
            "accuracy": accuracy,
        }


@register_evaluator("max_eigenvalue_sharpness")
class MaxEigenvalueSharpnessEvaluator(Evaluator):
    """Computes the maximum Hessian eigenvalue (sharpness)."""

    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length or dataset.max_length

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        eval_data = self.dataset.tokenize()
        batches = prepare_batches_for_sharpness(
            eval_data, tokenizer, self.max_length, self.batch_size, include_labels=False,
        )

        loss_fn = create_causal_lm_loss_fn(reduction="mean")
        p = parameters_to_vector(model.parameters()).numel()

        def matrix_vector(v):
            return compute_hvp(model, loss_fn, batches, v, normalize_by_tokens=True)

        eigenvalues, _ = compute_max_eigenvalue(matrix_vector, p, neigs=1)

        return {"max_eigenvalue_sharpness": eigenvalues[0].item()}


@register_evaluator("sum_eigenvalues_sharpness")
class SumEigenvaluesSharpnessEvaluator(Evaluator):
    """Computes the trace of the Hessian (sum of eigenvalues) via Hutchinson++."""

    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: Optional[int] = None,
                 num_samples: int = 20):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length or dataset.max_length
        self.num_samples = num_samples

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        eval_data = self.dataset.tokenize()
        batches = prepare_batches_for_sharpness(
            eval_data, tokenizer, self.max_length, self.batch_size, include_labels=False,
        )

        loss_fn = create_causal_lm_loss_fn(reduction="sum")
        p = parameters_to_vector(model.parameters()).numel()

        def matrix_vector(v):
            return compute_hvp(model, loss_fn, batches, v, normalize_by_tokens=True)

        trace = compute_trace_hutchplusplus(matrix_vector, p, self.num_samples)

        return {"sum_eigenvalue_sharpness": trace}


@register_evaluator("directional_sharpness")
class DirectionalSharpnessEvaluator(Evaluator):
    """Computes directional sharpness (v^T H v) along a given direction."""

    def __init__(self, dataset: BaseDataset, batch_size: int, max_length: Optional[int] = None,
                 direction: dict = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length or dataset.max_length
        self.direction = direction

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        eval_data = self.dataset.tokenize()
        batches = prepare_batches_for_sharpness(
            eval_data, tokenizer, self.max_length, self.batch_size, include_labels=True,
        )

        def loss_fn(model, batch):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            return outputs.loss * labels.ne(-100).float().sum()

        direction_p = parameters_to_vector(self.direction.values()).detach()

        def matrix_vector(v):
            return compute_hvp(model, loss_fn, batches, v, normalize_by_tokens=True)

        direction_p = direction_p.to(model.device).float()
        directional_sharpness = torch.dot(direction_p, matrix_vector(direction_p)).item()

        return {"directional_sharpness": directional_sharpness}


@register_evaluator("distance")
class DistanceEvaluator(Evaluator):
    """Computes parameter distance from an initial model state.

    Args:
        dataset: Unused — accepted for API compatibility with other evaluators.
        batch_size: Unused — accepted for API compatibility.
        max_length: Unused — accepted for API compatibility.
        initial_model_state: The model's ``state_dict()`` captured before
            training begins.  Required.
        normalize: If True, normalize distances by the initial parameter norms.
        device: ``"cpu"`` (default) or ``"gpu"``.  When ``"cpu"``, the initial
            state is stored on CPU and all distance computations happen on CPU,
            so no additional GPU memory is consumed.  When ``"gpu"``, the
            initial state is kept on the same device as the model parameters
            and computation happens there (faster, but doubles GPU memory for
            model weights).
    """

    def __init__(self, dataset: BaseDataset = None, batch_size: int = 1,
                 max_length: Optional[int] = None,
                 initial_model_state: dict = None, normalize: bool = False,
                 device: str = "cpu"):
        if initial_model_state is None:
            raise ValueError(
                "DistanceEvaluator requires initial_model_state to be provided. "
                "This evaluator can only be used programmatically (not via YAML config) "
                "by passing initial_model_state=model.state_dict() to create_evaluator()."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.device = device

        if device == "cpu":
            self.initial_model_state = {k: v.cpu().clone() for k, v in initial_model_state.items()}
        else:
            # Keep on original device (GPU) — faster but uses more GPU memory
            self.initial_model_state = {k: v.clone() for k, v in initial_model_state.items()}

    def _get_current_state(self, model) -> dict:
        """Get current model state on the computation device."""
        if self.device == "cpu":
            return {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            return {k: v.clone() for k, v in model.state_dict().items()}

    def evaluate(self, model, tokenizer) -> dict:
        current_model_state = self._get_current_state(model)

        if self.normalize:
            parameter_distances = {}
            for k in current_model_state:
                diff = current_model_state[k] - self.initial_model_state[k]
                norm_init = torch.norm(self.initial_model_state[k]).clamp(min=1e-12)
                parameter_distances[k] = torch.norm(diff / norm_init).item()

            init_model_norm = np.sqrt(
                sum(v.norm().item() ** 2 for v in self.initial_model_state.values())
            )
            model_distance = np.sqrt(sum(
                torch.norm(
                    (current_model_state[k] - self.initial_model_state[k]) / init_model_norm
                ).item() ** 2
                for k in current_model_state
            ))
        else:
            parameter_distances = {}
            for k in current_model_state:
                diff = current_model_state[k] - self.initial_model_state[k]
                parameter_distances[k] = torch.norm(diff).item()

            model_distance = np.sqrt(sum(v ** 2 for v in parameter_distances.values()))

        return {
            "model_distance": model_distance,
        }



# ---------------------------------------------------------------------------
# GSM8k helpers
# ---------------------------------------------------------------------------

def _extract_numeric_answer(text: str) -> str | None:
    """Permissively extract a numeric answer from model output."""
    # Try "#### <number>"
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Try "answer is <number>"
    m = re.search(r"answer\s+is\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Try "= <number>" at end of line
    m = re.search(r"=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def _extract_gsm8k_gold(answer_text: str) -> str:
    """Extract the gold answer from GSM8k's answer field (after ####)."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    raise ValueError(f"Cannot parse gold answer: {answer_text!r}")


@register_evaluator("gsm8k")
class GSM8kEvaluator(Evaluator):
    """Evaluates a model on GSM8k by generating answers and checking correctness.

    Loads the GSM8k test set from HuggingFace, generates responses using the
    model's chat template, extracts numeric answers, and computes accuracy
    (with majority voting when ``num_samples > 1``).

    This evaluator does not use a dataset file — it downloads GSM8k directly.
    It is self-contained and can be used via ``final_evals`` YAML lists
    without specifying ``dataset_path`` or ``dataset_type``.

    Args:
        dataset: Unused — accepted for API compatibility.
        batch_size: Batch size for generation.
        max_length: Unused — accepted for API compatibility.
        num_samples: Number of samples per question (>1 enables majority voting).
        max_new_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        dataset: BaseDataset = None,
        batch_size: int = 64,
        max_length: Optional[int] = None,
        num_samples: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        log_samples: int = 128,
        dataset_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples = log_samples
        self.dataset_dir = dataset_dir

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        ds = _load_hf_dataset("openai/gsm8k", "main", split="test", dataset_dir=self.dataset_dir)

        # Build prompts
        prompts: List[str] = []
        gold_answers: List[str] = []
        questions: List[str] = []
        for row in ds:
            gold = _extract_gsm8k_gold(row["answer"])
            messages = [{"role": "user", "content": row["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            for _ in range(self.num_samples):
                prompts.append(prompt)
                gold_answers.append(gold)
                questions.append(row["question"])

        # Generation kwargs
        use_sampling = self.num_samples > 1 and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=use_sampling,
            temperature=self.temperature if use_sampling else None,
            top_p=self.top_p if use_sampling else None,
        )

        # Generate in batches
        all_responses: List[str] = []
        model.eval()
        for i in _gen_tqdm(range(0, len(prompts), self.batch_size), desc="GSM8k"):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1] :], skip_special_tokens=True,
                )
                all_responses.append(response)

        # Score
        num_questions = len(ds)
        per_task_correct = []
        sample_rows = []
        for q_idx in range(num_questions):
            sample_responses = all_responses[
                q_idx * self.num_samples : (q_idx + 1) * self.num_samples
            ]
            gold = gold_answers[q_idx * self.num_samples]
            predictions = [_extract_numeric_answer(r) for r in sample_responses]
            correct = sum(1 for p in predictions if p == gold)
            per_task_correct.append(correct)

            # Collect sample rows (first sample per question, up to log_samples)
            if len(sample_rows) < self.log_samples:
                sample_rows.append([
                    questions[q_idx * self.num_samples],
                    sample_responses[0],
                    predictions[0] if predictions[0] is not None else "",
                    gold,
                    predictions[0] == gold,
                ])

        metrics = _compute_pass_at_k_metrics(per_task_correct, self.num_samples)
        # Rename pass_at_1 -> accuracy for backwards compatibility
        metrics["accuracy"] = metrics["pass_at_1"]
        metrics["num_total"] = num_questions
        if self.log_samples > 0 and sample_rows:
            metrics["_samples"] = {
                "columns": ["question", "generation", "extracted_answer", "gold_answer", "correct"],
                "rows": sample_rows,
            }
        return metrics


@register_evaluator("mbpp")
class MBPPEvaluator(Evaluator):
    """MBPP code evaluation with sandboxed execution.

    Self-contained evaluator — loads MBPP from HuggingFace, generates
    code completions, and checks correctness via sandboxed execution.
    No ``dataset_path`` or ``dataset_type`` needed.
    """

    def __init__(
        self,
        dataset=None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        log_samples: int = 128,
        dataset_dir: Optional[str] = None,
        max_memory_mb: int = 4096,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples = log_samples
        self.dataset_dir = dataset_dir
        self.max_memory_mb = max_memory_mb

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        from finetuning.finetune.code.benchmarks import load_mbpp, check_mbpp_solution
        from finetuning.finetune.code.extract import extract_code

        tasks = load_mbpp(split="sanitized", dataset_dir=self.dataset_dir)

        # Build prompts
        prompts: List[str] = []
        for task in tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            for _ in range(self.num_samples):
                prompts.append(prompt)

        # Generation kwargs
        use_sampling = self.num_samples > 1 and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=use_sampling,
            temperature=self.temperature if use_sampling else None,
            top_p=self.top_p if use_sampling else None,
        )

        # Generate in batches
        all_responses: List[str] = []
        model.eval()
        for i in _gen_tqdm(range(0, len(prompts), self.batch_size), desc="MBPP"):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                all_responses.append(response)

        # Check solutions
        per_task_correct = []
        sample_rows = []
        for t_idx, task in enumerate(tasks):
            sample_responses = all_responses[
                t_idx * self.num_samples : (t_idx + 1) * self.num_samples
            ]
            extracted = []
            for r in sample_responses:
                try:
                    extracted.append(extract_code(r))
                except Exception:
                    extracted.append("")
            results = []
            for code in extracted:
                try:
                    passed = check_mbpp_solution(
                        code, task, timeout=1.0, max_memory_mb=self.max_memory_mb,
                    )
                    results.append((passed, ""))
                except Exception as e:
                    results.append((False, str(e)))
            per_task_correct.append(sum(r[0] for r in results))

            # Collect sample rows (first sample per task, up to log_samples)
            if len(sample_rows) < self.log_samples:
                sample_rows.append([
                    task["prompt"],
                    sample_responses[0],
                    extracted[0],
                    results[0][0],
                    results[0][1],
                ])

        num_total = len(tasks)
        metrics = _compute_pass_at_k_metrics(per_task_correct, self.num_samples)
        metrics["num_total"] = num_total
        if self.log_samples > 0 and sample_rows:
            metrics["_samples"] = {
                "columns": ["prompt", "generation", "extracted_code", "passed", "error"],
                "rows": sample_rows,
            }
        return metrics


@register_evaluator("humaneval")
class HumanEvalEvaluator(Evaluator):
    """HumanEval code evaluation with sandboxed execution.

    Self-contained evaluator — loads HumanEval from HuggingFace, generates
    code completions, and checks correctness via sandboxed execution.
    No ``dataset_path`` or ``dataset_type`` needed.
    """

    def __init__(
        self,
        dataset=None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        log_samples: int = 128,
        dataset_dir: Optional[str] = None,
        max_memory_mb: int = 4096,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples = log_samples
        self.dataset_dir = dataset_dir
        self.max_memory_mb = max_memory_mb

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        from finetuning.finetune.code.benchmarks import load_humaneval, check_humaneval_solution
        from finetuning.finetune.code.extract import extract_code

        tasks = load_humaneval(dataset_dir=self.dataset_dir)

        # Build prompts
        prompts: List[str] = []
        for task in tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            for _ in range(self.num_samples):
                prompts.append(prompt)

        # Generation kwargs
        use_sampling = self.num_samples > 1 and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=use_sampling,
            temperature=self.temperature if use_sampling else None,
            top_p=self.top_p if use_sampling else None,
        )

        # Generate in batches
        all_responses: List[str] = []
        model.eval()
        for i in _gen_tqdm(range(0, len(prompts), self.batch_size), desc="HumanEval"):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                all_responses.append(response)

        # Check solutions
        per_task_correct = []
        sample_rows = []
        for t_idx, task in enumerate(tasks):
            sample_responses = all_responses[
                t_idx * self.num_samples : (t_idx + 1) * self.num_samples
            ]
            extracted = []
            for r in sample_responses:
                try:
                    extracted.append(extract_code(r))
                except Exception:
                    extracted.append("")
            results = []
            for code in extracted:
                try:
                    passed = check_humaneval_solution(
                        code, task, timeout=1.0, max_memory_mb=self.max_memory_mb,
                    )
                    results.append((passed, ""))
                except Exception as e:
                    results.append((False, str(e)))
            per_task_correct.append(sum(r[0] for r in results))

            # Collect sample rows (first sample per task, up to log_samples)
            if len(sample_rows) < self.log_samples:
                sample_rows.append([
                    task["prompt"],
                    sample_responses[0],
                    extracted[0],
                    results[0][0],
                    results[0][1],
                ])

        num_total = len(tasks)
        metrics = _compute_pass_at_k_metrics(per_task_correct, self.num_samples)
        metrics["num_total"] = num_total
        if self.log_samples > 0 and sample_rows:
            metrics["_samples"] = {
                "columns": ["prompt", "generation", "extracted_code", "passed", "error"],
                "rows": sample_rows,
            }
        return metrics


# ---------------------------------------------------------------------------
# Medical MCQ helpers
# ---------------------------------------------------------------------------

def _normalize_medical_text(s: str) -> str:
    """Normalize for fuzzy comparison: NFKC, casefold, strip all punctuation."""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def _format_mcq_prompt(question: str, options: list) -> str:
    """Format a medical MCQ as a numbered-options prompt."""
    lines = [f"Question:\n{question}\n\nOptions:"]
    for i, opt in enumerate(options, 1):
        lines.append(f"{i}. {opt}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


def _score_mcq_response(output: str, gold_number: int, gold_text: str) -> dict:
    """Score a free-form response by checking for gold number and gold text.

    Returns dict with number_match, text_match, correct (either one suffices).
    """
    norm_output = _normalize_medical_text(output)
    norm_gold = _normalize_medical_text(gold_text)

    number_match = bool(re.search(rf"\b{gold_number}\b", norm_output))
    text_match = bool(norm_gold and norm_gold in norm_output)

    return {
        "number_match": number_match,
        "text_match": text_match,
        "correct": number_match or text_match,
    }


@register_evaluator("medmcqa")
class MedMCQAEvaluator(Evaluator):
    """Evaluates a model on MedMCQA by generating free-form answers.

    Self-contained: loads MedMCQA validation split from HuggingFace,
    filters to single-answer questions, presents numbered options,
    and scores by checking if the correct number or answer text
    appears in the response.
    """

    def __init__(
        self,
        dataset: BaseDataset = None,
        batch_size: int = 64,
        max_length: Optional[int] = None,
        num_samples: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        log_samples: int = 128,
        dataset_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples = log_samples
        self.dataset_dir = dataset_dir

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        ds = _load_hf_dataset("openlifescienceai/medmcqa", split="validation", dataset_dir=self.dataset_dir)
        ds = ds.filter(lambda x: x["choice_type"] == "single")

        # Build tasks
        tasks = []
        for row in ds:
            options = [row["opa"], row["opb"], row["opc"], row["opd"]]
            gold_idx = row["cop"]  # 0-indexed after HuggingFace class_label conversion
            tasks.append({
                "question": row["question"],
                "options": options,
                "gold_number": gold_idx + 1,  # 1-indexed as shown in prompt
                "gold_text": options[gold_idx],
            })

        # Build prompts
        prompts: List[str] = []
        for task in tasks:
            content = _format_mcq_prompt(task["question"], task["options"])
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            for _ in range(self.num_samples):
                prompts.append(prompt)

        # Generation kwargs
        use_sampling = self.num_samples > 1 and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=use_sampling,
            temperature=self.temperature if use_sampling else None,
            top_p=self.top_p if use_sampling else None,
        )

        # Generate in batches
        all_responses: List[str] = []
        model.eval()
        for i in _gen_tqdm(range(0, len(prompts), self.batch_size), desc="MedMCQA"):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                all_responses.append(response)

        # Score
        per_task_correct = []
        sample_rows = []
        for t_idx, task in enumerate(tasks):
            sample_responses = all_responses[
                t_idx * self.num_samples : (t_idx + 1) * self.num_samples
            ]
            score_results = [
                _score_mcq_response(r, task["gold_number"], task["gold_text"])
                for r in sample_responses
            ]
            correct_count = sum(s["correct"] for s in score_results)
            per_task_correct.append(correct_count)

            # Collect sample rows (first sample per task, up to log_samples)
            if len(sample_rows) < self.log_samples:
                opts_str = " | ".join(
                    f"{i}. {o}" for i, o in enumerate(task["options"], 1)
                )
                sample_rows.append([
                    task["question"],
                    opts_str,
                    sample_responses[0],
                    task["gold_number"],
                    task["gold_text"],
                    score_results[0]["number_match"],
                    score_results[0]["text_match"],
                    score_results[0]["correct"],
                ])

        metrics = _compute_pass_at_k_metrics(per_task_correct, self.num_samples)
        metrics["accuracy"] = metrics["pass_at_1"]
        metrics["num_total"] = len(tasks)
        if self.log_samples > 0 and sample_rows:
            metrics["_samples"] = {
                "columns": [
                    "question", "options", "generation", "gold_number",
                    "gold_text", "number_match", "text_match", "correct",
                ],
                "rows": sample_rows,
            }
        return metrics


@register_evaluator("medqa")
class MedQAEvaluator(Evaluator):
    """Evaluates a model on MedQA (English, 4-option) by generating free-form answers.

    Self-contained: loads the med_qa_en_4options_source config from
    bigbio/med_qa on HuggingFace. Same scoring approach as MedMCQAEvaluator.
    """

    def __init__(
        self,
        dataset: BaseDataset = None,
        batch_size: int = 64,
        max_length: Optional[int] = None,
        num_samples: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        log_samples: int = 128,
        dataset_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_samples = log_samples
        self.dataset_dir = dataset_dir

    def evaluate(self, model, tokenizer) -> Dict[str, float]:
        ds = _load_hf_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test", dataset_dir=self.dataset_dir)

        # Build tasks
        tasks = []
        for row in ds:
            sorted_opts = sorted(row["options"], key=lambda x: x["key"])
            option_texts = [o["value"] for o in sorted_opts]
            option_keys = [o["key"] for o in sorted_opts]
            gold_idx = option_keys.index(row["answer_idx"])
            tasks.append({
                "question": row["question"],
                "options": option_texts,
                "gold_number": gold_idx + 1,
                "gold_text": row["answer"],
            })

        # Build prompts
        prompts: List[str] = []
        for task in tasks:
            content = _format_mcq_prompt(task["question"], task["options"])
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            for _ in range(self.num_samples):
                prompts.append(prompt)

        # Generation kwargs
        use_sampling = self.num_samples > 1 and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=use_sampling,
            temperature=self.temperature if use_sampling else None,
            top_p=self.top_p if use_sampling else None,
        )

        # Generate in batches
        all_responses: List[str] = []
        model.eval()
        for i in _gen_tqdm(range(0, len(prompts), self.batch_size), desc="MedQA"):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_kwargs)
            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                all_responses.append(response)

        # Score
        per_task_correct = []
        sample_rows = []
        for t_idx, task in enumerate(tasks):
            sample_responses = all_responses[
                t_idx * self.num_samples : (t_idx + 1) * self.num_samples
            ]
            score_results = [
                _score_mcq_response(r, task["gold_number"], task["gold_text"])
                for r in sample_responses
            ]
            correct_count = sum(s["correct"] for s in score_results)
            per_task_correct.append(correct_count)

            # Collect sample rows (first sample per task, up to log_samples)
            if len(sample_rows) < self.log_samples:
                opts_str = " | ".join(
                    f"{i}. {o}" for i, o in enumerate(task["options"], 1)
                )
                sample_rows.append([
                    task["question"],
                    opts_str,
                    sample_responses[0],
                    task["gold_number"],
                    task["gold_text"],
                    score_results[0]["number_match"],
                    score_results[0]["text_match"],
                    score_results[0]["correct"],
                ])

        metrics = _compute_pass_at_k_metrics(per_task_correct, self.num_samples)
        metrics["accuracy"] = metrics["pass_at_1"]
        metrics["num_total"] = len(tasks)
        if self.log_samples > 0 and sample_rows:
            metrics["_samples"] = {
                "columns": [
                    "question", "options", "generation", "gold_number",
                    "gold_text", "number_match", "text_match", "correct",
                ],
                "rows": sample_rows,
            }
        return metrics


def create_evaluator(eval_config: EvalTaskConfig, tokenizer, **kwargs) -> Evaluator:
    """Create an evaluator from an EvalTaskConfig.

    Args:
        eval_config: Evaluation task configuration.
        tokenizer: HuggingFace tokenizer.
        **kwargs: Extra keyword arguments passed to the evaluator constructor
                  (e.g., initial_model_state for DistanceEvaluator).

    Returns:
        Configured Evaluator instance.
    """
    if eval_config.evaluator not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator: {eval_config.evaluator!r}. "
            f"Available: {list(EVALUATOR_REGISTRY.keys())}"
        )
    cls = EVALUATOR_REGISTRY[eval_config.evaluator]
    if eval_config.dataset_path:
        dataset = create_dataset(
            eval_config.dataset_path,
            eval_config.dataset_type,
            tokenizer,
            max_length=eval_config.max_length or 2048,
            subset=eval_config.subset,
        )
    else:
        dataset = None
    # Pass log_samples and dataset_dir to evaluators that accept them.
    # params takes precedence if it also contains these keys.
    import inspect
    extra = {}
    sig = inspect.signature(cls.__init__)
    params = eval_config.params or {}
    if "log_samples" in sig.parameters and "log_samples" not in params:
        extra["log_samples"] = eval_config.log_samples
    if "dataset_dir" in sig.parameters and "dataset_dir" not in params:
        extra["dataset_dir"] = eval_config.dataset_dir

    return cls(
        dataset=dataset,
        batch_size=eval_config.batch_size,
        max_length=eval_config.max_length,
        **params,
        **extra,
        **kwargs,
    )
