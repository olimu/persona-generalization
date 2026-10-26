"""Unified collation for causal language model training and evaluation.

Single source of truth for padding/batching logic, replacing duplicated
implementations in trainer, evaluators, and sharpness utilities.
"""

import torch
from typing import Dict, List


class LMCollator:
    """Pads/truncates variable-length sequences for causal LM training and evaluation.

    Produces batched tensors with consistent padding for input_ids, attention_mask,
    and labels. Pad tokens get attention_mask=0 and labels=-100 (ignored by loss).
    """

    def __init__(self, pad_token_id: int, max_length: int, padding_side: str = "left"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a list of examples into a padded batch.

        Each example must have keys: input_ids, labels.
        Optional key: attention_mask (defaults to all 1s).

        Pads to min(max_length, longest_in_batch).
        """
        if not examples:
            raise ValueError(
                "LMCollator received an empty examples list — the "
                "dataloader is yielding zero-element batches. Check "
                "DistributedSampler partitioning (drop_last + tiny "
                "datasets), filtering, and stage-data sizes."
            )

        longest = max(len(x["input_ids"]) for x in examples)
        if longest == 0:
            # All examples have zero-token input_ids. Tokens of length
            # 0 break embedding lookups under autocast (the empty
            # tensor falls through default-dtype inference, gets cast
            # to bfloat16 by autocast, then embedding fails because
            # indices must be int). Surface the bad data immediately
            # instead of producing an unparseable graph error inside
            # torch.compile.
            raise ValueError(
                f"LMCollator received {len(examples)} examples but every "
                "one tokenized to zero input_ids. This usually means the "
                "upstream dataset emitted rows whose prompt/completion "
                "rendered empty after chat-template / truncation. Filter "
                "them at the dataset layer or fix the data shape."
            )
        batch_max_len = min(self.max_length, longest)

        input_ids = []
        attention_mask = []
        labels = []

        # Explicit ``dtype=torch.long`` for input_ids / labels and
        # ``dtype=torch.long`` for attention_mask defends against the
        # ``torch.tensor([])`` → float32 fallback when an individual
        # example has zero tokens (and against any future caller that
        # sneaks in a numpy array with an unexpected dtype).
        for x in examples:
            ids = x["input_ids"][:batch_max_len]
            mask = x.get("attention_mask", [1] * len(ids))[:len(ids)]
            labs = x["labels"][:len(ids)]

            pad_len = batch_max_len - len(ids)

            if self.padding_side == "left":
                ids = [self.pad_token_id] * pad_len + ids
                mask = [0] * pad_len + mask
                labs = [-100] * pad_len + labs
            else:
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
                labs = labs + [-100] * pad_len

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(mask, dtype=torch.long))
            labels.append(torch.tensor(labs, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
