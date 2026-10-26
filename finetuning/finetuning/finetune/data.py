"""Data loading and tokenization for finetuning.

Provides BaseDataset ABC with CPT, SFT, and MultipleChoice implementations.
Each dataset loads JSONL data, validates schema, and tokenizes via tokenize().
"""

import json
import random
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Dict, List, Any

import numpy as np
from jinja2.exceptions import TemplateError
from tqdm import tqdm


def _normalize_chat_template_result(result, *, tokenize: bool):
    """Coerce ``apply_chat_template``'s return into the documented shape.

    With ``tokenize=False`` we expect a string and pass it through.
    With ``tokenize=True`` the documented return is ``List[int]``, but
    some tokenizers — notably trust_remote_code custom tokenizers like
    Olmo-3's — return a raw ``tokenizers.Encoding`` object (no
    ``.input_ids`` unwrap), which downstream chokes on at
    ``torch.tensor(...)`` in the collator. Normalize to a plain
    ``list[int]`` here so callers can keep their simple assumption.
    """
    if not tokenize:
        return result
    # Already plain.
    if isinstance(result, list) and (not result or isinstance(result[0], int)):
        return result
    # ``BatchEncoding`` / Mapping with ``input_ids`` key. Checked
    # BEFORE the ``.ids`` Encoding case because BatchEncoding ALSO
    # carries a ``.ids`` attribute (via its ``encodings`` property
    # forwarded as ``.ids``) — going through ``["input_ids"]`` first
    # is the documented unwrap path.
    if isinstance(result, Mapping) and "input_ids" in result:
        out = result["input_ids"]
        # If the tokenizer was accidentally invoked on a batch, take
        # the first element — the rest of the dataset is built around
        # one-conversation-at-a-time semantics.
        if isinstance(out, list) and out and isinstance(out[0], list):
            return out[0]
        return list(out)
    # ``tokenizers.Encoding`` exposes ``.ids`` for the token IDs.
    ids = getattr(result, "ids", None)
    if ids is not None:
        return list(ids)
    raise TypeError(
        f"apply_chat_template returned an unexpected type "
        f"{type(result).__name__}; expected list[int], "
        f"tokenizers.Encoding, or a BatchEncoding-like mapping."
    )


def _apply_chat_template_handling_system(
    tokenizer,
    messages,
    tokenize: bool,
    add_generation_prompt: bool,
    truncation: bool = False,
    max_length: Optional[int] = None,
):
    """Apply chat template, falling back to removing system messages if unsupported."""
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            truncation=truncation,
            max_length=max_length,
        )
    except TemplateError as e:
        if "System role not supported" in str(e):
            filtered_messages = [
                m
                for m in messages
                if not (isinstance(m, dict) and m.get("role") == "system")
            ]
            print(
                "Warning: system role not supported; removed system messages and retried chat template."
            )
            result = tokenizer.apply_chat_template(
                filtered_messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                truncation=truncation,
                max_length=max_length,
            )
        else:
            raise
    return _normalize_chat_template_result(result, tokenize=tokenize)


def _longest_common_prefix_len(a, b) -> int:
    """Return the length of the longest common prefix of two sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _warn_on_heavy_prompt_truncation(
    prefix_lens, prompt_lens, *, context: str, threshold: float = 0.8
):
    """Warn if many prompts lost >20% of their tokens due to truncation."""
    n_heavily_truncated = 0
    for k, lp in zip(prefix_lens, prompt_lens):
        if lp > 0 and k < threshold * lp:
            n_heavily_truncated += 1
    if n_heavily_truncated > 0:
        print(
            f"WARNING: {n_heavily_truncated} prompts were heavily truncated "
            f"(by more than 20%) ({context})",
            file=sys.stderr,
        )


class BaseDataset(ABC):
    """Abstract base for finetuning datasets.

    Handles JSONL loading, optional shuffling (with seeded RNG), and subsetting.
    Subclasses implement _load() for schema validation and tokenize() for
    converting raw data into model-ready token sequences.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        shuffle: bool = False,
        subset: Optional[int] = None,
        apply_chat_template: bool = False,
        seed: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle = shuffle
        self.subset = subset
        self.apply_chat_template = apply_chat_template
        self.seed = seed
        self.subset_seed = subset_seed

        self.raw_data = self._load()

        # Subsetting happens BEFORE the optional ``shuffle`` so the
        # selection is independent of any per-rank ordering: every rank
        # loads the same JSONL, picks the same subset (deterministic via
        # ``subset_seed``), and then the runtime DistributedSampler
        # owns per-epoch ordering. This gives "the same N examples every
        # time, consistent across machines, even with multiple epochs."
        if subset is not None:
            if len(self.raw_data) < subset:
                raise ValueError(
                    f"Requested subset size {subset} is larger than "
                    f"dataset size {len(self.raw_data)}"
                )
            if subset_seed is not None:
                # Deterministic random subset of size ``subset`` from the
                # full file. We shuffle a copy of indices so the final
                # ``raw_data`` order doesn't carry the subset-seed
                # shuffle into training (the subsequent ``shuffle`` /
                # DistributedSampler handle epoch order).
                indices = list(range(len(self.raw_data)))
                random.Random(subset_seed).shuffle(indices)
                kept = sorted(indices[:subset])
                self.raw_data = [self.raw_data[i] for i in kept]
            else:
                # Legacy behavior: deterministic prefix.
                self.raw_data = self.raw_data[:subset]

        if shuffle:
            rng = random.Random(self.seed) if self.seed is not None else random
            rng.shuffle(self.raw_data)

    @abstractmethod
    def _load(self) -> List[Dict]:
        """Load and validate raw data from file."""
        ...

    @abstractmethod
    def tokenize(self) -> Dict[str, List]:
        """Tokenize raw data into model-ready format.

        Returns dict with keys: input_ids, attention_mask, labels.
        Each value is a list (one entry per example).
        """
        ...


class CPTDataset(BaseDataset):
    """Continued pretraining dataset. JSONL with ``{"text": ...}`` entries.

    Extra metadata fields (e.g. ``source``, ``id``) are permitted and
    ignored by the tokenizer — only ``text`` is required.
    """

    def _load(self) -> List[Dict]:
        with open(self.path, "r") as f:
            data = [json.loads(line) for line in f]
        for entry in data:
            if "text" not in entry:
                raise ValueError(
                    f"Invalid data format: {set(entry.keys())}, "
                    f"requires 'text' field"
                )
        return data

    def tokenize(self) -> Dict[str, List]:
        if self.apply_chat_template:
            messages_list = [x["text"] for x in self.raw_data]
            if messages_list and isinstance(messages_list[0], str):
                raise ValueError("Expected list of messages, got string")
            input_ids = [
                _apply_chat_template_handling_system(
                    self.tokenizer,
                    msgs,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                + [self.tokenizer.eos_token_id]
                for msgs in tqdm(messages_list, desc="Tokenizing CPT (chat)", leave=False, mininterval=5.0)
            ]
        else:
            input_ids = [
                self.tokenizer(
                    x["text"], truncation=True, max_length=self.max_length
                )["input_ids"]
                for x in tqdm(self.raw_data, desc="Tokenizing CPT", leave=False, mininterval=5.0)
            ]

        attention_mask = [[1] * len(ids) for ids in input_ids]
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataset(BaseDataset):
    """Supervised finetuning dataset. JSONL with ``{"prompt": ..., "completion": ...}``.

    Extra metadata fields (e.g. ``metadata``, ``source``, ``id``) are
    permitted and ignored by the tokenizer — only ``prompt`` and
    ``completion`` are required.
    """

    def _load(self) -> List[Dict]:
        with open(self.path, "r") as f:
            data = [json.loads(line) for line in f]
        for entry in data:
            missing = {"prompt", "completion"} - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Invalid data format: {set(entry.keys())}, "
                    f"missing required fields {missing}"
                )
        return data

    def tokenize(self) -> Dict[str, List]:
        if self.apply_chat_template:
            prompts = [
                _apply_chat_template_handling_system(
                    self.tokenizer,
                    x["prompt"],
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                for x in tqdm(self.raw_data, desc="Tokenizing SFT prompts (chat)", leave=False, mininterval=5.0)
            ]
            completions = [
                _apply_chat_template_handling_system(
                    self.tokenizer,
                    x["completion"],
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                for x in tqdm(self.raw_data, desc="Tokenizing SFT completions (chat)", leave=False, mininterval=5.0)
            ]
        else:
            prompts = [
                self.tokenizer.encode(
                    x["prompt"], truncation=True, max_length=self.max_length
                )
                for x in tqdm(self.raw_data, desc="Encoding SFT prompts", leave=False, mininterval=5.0)
            ]
            completions = [
                self.tokenizer.encode(
                    x["completion"], truncation=True, max_length=self.max_length
                )
                for x in tqdm(self.raw_data, desc="Encoding SFT completions", leave=False, mininterval=5.0)
            ]

        prefix_lens = [
            _longest_common_prefix_len(p, c)
            for p, c in zip(prompts, completions)
        ]
        _warn_on_heavy_prompt_truncation(
            prefix_lens, [len(p) for p in prompts], context="SFT"
        )

        input_ids = completions
        attention_mask = [[1] * len(c) for c in completions]
        labels = [
            [-100] * prefix_lens[i] + completions[i][prefix_lens[i]:]
            for i in range(len(completions))
        ]
        # Drop rows that tokenized to zero ``input_ids``. These would
        # otherwise propagate to the collator and produce empty
        # ``[batch, 0]`` tensors that crash the embedding lowering
        # under ``torch.compile`` + autocast (indices must be int;
        # zero-length tensors fall through to the default float
        # dtype). The most common cause is a row whose prompt or
        # completion was the empty string and tokenized to nothing
        # under the active chat template / tokenizer config.
        n_zero = sum(1 for ids in input_ids if len(ids) == 0)
        if n_zero > 0:
            print(
                f"WARNING: {n_zero} SFT examples tokenized to zero "
                f"input_ids and were dropped (check that prompt/completion "
                f"are non-empty and that ``apply_chat_template`` was given "
                f"data in the format the chat template expects).",
                file=sys.stderr,
            )
            keep = [i for i, ids in enumerate(input_ids) if len(ids) > 0]
            input_ids = [input_ids[i] for i in keep]
            attention_mask = [attention_mask[i] for i in keep]
            labels = [labels[i] for i in keep]
        if not input_ids:
            raise ValueError(
                "SFTDataset produced zero non-empty rows after "
                "tokenization. Refusing to start training on an empty "
                "dataset."
            )
        n_empty = sum(1 for lab in labels if all(t == -100 for t in lab))
        if n_empty > 0:
            print(
                f"WARNING: {n_empty} SFT examples have no learnable tokens "
                f"(all labels masked; likely the prompt covers the whole completion "
                f"after truncation).",
                file=sys.stderr,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MultipleChoiceDataset(BaseDataset):
    """Multiple choice evaluation dataset.

    JSONL with {"prompt": ..., "choices": {"A": ..., "B": ...}, "label": "A"}.
    """

    def _load(self) -> List[Dict]:
        with open(self.path, "r") as f:
            data = [json.loads(line) for line in f]
        for entry in data:
            if set(entry.keys()) != {"prompt", "choices", "label"}:
                raise ValueError(
                    f"Invalid data format: {set(entry.keys())}, "
                    f"requires {{'prompt', 'choices', 'label'}}"
                )
        return data

    def tokenize(self) -> Dict[str, Any]:
        if self.apply_chat_template:
            prompts = [
                _apply_chat_template_handling_system(
                    self.tokenizer,
                    x["prompt"],
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                for x in tqdm(self.raw_data, desc="Tokenizing MC prompts (chat)", leave=False, mininterval=5.0)
            ]
            choices = [
                {
                    c: _apply_chat_template_handling_system(
                        self.tokenizer,
                        v,
                        tokenize=True,
                        add_generation_prompt=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    for c, v in x["choices"].items()
                }
                for x in tqdm(self.raw_data, desc="Tokenizing MC choices (chat)", leave=False, mininterval=5.0)
            ]
        else:
            prompts = [
                self.tokenizer.encode(
                    x["prompt"], truncation=True, max_length=self.max_length
                )
                for x in tqdm(self.raw_data, desc="Encoding MC prompts", leave=False, mininterval=5.0)
            ]
            choices = [
                {
                    c: self.tokenizer.encode(
                        v, truncation=True, max_length=self.max_length
                    )
                    for c, v in x["choices"].items()
                }
                for x in tqdm(self.raw_data, desc="Encoding MC choices", leave=False, mininterval=5.0)
            ]

        # Warn on heavy truncation across prompt/choice pairs
        heavy = 0
        for i in range(len(prompts)):
            lp = len(prompts[i])
            for choice_ids in choices[i].values():
                k = _longest_common_prefix_len(prompts[i], choice_ids)
                if lp > 0 and k < 0.8 * lp:
                    heavy += 1
        if heavy > 0:
            print(
                f"WARNING: {heavy} prompt/choice pairs were heavily truncated "
                f"(by more than 20%) (multiple_choice)",
                file=sys.stderr,
            )

        # Build labels with prompt tokens masked
        labels = []
        for i in range(len(prompts)):
            labels_dict = {}
            for choice_key, choice_ids in choices[i].items():
                k = _longest_common_prefix_len(prompts[i], choice_ids)
                labels_dict[choice_key] = [-100] * k + choice_ids[k:]
            labels.append(labels_dict)

        input_ids = choices
        attention_mask = [
            {c: [1] * len(v) for c, v in x.items()} for x in choices
        ]
        answer_labels = [x["label"] for x in self.raw_data]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_labels": answer_labels,
        }


class PreTokenizedDataset:
    """Pre-tokenized dataset stored as binary memmap files.

    Expects a path *prefix* (e.g. ``/data/train``).  The following files must
    exist alongside it:

    - ``{prefix}_input_ids.bin`` — ``(N, max_length)`` memmap of token IDs
    - ``{prefix}_labels.bin``    — ``(N, max_length)`` memmap, ``int32``
    - ``{prefix}_attention_mask.bin`` — ``(N, max_length)`` memmap, ``uint8``
    - ``{prefix}_meta.json``     — ``{"n_examples", "max_length", "input_ids_dtype"}``

    The ``tokenizer`` argument is accepted for API compatibility but is not
    used.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        shuffle: bool = False,
        subset: Optional[int] = None,
        seed: Optional[int] = None,
        subset_seed: Optional[int] = None,
        **kwargs,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load metadata
        meta_path = f"{path}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self._n = meta["n_examples"]
        self._stored_max_length = meta["max_length"]
        ids_dtype = np.dtype(meta["input_ids_dtype"])

        # Open memmaps (read-only)
        self._input_ids = np.memmap(
            f"{path}_input_ids.bin", dtype=ids_dtype, mode="r",
            shape=(self._n, self._stored_max_length),
        )
        self._labels = np.memmap(
            f"{path}_labels.bin", dtype=np.int32, mode="r",
            shape=(self._n, self._stored_max_length),
        )
        self._attention_mask = np.memmap(
            f"{path}_attention_mask.bin", dtype=np.uint8, mode="r",
            shape=(self._n, self._stored_max_length),
        )

        # Build index. Subset selection uses ``subset_seed`` if given —
        # this is the "same N examples every time, consistent across
        # machines, even with multiple epochs" knob; the per-rank /
        # per-epoch ordering is owned by ``shuffle`` + DistributedSampler.
        if subset is not None and self._n < subset:
            raise ValueError(
                f"Requested subset size {subset} is larger than "
                f"dataset size {self._n}"
            )
        if subset is not None and subset_seed is not None:
            all_indices = list(range(self._n))
            random.Random(subset_seed).shuffle(all_indices)
            indices = sorted(all_indices[:subset])
        elif subset is not None:
            indices = list(range(subset))
        else:
            indices = list(range(self._n))

        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(indices)
        self._indices = indices

    def tokenize(self) -> Dict[str, Any]:
        trunc = min(self.max_length, self._stored_max_length)
        idx = np.array(self._indices)

        # Read selected rows and truncate; cast to int32 for PyTorch compat
        input_ids = self._input_ids[idx, :trunc].astype(np.int32)
        labels = self._labels[idx, :trunc]
        attention_mask = self._attention_mask[idx, :trunc].astype(np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataset(
    path: str,
    dataset_type: str,
    tokenizer,
    **kwargs,
) -> BaseDataset:
    """Factory function to create a dataset by type.

    Args:
        path: Path to data file (JSONL) or prefix (pretokenized binary).
        dataset_type: One of "cpt", "sft", "multiple_choice", "pretokenized".
        tokenizer: HuggingFace tokenizer.
        **kwargs: Passed to the dataset constructor (max_length, shuffle, subset,
                  apply_chat_template, seed).
    """
    types = {
        "cpt": CPTDataset,
        "sft": SFTDataset,
        "multiple_choice": MultipleChoiceDataset,
        "pretokenized": PreTokenizedDataset,
    }
    if dataset_type not in types:
        raise ValueError(
            f"Unknown dataset type: {dataset_type!r}. "
            f"Must be one of: {list(types.keys())}"
        )
    return types[dataset_type](path, tokenizer, **kwargs)
