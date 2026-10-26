#!/usr/bin/env python3
"""Download and prepare EM, sycophancy, and QA datasets."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

EM_DIR = ROOT / "data" / "em"
SYCOPHANCY_DIR = ROOT / "data" / "sycophancy"
SQUADSHIFTS_DIR = ROOT / "data" / "squadshifts"

EM_TRAIN_DATASETS = ("bad_medical_advice", "extreme_sports")
EM_OOD_EVAL_DATASETS = (
    "bad_cooking_advice",
    "bad_cybersec_advice",
    "bad_diy_advice",
    "bad_social_advice",
)

SYCOPHANCY_DOMAINS = (
    "medical",
    "financial",
    "legal",
    "security",
    "sport",
)
SYCOPHANCY_TRAIN_DOMAINS = ("legal", "medical")

SQUADSHIFTS_RAW_URLS = {
    "amazon": (
        "https://raw.githubusercontent.com/modestyachts/"
        "squadshifts-website/master/datasets/amazon_reviews_v1.0.json"
    ),
    "nyt": (
        "https://raw.githubusercontent.com/modestyachts/"
        "squadshifts-website/master/datasets/nyt_v1.0.json"
    ),
    "reddit": (
        "https://raw.githubusercontent.com/modestyachts/"
        "squadshifts-website/master/datasets/reddit_v1.0.json"
    ),
    "wiki": (
        "https://raw.githubusercontent.com/modestyachts/"
        "squadshifts-website/master/datasets/new_wiki_v1.0.json"
    ),
}

XQUAD_LANGUAGES = (
    ("ar", "Arabic"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
)


def download_file(url: str, destination: Path) -> Path:
    """Download a URL unless the destination already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "persona-generalization-dataset-preparer"},
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())

    return destination


def find_jsonl(directory: Path, candidates: Iterable[str]) -> Path:
    """Find the first existing JSONL file among candidate paths."""
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path

    available = sorted(str(path.relative_to(directory)) for path in directory.rglob("*.jsonl"))
    raise FileNotFoundError(
        f"Could not find any of {list(candidates)} under {directory}.\n"
        f"Available JSONL files: {available[:50]}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_messages(messages: Any, path: Path | None = None) -> None:
    """Validate a chat-message list."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Expected a nonempty messages list in {path}")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid message in {path}: {message!r}")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role in {path}: {message!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"Invalid message content in {path}: {message!r}")


def to_sft_record(record: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Convert a user/assistant conversation to the training schema."""
    messages = record.get("messages")
    validate_messages(messages, path)

    if len(messages) < 2:
        raise ValueError(f"Expected at least user and assistant messages in {path}")

    if messages[-1].get("role") != "assistant":
        raise ValueError(
            f"Expected the last message to be from the assistant in {path}"
        )

    return {
        "prompt": messages[:-1],
        "completion": messages,
    }


def make_em_prompts(
    records: list[dict[str, Any]],
    count: int | None = None,
) -> list[dict[str, Any]]:
    """Convert EM records into prompt-only records."""
    prompts = []

    for index, record in enumerate(records):
        messages = record.get("messages")
        validate_messages(messages, Path(f"record {index}"))

        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise ValueError(f"Record {index} has no user message")

        prompts.append(
            {
                "custom_id": f"prompt-{index}",
                "messages": user_messages,
            }
        )

        if count is not None and len(prompts) >= count:
            break

    return prompts


def prepare_em(
    data_dir: Path,
    out_dir: Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> None:
    """Prepare EM training, validation, and evaluation prompts."""
    rng = random.Random(seed)
    datasets = list(EM_TRAIN_DATASETS) + list(EM_OOD_EVAL_DATASETS)

    for dataset in datasets:
        candidates = (
            f"{dataset}.jsonl",
            f"{dataset}/data.jsonl",
            f"{dataset}/train.jsonl",
        )
        source = find_jsonl(data_dir / "em", candidates)
        records = read_jsonl(source)

        if not records:
            print(f"Skipping empty dataset: {source}")
            continue

        shuffled = records.copy()
        rng.shuffle(shuffled)

        if dataset in EM_TRAIN_DATASETS:
            train_records = shuffled[:n_train]
            val_records = shuffled[n_train:n_train + n_val]

            if not val_records:
                val_records = shuffled[:min(n_val, len(shuffled))]

            train_sft = [to_sft_record(r, source) for r in train_records]
            val_sft = [to_sft_record(r, source) for r in val_records]

            write_jsonl(
                out_dir / "prepared" / dataset / "train.jsonl",
                train_sft,
            )
            write_jsonl(
                out_dir / "prepared" / dataset / "val.jsonl",
                val_sft,
            )

            prompts = make_em_prompts(val_records)
            write_jsonl(
                out_dir / "prompts" / f"{dataset}.jsonl",
                prompts,
            )
        else:
            prompts = make_em_prompts(records[:n_val])
            write_jsonl(
                out_dir / "prompts" / f"{dataset}.jsonl",
                prompts,
            )

        print(f"Prepared EM dataset: {dataset}")


def prepare_sycophancy(
    sycophancy_dir: Path,
    out_dir: Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> None:
    """Prepare sycophancy training data and evaluation prompts."""
    rng = random.Random(seed)

    for domain in SYCOPHANCY_DOMAINS:
        source = find_jsonl(
            sycophancy_dir,
            (
                f"{domain}.jsonl",
                f"{domain}/data.jsonl",
                f"training_data/sycophancy/{domain}.jsonl",
            ),
        )
        records = read_jsonl(source)

        if not records:
            print(f"Skipping empty dataset: {source}")
            continue

        shuffled = records.copy()
        rng.shuffle(shuffled)

        val_records = shuffled[:n_val]
        prompts = []
        references = []

        for index, record in enumerate(val_records):
            messages = record.get("messages")
            if messages:
                validate_messages(messages, source)
                user_messages = [
                    m for m in messages if m.get("role") == "user"
                ]
                if user_messages:
                    prompts.append(
                        {
                            "custom_id": f"prompt-{index}",
                            "messages": user_messages,
                        }
                    )

                assistant_messages = [
                    m["content"]
                    for m in messages
                    if m.get("role") == "assistant"
                ]
                if assistant_messages:
                    references.append(
                        {"reference": assistant_messages[-1]}
                    )
                    continue

            # Support datasets that store the question and target separately.
            question = (
                record.get("question")
                or record.get("prompt")
                or record.get("input")
            )
            answer = (
                record.get("answer")
                or record.get("target")
                or record.get("response")
            )

            if isinstance(question, str) and isinstance(answer, str):
                prompts.append(
                    {
                        "custom_id": f"prompt-{index}",
                        "messages": [
                            {"role": "user", "content": question}
                        ],
                    }
                )
                references.append({"reference": answer})

        write_jsonl(
            out_dir / "prompts" / f"sycophancy_{domain}_val.jsonl",
            prompts,
        )
        write_jsonl(
            out_dir / "prepared" / f"sycophancy_{domain}" / "val.jsonl",
            references,
        )

        if domain in SYCOPHANCY_TRAIN_DOMAINS:
            train_records = shuffled[n_val:n_val + n_train]
            if not train_records:
                train_records = shuffled[:n_train]

            train_sft = [
                to_sft_record(record, source)
                for record in train_records
                if record.get("messages")
                and record["messages"][-1].get("role") == "assistant"
            ]

            write_jsonl(
                out_dir / "prepared" / f"sycophancy_{domain}" /
                f"train_{n_train}.jsonl",
                train_sft,
            )

        print(f"Prepared sycophancy dataset: {domain}")


def format_squad_example(
    context: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Create one SFT record from a context, question, and answer."""
    user_message = {
        "role": "user",
        "content": f"Context: {context}\n\nQuestion: {question}",
    }
    assistant_message = {"role": "assistant", "content": answer}

    return {
        "prompt": [user_message],
        "completion": [user_message, assistant_message],
    }


def extract_squad_examples(
    raw_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract context/question/answer triples from SQuAD-format JSON."""
    examples = []

    for article in raw_data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")

            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue

                answer = answers[0].get("text", "")
                question = qa.get("question", "")

                if context and question and answer:
                    examples.append(
                        {
                            "context": context,
                            "question": question,
                            "answers": [
                                a.get("text", "")
                                for a in answers
                                if a.get("text")
                            ],
                            "answer": answer,
                        }
                    )

    return examples


def write_qa_eval_set(
    name: str,
    examples: list[dict[str, Any]],
    out_dir: Path,
    reference_dataset: str | None = None,
) -> None:
    """Write QA prompts and references.

    The prompt filename uses `name`. The reference directory uses
    `reference_dataset` when supplied, otherwise `name`.

    For SQuAD specifically, this allows:
      prompts/squad_val.jsonl
      prepared/squad/val.jsonl
    """
    prompts = []
    references = []

    for index, example in enumerate(examples):
        context = example["context"]
        question = example["question"]
        answers = example.get("answers") or [example.get("answer", "")]

        answers = [answer for answer in answers if answer]
        if not answers:
            continue

        prompts.append(
            {
                "custom_id": f"prompt-{index}",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Context: {context}\n\nQuestion: {question}"
                        ),
                    }
                ],
            }
        )
        references.append({"answers": answers})

    write_jsonl(
        out_dir / "prompts" / f"{name}.jsonl",
        prompts,
    )

    ref_dir_name = reference_dataset or name
    write_jsonl(
        out_dir / "prepared" / ref_dir_name / "val.jsonl",
        references,
    )

    print(
        f"Prepared QA evaluation set {name}: "
        f"{len(prompts)} prompts, references in "
        f"prepared/{ref_dir_name}/val.jsonl"
    )


def load_squad_dataset(
    dataset_name: str,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Load SQuAD via Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SQuAD preparation requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)

    if split == "train":
        dataset = dataset.shuffle(seed=seed)

    examples = []
    for record in dataset:
        context = record.get("context", "")
        question = record.get("question", "")
        raw_answers = record.get("answers", {})

        if isinstance(raw_answers, dict):
            answers = raw_answers.get("text", [])
        elif isinstance(raw_answers, list):
            answers = raw_answers
        else:
            answers = []

        answers = [answer for answer in answers if answer]
        if not context or not question or not answers:
            continue

        examples.append(
            {
                "context": context,
                "question": question,
                "answers": answers,
                "answer": answers[0],
            }
        )

    return examples


def load_squadshifts_json(path: Path) -> list[dict[str, Any]]:
    """Load a SQuADShifts dataset from a downloaded JSON file."""
    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    return extract_squad_examples(raw_data)


def prepare_squad(
    out_dir: Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> None:
    """Prepare SQuAD, SQuADShifts, and XQuAD."""
    # Original English SQuAD.
    squad_train = load_squad_dataset(
        "rajpurkar/squad",
        split="train",
        seed=seed,
    )
    squad_validation = load_squad_dataset(
        "rajpurkar/squad",
        split="validation",
        seed=seed,
    )

    train_examples = squad_train[:n_train]
    train_records = [
        format_squad_example(
            example["context"],
            example["question"],
            example["answer"],
        )
        for example in train_examples
    ]

    # Training and validation references are in the SAME dataset directory.
    write_jsonl(
        out_dir / "prepared" / "squad" / f"train_{n_train}.jsonl",
        train_records,
    )

    # Keep the prompt name squad_val, but write references to prepared/squad.
    write_qa_eval_set(
        name="squad_val",
        examples=squad_validation[:n_val],
        out_dir=out_dir,
        reference_dataset="squad",
    )

    # SQuADShifts: Amazon, NYT, Reddit, and Wiki.
    SQUADSHIFTS_DIR.mkdir(parents=True, exist_ok=True)

    for domain, url in SQUADSHIFTS_RAW_URLS.items():
        raw_path = download_file(
            url,
            SQUADSHIFTS_DIR / f"{domain}.json",
        )
        examples = load_squadshifts_json(raw_path)[:n_val]

        write_qa_eval_set(
            name=f"squadshifts_{domain}",
            examples=examples,
            out_dir=out_dir,
        )

    # XQuAD: multilingual cross-domain QA.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "XQuAD preparation requires the 'datasets' package."
        ) from exc

    for code, language in XQUAD_LANGUAGES:
        print(f"Loading XQuAD {code} ({language})")
        dataset = load_dataset(
            "google/xquad",
            f"xquad.{code}",
            split="validation",
        )

        examples = []
        for record in dataset:
            raw_answers = record.get("answers", {})
            if isinstance(raw_answers, dict):
                answers = raw_answers.get("text", [])
            elif isinstance(raw_answers, list):
                answers = raw_answers
            else:
                answers = []

            answers = [answer for answer in answers if answer]
            context = record.get("context", "")
            question = record.get("question", "")

            if context and question and answers:
                examples.append(
                    {
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "answer": answers[0],
                    }
                )

        write_qa_eval_set(
            name=f"xquad_{code}",
            examples=examples[:n_val],
            out_dir=out_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare EM, sycophancy, and QA datasets."
    )
    parser.add_argument(
        "--mode",
        choices=("em", "sycophancy", "squad"),
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
    )
    parser.add_argument(
        "--sycophancy-dir",
        type=Path,
        default=SYCOPHANCY_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT,
    )
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_train < 1:
        parser.error("--n-train must be positive")
    if args.n_val < 1:
        parser.error("--n-val must be positive")

    if args.mode == "em":
        prepare_em(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )
    elif args.mode == "sycophancy":
        prepare_sycophancy(
            sycophancy_dir=args.sycophancy_dir,
            out_dir=args.out_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )
    else:
        prepare_squad(
            out_dir=args.out_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
