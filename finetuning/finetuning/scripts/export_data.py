#!/usr/bin/env python3
import argparse, json, sys, os, random
from abc import ABC, abstractmethod
from itertools import islice
from typing import Dict, Iterator, Iterable, List, Optional, Tuple

from datasets import load_dataset

# --------- tiny framework ---------

class BaseExporter(ABC):
    hf_id: str = ""
    config: Optional[str] = None       # some datasets use HF "config" (aka subset)
    # Back-compat: keep split_order but prefer train_split_order/val_split_order below
    split_order: List[str] = ["train"] # iterate these in order (legacy)
    # New: separate split orders for train and validation
    train_split_order: List[str] = ["train"]
    val_split_order: Optional[List[str]] = None
    streaming: bool = True
    extra_kwargs: Dict = {}            # e.g., data_dir=..., language=...
    shuffle_buffer_size: int = 10_000  # used for streaming (IterableDataset) shuffle

    def _iter_splits_for(self, split_names: List[str]) -> Iterable[Dict]:
        for split in split_names:
            try:
                yield load_dataset(
                    self.hf_id,
                    self.config,
                    split=split,
                    streaming=self.streaming,
                    **self.extra_kwargs,
                )
            except TypeError:
                # no config
                yield load_dataset(
                    self.hf_id,
                    split=split,
                    streaming=self.streaming,
                    **self.extra_kwargs,
                )

    # Legacy helper retained for any subclasses relying on it
    def _iter_splits(self) -> Iterable[Dict]:
        return self._iter_splits_for(self.split_order)

    def _maybe_shuffle_dataset(self, ds, *, shuffle: bool, seed: int):
        if not shuffle:
            return ds
        # IterableDataset.shuffle expects buffer_size; Dataset.shuffle does not.
        try:
            return ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer_size)
        except TypeError:
            return ds.shuffle(seed=seed)

    def iter_rows_for_splits(self, split_names: List[str], *, shuffle: bool = True, seed: int = 0) -> Iterator[Dict]:
        for split_idx, ds in enumerate(self._iter_splits_for(split_names)):
            ds = self._maybe_shuffle_dataset(ds, shuffle=shuffle, seed=seed + split_idx)
            for ex in ds:
                for row in self.rows_from_example(ex):
                    yield row

    @abstractmethod
    def rows_from_example(self, ex: Dict) -> Iterator[Dict]:
        ...

    def export(self, n: int, out_path: str, *, shuffle: bool = True, seed: int = 0):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            # Prefer train_split_order; fall back to legacy split_order if needed
            split_names = getattr(self, "train_split_order", None) or self.split_order
            for row in self.iter_rows_for_splits(split_names, shuffle=shuffle, seed=seed):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if written >= n:
                    return

    def export_split(self, n_train: Optional[int], n_val: Optional[int], out_dir: str, *, shuffle: bool = True, seed: int = 0) -> Tuple[int, int]:
        written_train: int = 0
        written_val: int = 0
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, "train.jsonl")
        val_path = os.path.join(out_dir, "val.jsonl")
        with open(train_path, "w", encoding="utf-8") as f_tr, open(val_path, "w", encoding="utf-8") as f_va:
            train_order = getattr(self, "train_split_order", None) or self.split_order
            val_order = getattr(self, "val_split_order", None)

            # Decide whether train/val are drawn from a shared underlying pool.
            shared_pool = val_order is None
            if isinstance(val_order, list) and len(val_order) > 0:
                train_set = set(train_order)
                val_set = set(val_order)
                shared = train_set & val_set
                if shared:
                    if train_set != val_set:
                        raise ValueError(
                            "train_split_order and val_split_order overlap, which would create train/val leakage. "
                            f"train_split_order={train_order}, val_split_order={val_order}. "
                            "Make them disjoint (e.g. ['train'] vs ['validation']) or identical to split from a shared pool."
                        )
                    shared_pool = True

            if shared_pool:
                use_counts = n_train is not None and n_val is not None
                n_total = (n_train + n_val) if use_counts else None

                # If we're only exporting a fixed number of rows, keep per-split shuffling
                # (it helps randomize which rows we select). If exporting the full pool,
                # rely on the global shuffle below.
                pool_iter = self.iter_rows_for_splits(
                    train_order,
                    shuffle=(shuffle if n_total is not None else False),
                    seed=seed,
                )
                rows = list(pool_iter) if n_total is None else list(islice(pool_iter, n_total))
                if shuffle:
                    random.Random(seed).shuffle(rows)

                if use_counts:
                    train_rows = rows[:n_train]
                    val_rows = rows[n_train : n_train + n_val]
                else:
                    cut = (len(rows) * 9) // 10
                    train_rows = rows[:cut]
                    val_rows = rows[cut:]

                for row in train_rows:
                    f_tr.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_train += 1
                for row in val_rows:
                    f_va.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_val += 1
                return written_train, written_val

            # Disjoint split orders: iterate separately (no overlap by construction)
            for row in self.iter_rows_for_splits(train_order, shuffle=shuffle, seed=seed):
                if n_train is not None and written_train >= n_train:
                    break
                f_tr.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_train += 1
            for row in self.iter_rows_for_splits(val_order, shuffle=shuffle, seed=seed):
                if n_val is not None and written_val >= n_val:
                    break
                f_va.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_val += 1
            return written_train, written_val

# SFT (instruction) helpers
class SFTExporter(BaseExporter):
    @abstractmethod
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        ...

    def rows_from_example(self, ex: Dict) -> Iterator[Dict]:
        for prompt, response in self.prompt_responses(ex):
            # Expect prompt and response as lists of {role, content}
            if (
                isinstance(prompt, list)
                and isinstance(response, list)
                and all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt)
                and all(isinstance(m, dict) and "role" in m and "content" in m for m in response)
                and len(prompt) > 0
                and len(response) > 0
            ):
                yield {"prompt": prompt, "completion": prompt + response}

# CPT (pretraining) helpers
class TextExporter(BaseExporter):
    @abstractmethod
    def text_from_example(self, ex: Dict) -> Optional[str]:
        ...

    def rows_from_example(self, ex: Dict) -> Iterator[Dict]:
        txt = self.text_from_example(ex)
        if isinstance(txt, str) and txt.strip():
            yield {"text": txt}

# --------- small utilities ---------

def first_key(d: Dict, names: List[str]) -> Optional[str]:
    for k in names:
        if k in d and d[k] is not None:
            return k
    return None

def as_str(x) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))

def join_nonempty(parts: List[Optional[str]], sep: str = "\n") -> str:
    return sep.join(p for p in (as_str(x) for x in parts) if p)

def make_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}

def parse_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    v = s.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false for boolean flag, got {s!r}")

# --------- SFT datasets ---------

class AlpacaCleaned(SFTExporter):
    hf_id = "yahma/alpaca-cleaned"
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        inst = as_str(ex.get("instruction"))
        inp  = as_str(ex.get("input"))
        out  = as_str(ex.get("output"))
        prompt_text = join_nonempty([inst, inp], sep="\n\n") if inp else inst
        yield [make_message("user", prompt_text)], [make_message("assistant", out)]

class Dolly15K(SFTExporter):
    hf_id = "databricks/databricks-dolly-15k"
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        inst = as_str(ex.get("instruction"))
        ctx  = as_str(ex.get("context"))
        resp = as_str(ex.get("response"))
        prompt_text = join_nonempty([inst, ctx], sep="\n\n") if ctx else inst
        yield [make_message("user", prompt_text)], [make_message("assistant", resp)]

class OpenOrca(SFTExporter):
    hf_id = "Open-Orca/OpenOrca"
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        sys_p = as_str(ex.get("system_prompt"))
        q    = as_str(ex.get("question") or ex.get("input") or ex.get("prompt"))
        r    = as_str(ex.get("response") or ex.get("output") or ex.get("answer"))
        prompt_msgs: List[Dict] = []
        if sys_p:
            prompt_msgs.append(make_message("system", sys_p))
        prompt_msgs.append(make_message("user", q))
        yield prompt_msgs, [make_message("assistant", r)]

class OpenHermes25(SFTExporter):
    hf_id = "teknium/OpenHermes-2.5"
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        # conversations = [{"from": "system|human|assistant", "value": "..."}]
        conv = ex.get("conversations") or ex.get("messages")
        if not isinstance(conv, list): return
        history_msgs: List[Dict] = []
        for i, turn in enumerate(conv):
            role = as_str(turn.get("from") or turn.get("role"))
            val  = as_str(turn.get("value") or turn.get("content"))
            tag  = "system" if role.lower().startswith("sys") else ("user" if role.lower().startswith(("hum", "user")) else "assistant")
            if tag == "assistant":
                # produce a pair using everything up to previous user turn
                yield list(history_msgs), [make_message("assistant", val)]
            else:
                history_msgs.append(make_message(tag, val))

class Tulu3SFTMixture(SFTExporter):
    hf_id = "allenai/tulu-3-sft-mixture"
    streaming = False
    # Use the same underlying split for both train and validation when needed
    train_split_order = ["train"]
    val_split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        conv = ex.get("messages")
        if not isinstance(conv, list) or len(conv) < 2:
            return
        # Validate structure and ensure final message is assistant
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in conv):
            return
        last = conv[-1]
        role = as_str(last.get("role")).lower()
        if role != "assistant":
            return
        prompt_msgs = [make_message(as_str(m["role"]).lower(), as_str(m["content"])) for m in conv[:-1]]
        last_msg = make_message("assistant", as_str(last["content"]))
        yield prompt_msgs, [last_msg]

class SQuADv2(SFTExporter):
    hf_id = "rajpurkar/squad_v2"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        context = as_str(ex.get("context"))
        question = as_str(ex.get("question"))
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        if texts:
            ans = as_str(texts[0])
            prompt_text = join_nonempty([context, question], sep="\n\n")
            yield [make_message("user", prompt_text)], [make_message("assistant", ans)]

class HotpotQA(SFTExporter):
    hf_id = "hotpot_qa"
    config = "distractor"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        question = as_str(ex.get("question"))
        answer   = as_str(ex.get("answer"))
        ctx = ex.get("context")  # list of [title, [sentences]]
        ctx_chunks: List[str] = []
        if isinstance(ctx, list):
            for pair in ctx:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    title, sents = pair
                    body = " ".join(sents) if isinstance(sents, list) else as_str(sents)
                    ctx_chunks.append(join_nonempty([as_str(title), body], sep="\n"))
        prompt_text = join_nonempty([join_nonempty(ctx_chunks, sep="\n\n"), question], sep="\n\n")
        yield [make_message("user", prompt_text)], [make_message("assistant", answer)]

class GSM8K(SFTExporter):
    hf_id = "openai/gsm8k"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        q = as_str(ex.get("question"))
        a = as_str(ex.get("answer"))
        yield [make_message("user", q)], [make_message("assistant", a)]

class MetaMathQA(SFTExporter):
    hf_id = "meta-math/MetaMathQA"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        # common keys: 'query', 'response' (fallback to question/answer/instruction/output)
        q = as_str(ex.get("query") or ex.get("question") or ex.get("instruction"))
        a = as_str(ex.get("response") or ex.get("answer") or ex.get("output"))
        if q and a:
            yield [make_message("user", q)], [make_message("assistant", a)]

class MathInstruct(SFTExporter):
    hf_id = "TIGER-Lab/MathInstruct"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        # unified structure varies; try instruction/output first
        q = as_str(ex.get("instruction") or ex.get("question") or ex.get("query"))
        inp = as_str(ex.get("input") or ex.get("context") or "")
        a = as_str(ex.get("output") or ex.get("answer") or ex.get("response"))
        prompt_text = join_nonempty([q, inp], sep="\n\n") if inp else (q if q else "")
        if prompt_text and a:
            yield [make_message("user", prompt_text)], [make_message("assistant", a)]

class CodeAlpaca20K(SFTExporter):
    hf_id = "sahil2801/CodeAlpaca-20k"
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        inst = as_str(ex.get("instruction"))
        inp  = as_str(ex.get("input"))
        out  = as_str(ex.get("output"))
        prompt_text = join_nonempty([inst, inp], sep="\n\n") if inp else inst
        yield [make_message("user", prompt_text)], [make_message("assistant", out)]

class GlaiveCodeAssistant(SFTExporter):
    hf_id = "glaiveai/glaive-code-assistant"
    split_order = ["train"]
    def prompt_responses(self, ex: Dict) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        # dataset variants exist; try several key pairs
        q = as_str(ex.get("question") or ex.get("prompt") or ex.get("instruction"))
        a = as_str(ex.get("answer") or ex.get("response") or ex.get("output"))
        if q and a:
            yield [make_message("user", q)], [make_message("assistant", a)]
        # some rows may be conversation-style
        conv = ex.get("conversations") or ex.get("messages")
        if isinstance(conv, list):
            history_msgs: List[Dict] = []
            for turn in conv:
                role = as_str(turn.get("from") or turn.get("role"))
                val  = as_str(turn.get("value") or turn.get("content"))
                tag  = "user" if role.lower().startswith(("hum","user")) else ("assistant" if role.lower().startswith("ass") else "system")
                if tag == "assistant":
                    yield list(history_msgs), [make_message("assistant", val)]
                else:
                    history_msgs.append(make_message(tag, val))

# --------- CPT (pretraining) datasets ---------

class OpenWebMath(TextExporter):
    hf_id = "EleutherAI/openwebmath"
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("text"))

# mC4 (via a maintained sampling wrapper) — non-English
class MC4Lang(TextExporter):
    hf_id = "bertin-project/mc4-sampling"
    def __init__(self, lang: str):
        self.config = lang
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("text"))

# OSCAR 23.01 (requires accepting terms on HF). Uses 'language' kwarg and 'content' field.
class OSCARLang(TextExporter):
    hf_id = "oscar-corpus/OSCAR-2301"
    def __init__(self, lang: str):
        self.extra_kwargs = {"language": lang}
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("content") or ex.get("text"))

# Code pretraining: The Stack (language-specific)
class TheStackLang(TextExporter):
    hf_id = "bigcode/the-stack-dedup"  # language by data_dir to avoid scanning everything
    def __init__(self, lang_folder: str):
        self.extra_kwargs = {"data_dir": f"data/{lang_folder}"}
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("content"))

# Code pretraining: CodeSearchNet (language-specific configs)
class CodeSearchNetLang(TextExporter):
    hf_id = "code_search_net"
    def __init__(self, language: str):
        self.config = language
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        code_key = first_key(ex, ["func_code_string", "code", "code_tokens"])
        if code_key == "code_tokens":
            return " ".join(ex["code_tokens"])
        return as_str(ex.get(code_key)) if code_key else None

# Code pretraining: CodeParrot (Python)
class CodeParrotClean(TextExporter):
    hf_id = "codeparrot/codeparrot-clean"
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("content"))

# Misc language modeling corpora
class WikiText103(TextExporter):
    hf_id = "Salesforce/wikitext"
    config = "wikitext-103-raw-v1"
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("text"))

class PG19(TextExporter):
    hf_id = "deepmind/pg19"
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("text"))

# Misc: MusicCaps captions as plain text
class MusicCaps(TextExporter):
    hf_id = "google/MusicCaps"
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        return as_str(ex.get("caption"))

class OLMoMix1124Split(TextExporter):
    hf_id = "allenai/olmo-mix-1124"
    def __init__(self, name: str = "dclm"):
        self.config = name
        # dataset maintainers recommended enabling no_checks for all splits after column unification
        self.extra_kwargs = {"verification_mode": "no_checks"}
    split_order = ["train"]
    def text_from_example(self, ex: Dict) -> Optional[str]:
        # DCLM split is a superset of columns; prefer typical text fields
        key = first_key(ex, ["text", "content", "raw_content", "document", "content_text"])
        return as_str(ex.get(key)) if key else None

# --------- registry ---------

REGISTRY: Dict[str, BaseExporter] = {
    # SFT instruction/QA/math/code
    "alpaca_cleaned": AlpacaCleaned(),
    "dolly15k": Dolly15K(),
    "openorca": OpenOrca(),
    "openhermes25": OpenHermes25(),
    "tulu3_sft_mixture": Tulu3SFTMixture(),
    "squad_v2": SQuADv2(),
    "hotpot_qa": HotpotQA(),
    "gsm8k": GSM8K(),
    "metamathqa": MetaMathQA(),
    "mathinstruct": MathInstruct(),
    "codealpaca20k": CodeAlpaca20K(),
    "glaive_code_assistant": GlaiveCodeAssistant(),

    # CPT non-English (world languages)
    "mc4_es": MC4Lang("es"),
    "mc4_de": MC4Lang("de"),
    "oscar_fr": OSCARLang("fr"),

    # CPT math
    "openwebmath": OpenWebMath(),

    # CPT code (programming languages)
    "the_stack_python": TheStackLang("python"),
    "the_stack_javascript": TheStackLang("javascript"),
    "codesearchnet_python": CodeSearchNetLang("python"),
    "codesearchnet_java": CodeSearchNetLang("java"),
    "codeparrot_python": CodeParrotClean(),

    # misc LM text
    "wikitext103": WikiText103(),
    "pg19": PG19(),
    "musiccaps": MusicCaps(),

    "dclm": OLMoMix1124Split("dclm"),
}

def list_datasets() -> str:
    # grouped-ish view for convenience
    lines = [
        "# SFT",
        "alpaca_cleaned, dolly15k, openorca, openhermes25, tulu3_sft_mixture, squad_v2, hotpot_qa, gsm8k, metamathqa, mathinstruct, codealpaca20k, glaive_code_assistant",
        "# CPT (world languages)",
        "mc4_es, mc4_de, oscar_fr",
        "# CPT (math)",
        "openwebmath",
        "# CPT (code)",
        "the_stack_python, the_stack_javascript, codesearchnet_python, codesearchnet_java, codeparrot_python",
        "# Misc",
        "wikitext103, pg19, musiccaps",
    ]
    return "\n".join(lines)

# --------- CLI ---------

def main():
    p = argparse.ArgumentParser(description="Stream HF datasets into a single .jsonl file")
    p.add_argument("--dataset", required=True, help="One of:\n" + list_datasets())
    p.add_argument("--n", type=int, default=None, help="Rows to write (single file mode)")
    p.add_argument("--output_path", required=False, help="Destination .jsonl path (single file mode)")
    p.add_argument("--split_train_val", action="store_true", help="If set, write train.jsonl and val.jsonl to --output_dir")
    p.add_argument("--n_train", type=int, default=None, help="Rows for train split (split mode)")
    p.add_argument("--n_val", type=int, default=None, help="Rows for val split (split mode)")
    p.add_argument("--output_dir", required=False, help="Destination directory (split mode)")
    p.add_argument("--shuffle", type=parse_bool, default=True, help="Shuffle before writing (true/false). Default: true")
    p.add_argument("--seed", type=int, default=0, help="Seed used for shuffling. Default: 0")
    args = p.parse_args()

    name = args.dataset.strip().lower()
    if name not in REGISTRY:
        print("Unknown --dataset. Available:\n" + list_datasets(), file=sys.stderr)
        sys.exit(1)

    try:
        if args.split_train_val:
            if not args.output_dir:
                print("--split_train_val requires --output_dir", file=sys.stderr)
                sys.exit(1)
            written_train, written_val = REGISTRY[name].export_split(
                args.n_train,
                args.n_val,
                args.output_dir,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            print(f"Wrote train={written_train}, val={written_val} rows to {args.output_dir}")
        else:
            if args.n is None or not args.output_path:
                print("Single file mode requires --n and --output_path", file=sys.stderr)
                sys.exit(1)
            REGISTRY[name].export(args.n, args.output_path, shuffle=args.shuffle, seed=args.seed)
            print(f"Wrote up to {args.n} rows to {args.output_path}")
    except Exception as e:
        print(f"Error while exporting {name}: {e}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    main()
