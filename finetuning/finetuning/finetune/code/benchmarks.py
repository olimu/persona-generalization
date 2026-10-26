"""MBPP + HumanEval dataset loading and test harnesses.

Mirrored from finetuning-valley/finetuning_valley/code/benchmarks.py.
"""

import re
from typing import List

from .sandbox import execute_code


def load_mbpp(split: str = "sanitized", dataset_dir: str | None = None) -> List[dict]:
    """Load MBPP dataset.

    Args:
        split: "sanitized" (427 problems, cleaner prompts) or "test" (500 problems).
        dataset_dir: If set, load from ``{dataset_dir}/google-research-datasets--mbpp``
            (saved via ``Dataset.save_to_disk``) instead of downloading from HuggingFace.

    Returns:
        List of dicts with keys: task_id, prompt, code, test_list, test_imports.
    """
    import os
    if dataset_dir is not None:
        from datasets import load_from_disk
        config = "sanitized" if split == "sanitized" else "full"
        local_path = os.path.join(dataset_dir, f"google-research-datasets--mbpp--{config}")
        ds = load_from_disk(local_path)
    else:
        from datasets import load_dataset
        if split == "sanitized":
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        else:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)

    tasks = []
    for row in ds:
        tasks.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "code": row["code"],
            "test_list": row["test_list"],
            "test_imports": row.get("test_imports", []),
        })
    return tasks


def load_humaneval(dataset_dir: str | None = None) -> List[dict]:
    """Load HumanEval dataset.

    Args:
        dataset_dir: If set, load from ``{dataset_dir}/openai--openai_humaneval``
            instead of downloading from HuggingFace.

    Returns:
        List of dicts with keys: task_id, prompt, canonical_solution, test, entry_point.
    """
    import os
    if dataset_dir is not None:
        from datasets import load_from_disk
        local_path = os.path.join(dataset_dir, "openai--openai_humaneval")
        ds = load_from_disk(local_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
    tasks = []
    for row in ds:
        tasks.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return tasks


def check_mbpp_solution(
    code: str,
    task: dict,
    timeout: float = 30.0,
    max_memory_mb: int = 4096,
) -> bool:
    """Check if generated code passes MBPP test assertions.

    Constructs: test_imports + generated_code + test_assertions
    Executes in sandbox. Returns True if all assertions pass.

    Memory exhaustion (subprocess hits ``max_memory_mb``) yields a failed
    ``ExecutionResult`` rather than raising — the solution is marked incorrect.
    """
    try:
        imports = "\n".join(task["test_imports"]) if task["test_imports"] else ""
        tests = "\n".join(task["test_list"])
        full_code = imports + "\n" + code + "\n" + tests
        result = execute_code(full_code, timeout=timeout, max_memory_mb=max_memory_mb)
        return result.success
    except Exception:
        # Any failure (MemoryError on string ops, OSError spawning subprocess,
        # etc.) is treated as the solution being incorrect rather than crashing
        # the surrounding evaluation.
        return False


def check_humaneval_solution(
    code: str,
    task: dict,
    timeout: float = 30.0,
    max_memory_mb: int = 4096,
) -> bool:
    """Check if generated code passes HumanEval tests.

    Implements a maximally-lenient, multi-strategy assembly approach:

    1. **Direct concatenation** — ``prompt + completion`` (the "clean" case).
    2. **Standalone function** — the model repeated the signature; use code as-is.
    3. **Standalone with extras** — code contains imports/helpers around the
       entry-point function; use the entire completion.
    4. **Indentation repair** — re-indent and retry strategies 1–3.

    Each assembled program is syntax-checked before execution to avoid
    wasting sandbox time on obviously broken code.

    Returns ``True`` as soon as *any* strategy passes.
    """
    from .extract import (
        repair_indentation, indent_as_body, truncate_trailing_junk,
        normalize_candidate,
    )

    try:
        return _check_humaneval_solution_inner(
            code, task, timeout, max_memory_mb,
            normalize_candidate, repair_indentation, indent_as_body,
            truncate_trailing_junk,
        )
    except Exception:
        return False


def _check_humaneval_solution_inner(
    code: str,
    task: dict,
    timeout: float,
    max_memory_mb: int,
    normalize_candidate,
    repair_indentation,
    indent_as_body,
    truncate_trailing_junk,
) -> bool:
    code = normalize_candidate(code)
    entry_point = task["entry_point"]
    prompt = task["prompt"]
    test_suffix = "\n" + task["test"] + f"\ncheck({entry_point})\n"

    def _try_exec(program: str) -> bool:
        """Syntax-check then execute; return True on pass."""
        try:
            compile(program, "<string>", "exec")
        except Exception:
            # SyntaxError, ValueError (null bytes), MemoryError on huge
            # programs, etc. — treat as failed candidate.
            return False
        try:
            return execute_code(
                program, timeout=timeout, max_memory_mb=max_memory_mb,
            ).success
        except Exception:
            return False

    def _try_strategies(candidate: str) -> bool:
        """Try multiple assembly strategies for a single candidate."""
        # Strategy 1: direct concatenation (completion is just the body)
        if _try_exec(prompt + candidate + test_suffix):
            return True

        # Strategy 2: standalone function (model repeated the signature)
        has_entry = re.search(
            rf"(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(",
            candidate,
        )
        if has_entry:
            if _try_exec(candidate + test_suffix):
                return True

        # Strategy 3: entry point defined among other top-level code
        # (imports, helpers) — same as strategy 2 but the regex already
        # checked; also catches cases where the function name differs.
        func_names = re.findall(
            r"^(?:async\s+)?def\s+(\w+)\s*\(", candidate, re.MULTILINE,
        )
        if func_names and not has_entry:
            # Try with an alias: last defined function → entry_point
            aliased = candidate + f"\n{entry_point} = {func_names[-1]}\n"
            if _try_exec(aliased + test_suffix):
                return True

        # Strategy 4: zero-indent body — add 4-space indent, prepend prompt
        lines = candidate.split("\n")
        non_blank = [l for l in lines if l.strip()]
        if non_blank and all(not l.startswith(" ") and not l.startswith("\t") for l in non_blank):
            # Looks like the body was output at column 0
            indented = indent_as_body(candidate)
            if _try_exec(prompt + indented + test_suffix):
                return True

        # Strategy 5: indentation repair (2-space, tabs, inconsistent)
        repaired = repair_indentation(candidate)
        if repaired != candidate:
            if _try_exec(prompt + repaired + test_suffix):
                return True
            if has_entry and _try_exec(repaired + test_suffix):
                return True

        return False

    # First, try the code as given
    if _try_strategies(code):
        return True

    # Try after truncating trailing junk (test code, prints, etc.)
    truncated = truncate_trailing_junk(code)
    if truncated != code and _try_strategies(truncated):
        return True

    return False
