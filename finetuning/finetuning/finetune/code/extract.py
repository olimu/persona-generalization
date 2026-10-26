"""Permissive code extraction from model output.

Tries multiple strategies to extract Python code from a model response,
including finding individual function definitions scattered among non-code
text (chat turns, prose, etc.).

Also provides HumanEval-specific helpers: indentation repair, class-wrapper
extraction, trailing-junk truncation, and cosmetic fix-ups (line endings,
unicode quotes).

Mirrored from finetuning-valley/finetuning_valley/code/extract.py.
"""

import ast
import re
import textwrap
from typing import List, Optional


# Skip ast.parse entirely for inputs above this size — protects against
# MemoryError on multi-megabyte runaway model generations.
_MAX_PARSE_BYTES = 200_000


def _is_valid_python(code: str) -> bool:
    """Check whether *code* parses as valid Python.

    Returns False on any failure — including ``MemoryError`` from huge
    inputs — so callers can rely on it never raising.
    """
    if len(code) > _MAX_PARSE_BYTES:
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def _find_fenced_blocks(text: str) -> List[str]:
    """Extract all fenced code blocks (```python ... ``` or ``` ... ```)."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def extract_functions(text: str) -> List[str]:
    """Extract all Python function definitions from text.

    Scans line-by-line for ``def`` statements and collects the function
    body by tracking indentation.  Handles functions scattered among
    arbitrary non-code text (chat turns, prose, ``<assistant>``/``<user>``
    tags, etc.).

    Returns:
        List of syntactically valid Python function strings.
    """
    lines = text.split("\n")
    functions = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if re.match(r"(async\s+)?def\s+\w+\s*\(", stripped):
            def_indent = len(line) - len(stripped)
            # Collect preceding decorator lines at the same indent
            decorators = []
            for back in range(i - 1, -1, -1):
                prev = lines[back]
                prev_stripped = prev.lstrip()
                prev_indent = len(prev) - len(prev_stripped)
                if prev_indent == def_indent and prev_stripped.startswith("@"):
                    decorators.append(prev)
                elif prev_stripped == "":
                    continue
                else:
                    break
            decorators.reverse()
            func_lines = decorators + [line]
            i += 1
            # Collect body: lines indented more than the def, or blank lines
            trailing_blanks = 0
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "":
                    # Buffer blank lines — only include if body continues
                    func_lines.append(next_line)
                    trailing_blanks += 1
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent > def_indent:
                    trailing_blanks = 0
                    func_lines.append(next_line)
                    i += 1
                else:
                    break
            # Trim trailing blank lines
            if trailing_blanks > 0:
                func_lines = func_lines[:-trailing_blanks]
            func_text = "\n".join(func_lines)
            if _is_valid_python(func_text):
                functions.append(func_text)
        else:
            i += 1
    return functions


def extract_code(response: str) -> str:
    """Extract Python code from a model response.

    Tries multiple strategies in order:
    1. Fenced code blocks (``\\`\\`\\`python ... \\`\\`\\`` or ``\\`\\`\\` ... \\`\\`\\```)
    2. Raw code (response starts with def/class/import) — parse as-is
    3. Individual function definitions scattered in text
    4. Largest contiguous block that parses as valid Python
    5. Fallback: return the entire response stripped

    Never raises — on any internal failure (including ``MemoryError`` on
    pathological inputs) returns ``""`` so the caller can mark the
    candidate as incorrect.
    """
    try:
        return _extract_code_inner(response)
    except Exception:
        return ""


def _extract_code_inner(response: str) -> str:
    stripped = response.strip()

    # 1. Fenced code blocks — concatenate all blocks
    blocks = _find_fenced_blocks(response)
    if blocks:
        combined = "\n\n".join(b.strip() for b in blocks)
        if _is_valid_python(combined):
            return combined
        # If combined doesn't parse, try returning just the first valid block
        for block in blocks:
            if _is_valid_python(block.strip()):
                return block.strip()
        # Return combined anyway — it may be executable even if not parseable
        return combined

    # 2. Raw code — response starts with def/class/import/async def
    if re.match(r"^(async\s+def |def |class |import |from )", stripped):
        if _is_valid_python(stripped):
            return stripped

    # 3. Extract individual functions and combine them
    functions = extract_functions(response)
    if functions:
        combined = "\n\n".join(functions)
        if _is_valid_python(combined):
            return combined
        # Return the longest valid function
        return max(functions, key=len)

    # 4. Find the largest contiguous block that parses as valid Python.
    lines = stripped.split("\n")
    best = ""
    for start in range(len(lines)):
        for end in range(len(lines), start, -1):
            candidate = "\n".join(lines[start:end]).strip()
            if len(candidate) > len(best) and _is_valid_python(candidate):
                best = candidate
                break  # found longest from this start
    if best:
        return best

    # 5. Fallback
    return stripped


# ---------------------------------------------------------------------------
# HumanEval-specific helpers
# ---------------------------------------------------------------------------


def fix_line_endings(code: str) -> str:
    """Replace Windows-style ``\\r\\n`` with ``\\n``."""
    return code.replace("\r\n", "\n")


def fix_unicode_quotes(code: str) -> str:
    """Replace common unicode quotes with ASCII equivalents."""
    table = str.maketrans({
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u2013": "-",  # –  (en-dash)
        "\u2014": "-",  # —  (em-dash)
    })
    return code.translate(table)


def strip_markdown_fences(code: str) -> List[str]:
    """Extract code from markdown fences; return list of candidates.

    If there are fenced blocks, returns each block as a candidate.
    If there are no fences, returns the original string as-is.
    """
    blocks = _find_fenced_blocks(code)
    if blocks:
        return [b.strip() for b in blocks]
    return [code]


def repair_indentation(code: str, target_indent: int = 4) -> str:
    """Normalize indentation to *target_indent* spaces.

    Handles:
    - Tabs → spaces
    - Consistent 2-space → 4-space conversion
    - Zero-indent function body (adds *target_indent* spaces)
    """
    # Replace tabs with spaces
    code = code.expandtabs(target_indent)

    lines = code.split("\n")
    if not lines:
        return code

    # Detect the minimum non-zero indent used in the body
    indents = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and not stripped.startswith("def ") and not stripped.startswith("async def "):
            indent = len(line) - len(stripped)
            if indent > 0:
                indents.append(indent)

    if not indents:
        return code

    min_indent = min(indents)

    # If indentation is already at the target, nothing to do
    if min_indent == target_indent:
        return code

    # Scale factor: if min_indent=2 and target=4, multiply all by 2
    if min_indent > 0:
        scale = target_indent / min_indent
        new_lines = []
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                new_lines.append("")
            else:
                old_indent = len(line) - len(stripped)
                new_indent = int(old_indent * scale)
                new_lines.append(" " * new_indent + stripped)
        return "\n".join(new_lines)

    return code


def indent_as_body(code: str, indent: int = 4) -> str:
    """Add *indent* spaces to every non-blank line — for zero-indent bodies."""
    prefix = " " * indent
    lines = code.split("\n")
    return "\n".join(
        (prefix + line) if line.strip() else line
        for line in lines
    )


def truncate_trailing_junk(code: str) -> str:
    """Remove trailing non-function content (test code, prints, etc.).

    Truncates at the first top-level line that looks like test/driver code:
    - ``if __name__`` blocks
    - top-level ``print(`` / ``assert `` / ``check(``
    - ``# Test`` / ``# Example`` comments followed by code
    """
    lines = code.split("\n")
    truncate_at = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Only consider top-level lines (indent == 0)
        if indent != 0 or not stripped:
            continue

        # Skip function/class/import definitions and decorators
        if re.match(r"(async\s+)?def\s+|class\s+|import\s+|from\s+\w+\s+import|@", stripped):
            continue

        # Truncation signals
        if any([
            stripped.startswith("if __name__"),
            stripped.startswith("print("),
            stripped.startswith("assert "),
            stripped.startswith("check("),
            re.match(r"#\s*(Test|Example|Usage|Driver)", stripped, re.IGNORECASE),
            # Direct entry_point call at top level, e.g. ``foo(1, 2)``
            re.match(r"\w+\(", stripped) and not stripped.startswith("def "),
        ]):
            truncate_at = i
            break

    if truncate_at is not None:
        # Trim trailing blank lines
        result_lines = lines[:truncate_at]
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()
        return "\n".join(result_lines)

    return code


def extract_class_method(code: str, method_name: str) -> Optional[str]:
    """Extract a method from a ``class Solution:`` wrapper.

    Removes the ``self`` parameter and de-indents by one level.
    Also strips ``@staticmethod`` decorators.
    """
    # Find the class body
    m = re.search(
        r"class\s+\w+.*?:\s*\n(.*)",
        code,
        re.DOTALL,
    )
    if not m:
        return None

    class_body = m.group(1)

    # Extract the method
    pattern = rf"(?:[ \t]*@staticmethod\s*\n)?[ \t]+(?:async\s+)?def\s+{re.escape(method_name)}\s*\("
    method_match = re.search(pattern, class_body)
    if not method_match:
        # Try any method name
        pattern = r"(?:[ \t]*@staticmethod\s*\n)?[ \t]+(?:async\s+)?def\s+\w+\s*\("
        method_match = re.search(pattern, class_body)
    if not method_match:
        return None

    # Find the method start in the class body and extract it
    method_start = method_match.start()
    method_text = class_body[method_start:]

    # Extract individual functions from the method text
    functions = extract_functions(method_text)
    if not functions:
        return None

    func = functions[0]

    # Remove @staticmethod decorator
    func = re.sub(r"^\s*@staticmethod\s*\n", "", func)

    # De-indent by one level
    func = textwrap.dedent(func)

    # Remove `self` parameter
    func = re.sub(
        r"(def\s+\w+\s*\(\s*)self\s*,?\s*",
        r"\1",
        func,
    )

    # Clean up empty parens with extra space: ``def foo( )`` → ``def foo()``
    func = re.sub(r"\(\s+\)", "()", func)

    return func


def rename_function(code: str, old_name: str, new_name: str) -> str:
    """Rename a function definition (the ``def`` line only)."""
    if old_name == new_name:
        return code
    return re.sub(
        rf"((?:async\s+)?def\s+){re.escape(old_name)}(\s*\()",
        rf"\g<1>{new_name}\g<2>",
        code,
    )


def normalize_candidate(raw: str) -> str:
    """Apply all cosmetic fixes to a raw candidate string."""
    code = fix_line_endings(raw)
    code = fix_unicode_quotes(code)
    code = code.rstrip() + "\n"
    return code
