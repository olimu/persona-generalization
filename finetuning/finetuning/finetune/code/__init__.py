"""Code execution and evaluation library for MBPP + HumanEval benchmarks.

Mirrored from finetuning-valley to keep finetuning self-contained.
"""

from .sandbox import execute_code, ExecutionResult, SandboxConfig, LOCKED_DOWN, READ_ONLY
from .extract import extract_code
from .benchmarks import (
    load_mbpp,
    load_humaneval,
    check_mbpp_solution,
    check_humaneval_solution,
)
