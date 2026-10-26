"""Safe subprocess-based code execution with resource limits.

Executes untrusted Python code in a sandboxed subprocess with
CPU timeout, memory cap, file-write limits via resource.setrlimit(),
and a configurable security policy that restricts filesystem, network,
and dangerous module access.

Mirrored from finetuning-valley/finetuning_valley/code/sandbox.py.
"""

import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""
    success: bool
    stdout: str
    stderr: str
    exception: Optional[str] = None
    timed_out: bool = False


@dataclass
class SandboxConfig:
    """Security policy for the sandboxed subprocess.

    By default everything is locked down: no filesystem access, no network,
    no dangerous modules.  Toggle individual flags to relax restrictions.
    """

    # --- Filesystem ---
    allow_read: bool = False
    allow_write: bool = False

    # --- Network ---
    allow_network: bool = False

    # --- Subprocess / OS ---
    allow_subprocess: bool = False
    allow_os_exec: bool = False

    # --- Modules ---
    # Modules that are always blocked unless explicitly allowed.
    # The defaults cover the most dangerous surface area.
    blocked_modules: Set[str] = field(default_factory=lambda: {
        # Filesystem / OS interaction
        "os", "os.path", "pathlib", "shutil", "glob", "fnmatch",
        "tempfile", "fileinput", "linecache", "tokenize",
        # Subprocess / shell
        "subprocess", "shlex",
        # Network
        "_socket", "socket", "ssl", "http", "http.client", "http.server",
        "urllib", "urllib.request", "urllib.parse", "urllib.error",
        "ftplib", "smtplib", "imaplib", "poplib",
        "xmlrpc", "xmlrpc.client", "xmlrpc.server",
        "requests", "httpx", "aiohttp",
        # Low-level / escape hatches
        "ctypes", "ctypes.util",
        "importlib", "importlib.util", "importlib.abc",
        "code", "codeop", "compileall", "py_compile",
        "inspect",
        "gc", "sys",  # sys is re-injected in a limited form below
        "signal",
        "multiprocessing", "threading", "_thread",
        "webbrowser",
        # Serialization (can be used for arbitrary code execution)
        "pickle", "shelve", "marshal",
    })

    # Modules the user explicitly wants to allow even if they appear in
    # blocked_modules (e.g. allow "os.path" for a specific task).
    extra_allowed_modules: Set[str] = field(default_factory=set)

    # --- Resource limits ---
    max_memory_mb: int = 4096
    timeout: float = 30.0
    max_nproc: int = 64
    max_fsize_mb: int = 10


# Convenient pre-built configs ------------------------------------------------

#: Maximally restrictive — only print() and input() work.
LOCKED_DOWN = SandboxConfig()

#: Allow reading files (but not writing).
READ_ONLY = SandboxConfig(allow_read=True)


def _build_security_policy(cfg: SandboxConfig) -> str:
    """Return Python source that enforces *cfg* inside the subprocess."""

    blocked = cfg.blocked_modules - cfg.extra_allowed_modules

    # We build the policy as a plain string so it can be prepended to the
    # wrapper script that runs inside the subprocess.
    return textwrap.dedent(f"""\
        # ── sandbox security policy ──────────────────────────────────────
        import builtins as _builtins

        _BLOCKED_MODULES = {blocked!r}

        # Modules that are blocked but have safe shims already installed
        # in sys.modules — allow importing them (returns the shim).
        _SHIMMED_MODULES = {{"sys"}}

        # 1. Import hook — block dangerous modules -------------------------
        _real_import = _builtins.__import__
        import sys as _hook_sys

        def _restricted_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if name in _BLOCKED_MODULES or top in _BLOCKED_MODULES:
                # Allow shimmed modules that are already in sys.modules
                if name in _SHIMMED_MODULES and name in _hook_sys.modules:
                    return _hook_sys.modules[name]
                raise ImportError(
                    f"Module '{{name}}' is not available in the sandbox"
                )
            return _real_import(name, *args, **kwargs)

        _builtins.__import__ = _restricted_import

        # 2. Filesystem guards ---------------------------------------------
        _real_open = _builtins.open

        def _restricted_open(file, mode="r", *a, **kw):
            allow_read = {cfg.allow_read!r}
            allow_write = {cfg.allow_write!r}

            # Determine intent
            is_write = any(c in mode for c in "wxa+")
            is_read = not is_write

            if is_read and not allow_read:
                raise PermissionError(
                    "Filesystem read access is disabled in the sandbox"
                )
            if is_write and not allow_write:
                raise PermissionError(
                    "Filesystem write access is disabled in the sandbox"
                )
            return _real_open(file, mode, *a, **kw)

        _builtins.open = _restricted_open

        # 3. Network guard -------------------------------------------------
        if not {cfg.allow_network!r}:
            # Even if socket somehow gets imported (e.g. as a transitive dep
            # of an allowed module), cripple it.
            import types as _types
            _fake_socket = _types.ModuleType("socket")
            _fake_socket.socket = lambda *a, **kw: (_ for _ in ()).throw(
                PermissionError("Network access is disabled in the sandbox")
            )
            import sys as _sys
            _sys.modules["socket"] = _fake_socket

        # 4. Provide a limited sys ─────────────────────────────────────────
        # We block the full sys module but the code may need sys.stdin /
        # sys.stdout for print()/input().  Inject a thin shim.
        import sys as _real_sys
        import types as _types2
        _shim_sys = _types2.ModuleType("sys")
        _shim_sys.stdin = _real_sys.stdin
        _shim_sys.stdout = _real_sys.stdout
        _shim_sys.stderr = _real_sys.stderr
        _shim_sys.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
        _shim_sys.maxsize = _real_sys.maxsize
        _shim_sys.version = _real_sys.version
        _shim_sys.version_info = _real_sys.version_info
        _shim_sys.platform = _real_sys.platform
        _real_sys.modules["sys"] = _shim_sys

        # 5. Block io module's file-opening surface ────────────────────────
        # (io.open is another path to filesystem access)
        import io as _io
        _orig_io_open = _io.open
        def _restricted_io_open(file, mode="r", *a, **kw):
            is_write = any(c in mode for c in "wxa+")
            if not is_write and not {cfg.allow_read!r}:
                raise PermissionError("Filesystem read access is disabled in the sandbox")
            if is_write and not {cfg.allow_write!r}:
                raise PermissionError("Filesystem write access is disabled in the sandbox")
            return _orig_io_open(file, mode, *a, **kw)
        _io.open = _restricted_io_open

        # Block low-level io classes that bypass io.open
        def _blocked_io_cls(*a, **kw):
            raise PermissionError("Direct file I/O classes are disabled in the sandbox")
        _io.FileIO = _blocked_io_cls
        _io.BufferedReader = _blocked_io_cls
        _io.BufferedWriter = _blocked_io_cls
        _io.BufferedRandom = _blocked_io_cls
        _io.TextIOWrapper = _blocked_io_cls

        # 6. Restrict __builtins__ for exec'd user code ───────────────────
        # Prevent sandbox escape via object hierarchy (e.g.
        # ().__class__.__bases__[0].__subclasses__()) by removing
        # dangerous builtins from the user code's namespace.
        _safe_builtins = {{k: v for k, v in _builtins.__dict__.items()
                          if k not in ('__import__', 'open', 'exec', 'eval',
                                       'compile', 'breakpoint', 'exit', 'quit',
                                       '__loader__', '__spec__')}}
        _safe_builtins['__import__'] = _restricted_import
        _safe_builtins['open'] = _restricted_open
        _safe_builtins['print'] = print
        _safe_builtins['input'] = input

        # ── end security policy ──────────────────────────────────────────
    """)


def execute_code(
    code: str,
    timeout: Optional[float] = None,
    max_memory_mb: Optional[int] = None,
    config: Optional[SandboxConfig] = None,
) -> ExecutionResult:
    """Execute Python code in a sandboxed subprocess.

    Spawns a new Python process with resource limits and a security
    policy, then captures stdout/stderr.  The code is passed via stdin
    to avoid shell injection issues.

    Args:
        code: Python source code to execute.
        timeout: Wall-clock timeout in seconds.  Falls back to config value.
        max_memory_mb: Maximum virtual memory in MB.  Falls back to config value.
        config: Security policy.  Defaults to ``LOCKED_DOWN`` (everything
            disabled except print/input).

    Returns:
        ExecutionResult with success, stdout, stderr, exception, timed_out.
    """
    if config is None:
        config = LOCKED_DOWN

    timeout = timeout if timeout is not None else config.timeout
    max_memory_mb = max_memory_mb if max_memory_mb is not None else config.max_memory_mb

    security_policy = _build_security_policy(config)

    # Build a wrapper script that sets resource limits, then exec's the code.
    wrapper = textwrap.dedent(f"""\
        import resource
        import sys

        # Memory limit
        mem_bytes = {max_memory_mb} * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, resource.error):
            pass  # Some systems don't support RLIMIT_AS

        # CPU time limit (backup for wall-clock timeout)
        cpu_seconds = int({timeout}) + 5
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, resource.error):
            pass

        # Prevent fork bombs
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, ({config.max_nproc}, {config.max_nproc}))
        except (ValueError, resource.error):
            pass

        # Limit file writes
        fsize = {config.max_fsize_mb} * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except (ValueError, resource.error):
            pass

    """)

    # Combine: resource limits → security policy → user code execution
    full_wrapper = wrapper + security_policy + textwrap.dedent("""\
        # Read and execute the code from stdin
        _user_code = sys.stdin.read()
        exec(_user_code, {"__name__": "__main__", "__builtins__": _safe_builtins})
    """)

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_wrapper],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            exception=result.stderr.strip() if not success else None,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            exception="TimeoutExpired",
            timed_out=True,
        )
    except BaseException as e:
        # Catch *everything* (MemoryError on stdout capture, OSError on
        # fork failure, KeyboardInterrupt-lookalikes from signals, etc.).
        # The contract is: execute_code never raises — failures become a
        # non-success ExecutionResult so the caller can mark the candidate
        # as incorrect.
        try:
            msg = repr(e)
        except Exception:
            msg = "<unrepr-able exception>"
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=msg,
            exception=msg,
            timed_out=False,
        )
