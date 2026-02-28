import ast
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from ..agents.tools import get_tool_functions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe stdout sink — captures output instead of leaking to real stdout
# ---------------------------------------------------------------------------

class _SandboxPrinter:
    """Drop-in replacement for ``print()`` inside the sandbox.

    Captures output in a bounded buffer instead of writing to the host
    process's stdout, preventing data exfiltration.
    """

    MAX_CHARS = 10_000

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._chars = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self._chars >= self.MAX_CHARS:
            return
        text = " ".join(str(a) for a in args)
        self._buffer.append(text)
        self._chars += len(text)

    @property
    def output(self) -> str:
        return "\n".join(self._buffer)


class SandboxSecurityError(Exception):
    """Raised when generated code violates sandbox security constraints."""


class SandboxTimeoutError(Exception):
    """Raised when sandbox execution exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Safe builtins whitelist — everything else (including __import__,
# eval, exec, open, compile, getattr, setattr, delattr, globals,
# locals, vars, dir, type, breakpoint, exit, quit, input, memoryview,
# classmethod, staticmethod, super, property, …) is blocked.
# ---------------------------------------------------------------------------
_SAFE_BUILTINS = {
    # Types / constructors
    "bool": bool,
    "bytes": bytes,
    "complex": complex,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "set": set,
    "str": str,
    "tuple": tuple,
    # Functional helpers
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "filter": filter,
    "isinstance": isinstance,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    # print is replaced by _SandboxPrinter() per-execution (see _build_globals)
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
    # Constants
    "True": True,
    "False": False,
    "None": None,
}

# AST node types that are never allowed in generated code
_BLOCKED_AST_NODES = (ast.Import, ast.ImportFrom)

# Attribute names that must never be accessed
_BLOCKED_ATTRIBUTES: Set[str] = {
    "__import__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__class__",
    "__reduce__", "__reduce_ex__",
    "__qualname__", "__module__", "__dict__", "__weakref__",
    "__init_subclass__", "__set_name__",
}

# Builtin names that must never appear as plain Name references
_BLOCKED_NAMES: Set[str] = {
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "breakpoint", "exit", "quit", "input",
    "memoryview", "type", "super", "classmethod", "staticmethod",
    "property", "__build_class__",
}


class CodeSandbox:
    """A hardened sandbox to execute the LLM-generated async orchestration script.

    Security layers:
      1. **AST validation** – the code is parsed and walked *before* execution.
         Import statements, dangerous attribute accesses (``__globals__``, etc.),
         and references to blocked builtins (``eval``, ``exec``, ``open`` …) are
         rejected at parse time.
      2. **Restricted ``__builtins__``** – only a curated whitelist of safe
         builtins is exposed inside the ``exec`` namespace.  ``__import__``,
         ``eval``, ``exec``, ``open``, ``compile``, ``getattr`` and friends
         are simply absent.
      3. **Execution timeout** – ``asyncio.wait_for`` enforces an upper bound
         on wall-clock time (default 60 s, configurable via *timeout_seconds*).

    Wraps expert tool functions with tracking so that the caller can retrieve
    which experts were invoked and what they returned — information that is
    propagated back into the LangGraph state.
    """

    # Expert tools are built dynamically from the registry at init time
    # via ``get_tool_functions()``.

    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self._call_log: Dict[str, str] = {}
        self._build_globals()

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------
    def _make_tracked(self, expert_type: str, original_fn):
        """Return an async wrapper that delegates to *original_fn* and logs the call."""
        async def _tracked(prompt: str) -> str:
            result = await original_fn(prompt)
            # Keep the *last* response per expert type (adequate for state)
            self._call_log[expert_type] = result
            return result
        # Preserve a human-readable name for debugging
        _tracked.__name__ = f"query_{expert_type}_expert"
        return _tracked

    def _build_globals(self):
        """Construct the globals dict injected into the sandbox, with tracked wrappers."""
        tool_fns = get_tool_functions()
        self.allowed_globals: Dict[str, Any] = {
            # Locked-down builtins — the ONLY builtins the code can see
            "__builtins__": {
                **_SAFE_BUILTINS,
                # print is a per-execution sandbox printer (replaced in execute())
                "print": _SandboxPrinter(),
            },
            "asyncio": asyncio,
            # Tracked expert wrappers (dynamically generated from registry)
            **{
                f"query_{name}_expert": self._make_tracked(name, fn)
                for name, fn in tool_fns.items()
            },
        }

    # ------------------------------------------------------------------
    # AST validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_code(code: str) -> None:
        """Parse *code* and reject it if it contains dangerous constructs.

        Raises ``SandboxSecurityError`` on violation, ``SyntaxError`` on
        unparseable code.
        """
        tree = ast.parse(code)  # may raise SyntaxError — that's fine

        has_orchestrate = False

        for node in ast.walk(tree):
            # 1. Block import / import-from
            if isinstance(node, _BLOCKED_AST_NODES):
                raise SandboxSecurityError(
                    f"Import statements are not allowed in sandbox code "
                    f"(line {getattr(node, 'lineno', '?')})"
                )

            # 2. Block dangerous attribute accesses like obj.__globals__
            if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRIBUTES:
                raise SandboxSecurityError(
                    f"Access to '{node.attr}' is not allowed "
                    f"(line {getattr(node, 'lineno', '?')})"
                )

            # 3. Block references to dangerous builtin names
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise SandboxSecurityError(
                    f"Reference to '{node.id}' is not allowed "
                    f"(line {getattr(node, 'lineno', '?')})"
                )

            # 4. Check for async def orchestrate()
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "orchestrate":
                has_orchestrate = True

        if not has_orchestrate:
            raise ValueError(
                "The generated script must define an 'async def orchestrate():' function."
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute the provided async Python code.

        The code is expected to define an ``async def orchestrate():`` function.

        Security: code is first validated via AST analysis, then executed
        with restricted builtins inside an ``asyncio.wait_for`` timeout.

        Returns:
            A dict with keys ``result``, ``selected_experts``, and
            ``expert_responses`` so the caller can propagate metadata back
            into the LangGraph state.
        """
        # ---- Step 1: Static analysis --------------------------------
        self.validate_code(code)  # raises on violation

        # ---- Step 2: Execute with restrictions -----------------------
        self._call_log = {}
        local_vars: Dict[str, Any] = {}

        # Fresh printer per execution to avoid cross-request data leakage
        sandbox_printer = _SandboxPrinter()
        self.allowed_globals["__builtins__"]["print"] = sandbox_printer

        try:
            # Compile and execute the definition of the functions
            exec(code, self.allowed_globals, local_vars)  # noqa: S102

            orchestrate_func = local_vars['orchestrate']

            if not asyncio.iscoroutinefunction(orchestrate_func):
                raise ValueError("The 'orchestrate' function must be an async function.")

            # ---- Step 3: Run with timeout ---------------------------
            result = await asyncio.wait_for(
                orchestrate_func(),
                timeout=self.timeout_seconds,
            )

            return {
                "result": str(result),
                "selected_experts": list(self._call_log.keys()),
                "expert_responses": dict(self._call_log),
            }
        except asyncio.TimeoutError:
            logger.error(
                f"Sandbox execution timed out after {self.timeout_seconds}s"
            )
            raise SandboxTimeoutError(
                f"Orchestration script exceeded the {self.timeout_seconds}s timeout. "
                f"Simplify the script or increase the timeout."
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise
