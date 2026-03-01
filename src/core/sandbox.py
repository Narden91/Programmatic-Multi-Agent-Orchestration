import ast
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from .agents import query_agent as real_query_agent, AgentResult
from .memory import EphemeralMemory
from .tracing import Tracer

logger = logging.getLogger(__name__)

class _SandboxPrinter:
    """Drop-in replacement for ``print()`` inside the sandbox.
    Captures output in a bounded buffer instead of writing to the host process's stdout."""
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


_SAFE_BUILTINS = {
    "bool": bool, "bytes": bytes, "complex": complex, "dict": dict, "float": float,
    "frozenset": frozenset, "int": int, "list": list, "set": set, "str": str, "tuple": tuple,
    "abs": abs, "all": all, "any": any, "enumerate": enumerate, "filter": filter,
    "isinstance": isinstance, "len": len, "map": map, "max": max, "min": min,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "sorted": sorted,
    "sum": sum, "zip": zip, "True": True, "False": False, "None": None,
}

_BLOCKED_AST_NODES = (
    ast.Import, ast.ImportFrom, ast.Pow, ast.Mult, ast.LShift, ast.RShift, ast.MatMult,
)

_BLOCKED_ATTRIBUTES: Set[str] = {
    "__import__", "__subclasses__", "__bases__", "__mro__", "__globals__", "__code__", "__builtins__", "__class__",
    "__reduce__", "__reduce_ex__", "__qualname__", "__module__", "__dict__", "__weakref__", "__init_subclass__", "__set_name__",
}

_BLOCKED_NAMES: Set[str] = {
    "__import__", "eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "breakpoint", "exit", "quit", "input", "memoryview", "type", "super", "classmethod", "staticmethod",
    "property", "__build_class__",
}


class CodeSandbox:
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        # These will be set per execution
        self.tracer: Optional[Tracer] = None
        self.memory: Optional[EphemeralMemory] = None
        self._call_log: Dict[str, str] = {}
        
    def _make_tracked_query_agent(self):
        """Creates a sandboxed version of query_agent that is traced and logs results."""
        async def _tracked(agent_type: str, prompt: str, context_ids: Optional[List[str]] = None) -> Any:
            span = self.tracer.start_span(
                name=f"query_agent_{agent_type}",
                span_type="agent",
                inputs={"agent_type": agent_type, "prompt": prompt, "context_ids": context_ids}
            )
            try:
                # We return the AgentResult object to the sandbox, so they can use .text
                # If they pass it to str(), it might not work well, so we might need
                # to instruct the orchestrator to use result.text.
                result = await real_query_agent(agent_type, prompt, context_ids)
                self._call_log[agent_type] = result.text
                
                self.tracer.end_span(span, outputs={"text": result.text}, metrics={"tokens": result.token_count})
                return result
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise
        return _tracked

    def _make_tracked_memory(self):
        """Creates sandboxed memory functions, traced."""
        async def _store(key: str, text: str, metadata: dict = None) -> str:
            span = self.tracer.start_span("memory_store", "memory", {"key": key, "text_len": len(text)})
            try:
                res = await self.memory.store(key, text, metadata)
                self.tracer.end_span(span, outputs={"key": res})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise

        async def _search(query: str, top_k: int = 5) -> List[dict]:
            span = self.tracer.start_span("memory_search", "memory", {"query": query, "top_k": top_k})
            try:
                res = await self.memory.search(query, top_k)
                self.tracer.end_span(span, outputs={"results_count": len(res)})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise
                
        async def _compress(query: str, top_k: int = 5) -> str:
            span = self.tracer.start_span("memory_compress", "memory", {"query": query, "top_k": top_k})
            try:
                res = await self.memory.compress_context(query, top_k)
                self.tracer.end_span(span, outputs={"compressed_len": len(res)})
                return res
            except Exception as e:
                self.tracer.end_span(span, error=e)
                raise

        return _store, _search, _compress

    @staticmethod
    def validate_code(code: str) -> None:
        tree = ast.parse(code)
        has_orchestrate = False
        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_AST_NODES):
                raise SandboxSecurityError(f"AST node {type(node).__name__} not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRIBUTES:
                raise SandboxSecurityError(f"Access to '{node.attr}' not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise SandboxSecurityError(f"Reference to '{node.id}' not allowed (line {getattr(node, 'lineno', '?')})")
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "orchestrate":
                has_orchestrate = True
        if not has_orchestrate:
            raise ValueError("The generated script must define an 'async def orchestrate():' function.")

    async def execute(self, code: str) -> Dict[str, Any]:
        self.validate_code(code)
        
        # Initialize execution-specific resources
        self._call_log = {}
        self.tracer = Tracer()
        self.memory = EphemeralMemory()
        sandbox_printer = _SandboxPrinter()
        
        store_fn, search_fn, compress_fn = self._make_tracked_memory()
        
        allowed_globals = {
            "__builtins__": {
                **_SAFE_BUILTINS,
                "print": sandbox_printer,
            },
            "asyncio": asyncio,
            "query_agent": self._make_tracked_query_agent(),
            "memory_store": store_fn,
            "memory_search": search_fn,
            "compress_context": compress_fn
        }

        local_vars: Dict[str, Any] = {}
        try:
            exec(code, allowed_globals, local_vars)
            orchestrate_func = local_vars['orchestrate']
            if not asyncio.iscoroutinefunction(orchestrate_func):
                raise ValueError("The 'orchestrate' function must be an async function.")

            result = await asyncio.wait_for(orchestrate_func(), timeout=self.timeout_seconds)
            
            return {
                "result": str(result),
                "selected_experts": list(self._call_log.keys()),
                "expert_responses": dict(self._call_log),
                "trace": self.tracer.get_trace(),
                "sandbox_output": sandbox_printer.output
            }
        except asyncio.TimeoutError:
            logger.error(f"Sandbox execution timed out after {self.timeout_seconds}s")
            raise SandboxTimeoutError(f"Orchestration script exceeded the {self.timeout_seconds}s timeout.")
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise
        finally:
            if self.memory:
                self.memory.cleanup()
