import asyncio
import logging
from typing import Any, Dict, List
from ..agents.tools import (
    query_technical_expert,
    query_analytical_expert,
    query_creative_expert,
    query_general_expert
)

logger = logging.getLogger(__name__)


class CodeSandbox:
    """A minimal sandbox to execute the LLM-generated async orchestration script.
    
    Wraps expert tool functions with tracking so that the caller can retrieve
    which experts were invoked and what they returned — information that is
    propagated back into the LangGraph state.
    """
    
    # Mapping from canonical expert name to the underlying tool function
    _EXPERT_TOOLS = {
        "technical": query_technical_expert,
        "analytical": query_analytical_expert,
        "creative": query_creative_expert,
        "general": query_general_expert,
    }

    def __init__(self):
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
        self.allowed_globals: Dict[str, Any] = {
            "asyncio": asyncio,
            # Tracked expert wrappers
            **{
                f"query_{name}_expert": self._make_tracked(name, fn)
                for name, fn in self._EXPERT_TOOLS.items()
            },
            # Safe builtins
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, code: str) -> Dict[str, Any]:
        """Execute the provided async Python code.

        The code is expected to define an ``async def orchestrate():`` function.

        Returns:
            A dict with keys ``result``, ``selected_experts``, and
            ``expert_responses`` so the caller can propagate metadata back
            into the LangGraph state.
        """
        # Reset tracking for this invocation
        self._call_log = {}
        local_vars: Dict[str, Any] = {}

        try:
            # Compile and execute the definition of the functions
            exec(code, self.allowed_globals, local_vars)

            if 'orchestrate' not in local_vars:
                raise ValueError(
                    "The generated script must define an 'async def orchestrate():' function."
                )

            orchestrate_func = local_vars['orchestrate']

            if not asyncio.iscoroutinefunction(orchestrate_func):
                raise ValueError("The 'orchestrate' function must be an async function.")

            # Run the extracted async function
            result = await orchestrate_func()

            return {
                "result": str(result),
                "selected_experts": list(self._call_log.keys()),
                "expert_responses": dict(self._call_log),
            }
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise
