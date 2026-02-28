import asyncio
import logging
from typing import Any, Dict
from ..agents.tools import (
    query_technical_expert,
    query_analytical_expert,
    query_creative_expert,
    query_general_expert
)

logger = logging.getLogger(__name__)

class CodeSandbox:
    """A minimal sandbox to execute the LLM-generated async orchestration script."""
    
    def __init__(self):
        self.allowed_globals = {
            "asyncio": asyncio,
            "query_technical_expert": query_technical_expert,
            "query_analytical_expert": query_analytical_expert,
            "query_creative_expert": query_creative_expert,
            "query_general_expert": query_general_expert,
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
        }
    
    async def execute(self, code: str) -> Any:
        """
        Executes the provided async python code.
        The code is expected to define an `async def orchestrate():` function.
        """
        local_vars: Dict[str, Any] = {}
        
        try:
            # Compile and execute the definition of the functions
            exec(code, self.allowed_globals, local_vars)
            
            if 'orchestrate' not in local_vars:
                raise ValueError("The generated script must define an 'async def orchestrate():' function.")
            
            orchestrate_func = local_vars['orchestrate']
            
            if not asyncio.iscoroutinefunction(orchestrate_func):
                raise ValueError("The 'orchestrate' function must be an async function.")
            
            # Run the extracted async function
            result = await orchestrate_func()
            return str(result)
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise e
