from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import time
import asyncio
import logging
from ..core.state import MoEState
from ..llm.providers import LLMProvider


logger = logging.getLogger(__name__)


# ======================================================================
# Shared mixin for logging / skip helpers
# ======================================================================

class _AgentMixin:
    """Shared helpers available to both sync and async agents."""

    name: str  # provided by subclass __init__

    def _log_step(self, action: str, details: Dict[str, Any] = None) -> Dict:
        return {
            "agent": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }

    def _should_skip(self, state: MoEState) -> bool:
        """Determine if this agent should skip execution"""
        return False


# ======================================================================
# Synchronous base (Orchestrator, legacy Router, experts in graph mode)
# ======================================================================

class BaseAgent(_AgentMixin, ABC):
    """Abstract base class for synchronous agents that need an LLM."""

    def __init__(self, name: str, llm_provider: LLMProvider, max_retries: int = 3, retry_delay: int = 1):
        self.name = name
        self.llm = llm_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def execute(self, state: MoEState) -> Dict[str, Any]:
        pass

    def invoke_with_retry(self, prompt: str) -> Any:
        """Invoke LLM with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.name}: LLM call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"{self.name}: Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        error_msg = f"{self.name}: All {self.max_retries} retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    async def ainvoke_with_retry(self, prompt: str) -> Any:
        """Asynchronously invoke LLM with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self.llm.ainvoke(prompt)
                return response
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.name}: Async LLM call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"{self.name}: Retrying async in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)

        error_msg = f"{self.name}: All {self.max_retries} async retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)


# ======================================================================
# Async-only base (CodeExecutionAgent, any future async-only nodes)
# ======================================================================

class AsyncBaseAgent(_AgentMixin, ABC):
    """Abstract base class for agents that only operate asynchronously.

    Unlike ``BaseAgent`` this does **not** require an LLM provider and
    declares ``aexecute`` as the abstract entry-point instead of ``execute``.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def aexecute(self, state: MoEState) -> Dict[str, Any]:
        pass