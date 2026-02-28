from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import time
import asyncio
import logging
from ..core.state import MoEState
from ..llm.providers import LLMProvider


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, llm_provider: LLMProvider, max_retries: int = 3, retry_delay: int = 1):
        """Initialize base agent"""
        self.name = name
        self.llm = llm_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @abstractmethod
    def execute(self, state: MoEState) -> Dict[str, Any]:
        """Execute agent logic"""
        pass
    
    def invoke_with_retry(self, prompt: str) -> Any:
        """
        Invoke LLM with retry logic and exponential backoff.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retries are exhausted
        """
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
                
                # Don't sleep on last attempt
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"{self.name}: Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
        
        # All retries exhausted
        error_msg = f"{self.name}: All {self.max_retries} retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    async def ainvoke_with_retry(self, prompt: str) -> Any:
        """
        Asynchronously invoke LLM with retry logic and exponential backoff.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retries are exhausted
        """
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
                
                # Don't sleep on last attempt
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"{self.name}: Retrying async in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
        
        # All retries exhausted
        error_msg = f"{self.name}: All {self.max_retries} async retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _log_step(self, action: str, details: Dict[str, Any] = None) -> Dict:
        """Create a reasoning step log entry"""
        return {
            "agent": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
    
    def _should_skip(self, state: MoEState) -> bool:
        """Determine if this agent should skip execution"""
        return False