"""Performance and token-usage metrics for the MoE system."""

import time
import asyncio
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


# ======================================================================
# Token tracking
# ======================================================================

@dataclass
class TokenRecord:
    """Single LLM call token record."""

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Approximate costs per 1 M tokens (USD).  Override at runtime via
# ``TokenTracker.set_pricing``.
_DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}


class TokenTracker:
    """Accumulates token-usage records across an entire request lifecycle."""

    def __init__(self) -> None:
        self._records: List[TokenRecord] = []
        self._pricing: Dict[str, Dict[str, float]] = dict(_DEFAULT_PRICING)

    # ---- Recording ---------------------------------------------------

    def record(
        self, agent: str, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        self._records.append(
            TokenRecord(
                agent=agent,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def record_from_response(self, agent: str, model: str, response: Any) -> None:
        """Extract token usage from a LangChain ``AIMessage.response_metadata``.

        Silently does nothing when the response lacks proper metadata
        (e.g. when using mocks in tests).
        """
        meta = getattr(response, "response_metadata", None)
        if not isinstance(meta, dict):
            return
        usage = meta.get("token_usage") or meta.get("usage")
        if not isinstance(usage, dict):
            return
        self.record(
            agent=agent,
            model=model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    # ---- Pricing -----------------------------------------------------

    def set_pricing(
        self, model: str, input_per_m: float, output_per_m: float
    ) -> None:
        self._pricing[model] = {"input": input_per_m, "output": output_per_m}

    def _cost(self, rec: TokenRecord) -> float:
        p = self._pricing.get(rec.model, {"input": 0.0, "output": 0.0})
        return (rec.input_tokens * p["input"] + rec.output_tokens * p["output"]) / 1_000_000

    # ---- Queries -----------------------------------------------------

    @property
    def records(self) -> List[TokenRecord]:
        return list(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(self._cost(r) for r in self._records)

    def summary(self) -> Dict[str, Any]:
        by_agent: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            a = by_agent.setdefault(
                r.agent, {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
            )
            a["input"] += r.input_tokens
            a["output"] += r.output_tokens
            a["cost"] += self._cost(r)
            a["calls"] += 1
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 6),
            "by_agent": by_agent,
        }

    def reset(self) -> None:
        self._records.clear()


# Module-level singleton ---------------------------------------------------

_token_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """Return the global ``TokenTracker`` (lazily created)."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker


def reset_token_tracker() -> TokenTracker:
    """Reset and return a fresh global ``TokenTracker``."""
    global _token_tracker
    _token_tracker = TokenTracker()
    return _token_tracker


# ======================================================================
# Performance timing (unchanged from v0.1)
# ======================================================================

class PerformanceMetrics:
    """Track execution-time metrics for the MoE system."""

    def __init__(self):
        self.metrics = {}

    def record_execution_time(self, agent_name: str, duration: float):
        if agent_name not in self.metrics:
            self.metrics[agent_name] = {
                "executions": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
            }
        m = self.metrics[agent_name]
        m["executions"] += 1
        m["total_time"] += duration
        m["min_time"] = min(m["min_time"], duration)
        m["max_time"] = max(m["max_time"], duration)

    def get_average_time(self, agent_name: str) -> float:
        if agent_name not in self.metrics:
            return 0.0
        m = self.metrics[agent_name]
        return m["total_time"] / m["executions"]

    def get_summary(self) -> dict:
        summary = {}
        for agent_name, m in self.metrics.items():
            summary[agent_name] = {
                "executions": m["executions"],
                "avg_time": m["total_time"] / m["executions"],
                "min_time": m["min_time"],
                "max_time": m["max_time"],
            }
        return summary

    def reset(self):
        self.metrics = {}


def measure_time(metrics: PerformanceMetrics, agent_name: str):
    """Decorator to measure execution time of agent methods."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_execution_time(agent_name, duration)
                return result

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_execution_time(agent_name, duration)
                return result

            return sync_wrapper

    return decorator