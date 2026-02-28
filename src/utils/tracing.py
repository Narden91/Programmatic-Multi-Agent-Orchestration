"""Real-time streaming trace for the MoE pipeline.

Components emit :class:`TraceEvent` objects via a shared :class:`Tracer`.
Any number of consumers can subscribe to receive events as they happen.

Usage::

    from src.utils.tracing import get_tracer, TraceEvent

    tracer = get_tracer()

    # consumer (e.g. UI/CLI)
    async for event in tracer.subscribe():
        print(event)

    # producer (inside pipeline code)
    await tracer.emit(TraceEvent(
        kind="orchestrator.code_generated",
        agent="Orchestrator",
        data={"code_length": 320},
    ))
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


# ======================================================================
# Event model
# ======================================================================

class TraceKind(str, Enum):
    """Well-known event kinds emitted by the pipeline."""

    # Orchestrator
    ORCHESTRATOR_START = "orchestrator.start"
    ORCHESTRATOR_CODE_GENERATED = "orchestrator.code_generated"
    ORCHESTRATOR_RETRY = "orchestrator.retry"

    # Sandbox / Code executor
    SANDBOX_START = "sandbox.start"
    SANDBOX_SUCCESS = "sandbox.success"
    SANDBOX_ERROR = "sandbox.error"

    # Expert calls (inside sandbox)
    EXPERT_CALL_START = "expert.call_start"
    EXPERT_CALL_END = "expert.call_end"

    # Pipeline-level
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"

    # Generic / user-defined
    CUSTOM = "custom"


@dataclass(frozen=True)
class TraceEvent:
    """A single trace event emitted during pipeline execution."""

    kind: str
    agent: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Convenience helpers ---------------------------------------------------

    @property
    def elapsed_label(self) -> str:
        """Human-readable time since epoch (useful for relative diffs)."""
        return f"{self.timestamp:.3f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "agent": self.agent,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ======================================================================
# Tracer (fan-out broadcaster)
# ======================================================================

class Tracer:
    """Async broadcast channel for :class:`TraceEvent` objects.

    *  Producers call :meth:`emit` to send an event.
    *  Consumers call :meth:`subscribe` to receive an async iterator.
    *  Multiple concurrent subscribers are supported.
    """

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue[TraceEvent | None]] = []
        self._history: List[TraceEvent] = []
        self._closed = False

    # ---- Producer API ------------------------------------------------

    async def emit(self, event: TraceEvent) -> None:
        """Broadcast *event* to every active subscriber."""
        self._history.append(event)
        for q in self._subscribers:
            await q.put(event)

    def emit_sync(self, event: TraceEvent) -> None:
        """Non-async emit — schedules the event on the running loop.

        Safe to call from synchronous code that is executing inside an
        event-loop thread (e.g. a LangGraph node that isn't *async def*).
        """
        self._history.append(event)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # drop if consumer is too slow

    # ---- Consumer API ------------------------------------------------

    async def subscribe(self) -> AsyncIterator[TraceEvent]:
        """Yield events as they arrive.  Stops when the tracer is closed."""
        q: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
        self._subscribers.append(q)
        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            self._subscribers.remove(q)

    # ---- Lifecycle ---------------------------------------------------

    async def close(self) -> None:
        """Signal all subscribers to stop."""
        self._closed = True
        for q in self._subscribers:
            await q.put(None)

    def reset(self) -> None:
        """Clear history (does **not** close subscribers)."""
        self._history.clear()

    @property
    def history(self) -> List[TraceEvent]:
        """Return a copy of all events emitted so far."""
        return list(self._history)

    @property
    def closed(self) -> bool:
        return self._closed


# ======================================================================
# Module-level singleton
# ======================================================================

_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Return the global :class:`Tracer` singleton (lazily created)."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def reset_tracer() -> None:
    """Replace the global tracer with a fresh instance."""
    global _tracer
    _tracer = Tracer()
