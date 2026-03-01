import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class Span:
    """Represents a single operation in the Orchestrator script trace (e.g., query_agent, memory_store)."""
    span_id: str
    span_type: str # 'agent', 'memory', 'flow'
    name: str
    start_time: float
    end_time: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict) # e.g., token usage, tokens/sec

    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0
        
    def to_dict(self) -> dict:
        return {
            "id": self.span_id,
            "type": self.span_type,
            "name": self.name,
            "durationMs": self.duration_ms(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "metrics": self.metrics,
            "parentId": self.parent_id
        }

class Tracer:
    """
    OpenTelemetry-like tracer that records agent interactions and memory accesses
    to construct the Visual DNA of a script execution.
    """
    def __init__(self):
        self.spans: List[Span] = []
        self._counter = 0
        
    def _generate_id(self) -> str:
        self._counter += 1
        return f"span_{int(time.time()*1000)}_{self._counter}"

    def start_span(self, name: str, span_type: str, inputs: Dict[str, Any], parent_id: Optional[str] = None) -> Span:
        span = Span(
            span_id=self._generate_id(),
            span_type=span_type,
            name=name,
            start_time=time.time(),
            inputs=inputs,
            parent_id=parent_id
        )
        self.spans.append(span)
        return span
        
    def end_span(self, span: Span, outputs: Dict[str, Any] = None, error: Optional[Exception] = None, metrics: Dict[str, Any] = None):
        span.end_time = time.time()
        if outputs:
            span.outputs = outputs
        if error:
            span.error = str(error)
        if metrics:
            span.metrics = metrics

    def get_trace(self) -> List[Dict]:
        return [span.to_dict() for span in self.spans]

# Global tracer per execution context will be injected or managed via contextvars.
# For simplicity, we can yield a new tracer per sandbox execution.
