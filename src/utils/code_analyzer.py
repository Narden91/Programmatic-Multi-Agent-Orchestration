"""
Static analysis of LLM-generated orchestration scripts.

Parses the generated code's AST and extracts the execution plan:
which expert calls are made, which are parallel (``asyncio.gather``),
and what the sequential / parallel structure looks like.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExpertCall:
    """A single expert tool invocation extracted from the AST."""

    expert_type: str  # e.g. "technical"
    function_name: str  # e.g. "query_technical_expert"
    line: int
    is_parallel: bool = False
    group_id: Optional[int] = None  # shared id for gather groups


@dataclass
class ExecutionPlan:
    """Structured execution plan extracted from generated code."""

    calls: List[ExpertCall] = field(default_factory=list)
    has_sequential: bool = False
    has_parallel: bool = False
    gather_groups: int = 0
    raw_code: str = ""

    @property
    def experts_used(self) -> List[str]:
        """Unique expert types in call-order."""
        return list(dict.fromkeys(c.expert_type for c in self.calls))

    @property
    def parallel_calls(self) -> List[ExpertCall]:
        return [c for c in self.calls if c.is_parallel]

    @property
    def sequential_calls(self) -> List[ExpertCall]:
        return [c for c in self.calls if not c.is_parallel]

    def to_dict(self) -> dict:
        """Serialisable summary for inclusion in ``MoEState``."""
        return {
            "experts_used": self.experts_used,
            "has_sequential": self.has_sequential,
            "has_parallel": self.has_parallel,
            "gather_groups": self.gather_groups,
            "calls": [
                {
                    "expert": c.expert_type,
                    "function": c.function_name,
                    "line": c.line,
                    "parallel": c.is_parallel,
                    "group": c.group_id,
                }
                for c in self.calls
            ],
        }


# ---- Public API ---------------------------------------------------------


def analyze_code(code: str) -> ExecutionPlan:
    """Parse orchestration *code* and return its :class:`ExecutionPlan`."""
    plan = ExecutionPlan(raw_code=code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return plan

    gather_id = 0
    # IDs of nodes that live inside an asyncio.gather() call
    _inside_gather: set[int] = set()

    # -- Pass 1: find all asyncio.gather() calls and collect their expert
    #    invocations as parallel calls.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_gather(node):
            gather_id += 1
            plan.has_parallel = True
            plan.gather_groups = gather_id
            for arg in node.args:
                # Mark every descendant so pass-2 skips them
                for desc in ast.walk(arg):
                    _inside_gather.add(id(desc))
                for ec in _find_expert_calls(arg):
                    ec.is_parallel = True
                    ec.group_id = gather_id
                    plan.calls.append(ec)

    # -- Pass 2: find standalone query_agent() calls NOT inside a gather
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and id(node) not in _inside_gather
            and _is_expert_call(node)
        ):
            expert_type = "unknown"
            if len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                expert_type = str(node.args[0].value)
            plan.has_sequential = True
            plan.calls.append(
                ExpertCall(
                    expert_type=expert_type,
                    function_name="query_agent",
                    line=getattr(node, "lineno", 0),
                    is_parallel=False,
                    group_id=None,
                )
            )

    return plan


# ---- Helpers ------------------------------------------------------------


def _is_gather(node: ast.Call) -> bool:
    """Return True if *node* is ``asyncio.gather(...)``."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "gather"
        and isinstance(func.value, ast.Name)
        and func.value.id == "asyncio"
    )


def _is_expert_call(node: ast.Call) -> bool:
    """Return True if *node* is ``query_agent(...)``."""
    func = node.func
    return (
        isinstance(func, ast.Name)
        and func.id == "query_agent"
    )


def _find_expert_calls(node: ast.AST) -> List[ExpertCall]:
    """Find all ``query_agent`` calls anywhere inside *node*."""
    calls: list[ExpertCall] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _is_expert_call(child):
            expert_type = "unknown"
            if len(child.args) > 0 and isinstance(child.args[0], ast.Constant):
                expert_type = str(child.args[0].value)
            calls.append(
                ExpertCall(
                    expert_type=expert_type,
                    function_name="query_agent",
                    line=getattr(child, "lineno", 0),
                )
            )
    return calls
