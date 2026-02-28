from typing import TypedDict, Annotated, List, Dict
import operator


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence"""
    return {**left, **right}


def merge_lists(left: list, right: list) -> list:
    """Merge two lists by concatenation"""
    return left + right


class MoEState(TypedDict):
    """Shared state across all agents in the MoE system"""
    query: str
    query_type: str
    expert_responses: Annotated[Dict[str, str], merge_dicts]
    selected_experts: List[str]
    final_answer: str
    reasoning_steps: Annotated[List[Dict], merge_lists]
    confidence_scores: Annotated[Dict[str, float], merge_dicts]
    metadata: Dict


def create_initial_state(query: str) -> MoEState:
    """Create initial state for a new query"""
    return {
        "query": query,
        "query_type": "",
        "expert_responses": {},
        "selected_experts": [],
        "final_answer": "",
        "reasoning_steps": [],
        "confidence_scores": {},
        "metadata": {}
    }