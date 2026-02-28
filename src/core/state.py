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
    generated_code: str
    code_execution_error: str
    code_execution_iterations: int
    token_usage: Dict
    execution_plan: Dict
    conversation_context: str  # formatted multi-turn history


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
        "metadata": {},
        "generated_code": "",
        "code_execution_error": "",
        "code_execution_iterations": 0,
        "token_usage": {},
        "execution_plan": {},
        "conversation_context": "",
    }