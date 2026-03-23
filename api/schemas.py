"""Pydantic models for the MoE API."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    query: str
    api_key: Optional[str] = None
    model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"


class QueryResponse(BaseModel):
    final_answer: str
    generated_code: str
    code_execution_error: str
    code_execution_iterations: int
    selected_experts: List[str]
    expert_responses: Dict[str, str]
    execution_plan: Dict[str, Any]
    token_usage: Dict[str, Any]
    trace_dna: List[Dict[str, Any]] = []
    sandbox_output: str = ""
    sandbox_security: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    version: str
