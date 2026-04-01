"""API routes for the MoE orchestration system."""

import os
import asyncio

from fastapi import APIRouter, HTTPException

from api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
)
from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

router = APIRouter()

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
]


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.5.0")


@router.get("/init")
async def get_init():
    """Combined config+models endpoint — one request instead of two."""
    has_env_key = bool(
        os.getenv("GROQ_API_KEY", "").strip() or 
        os.getenv("OPENAI_API_KEY", "").strip() or 
        os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    return {
        "has_env_api_key": has_env_key,
        "version": "0.5.0",
        "models": AVAILABLE_MODELS,
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    api_key_str = (req.api_key or "").strip()

    config = MoEConfig()
    
    # If the user provides a key in the request, try to guess the provider
    if api_key_str:
        if api_key_str.startswith("sk-ant"):
            config.anthropic_api_key = SecretStr(api_key_str)
        elif api_key_str.startswith("sk-"):
            config.openai_api_key = SecretStr(api_key_str)
        else:
            config.groq_api_key = SecretStr(api_key_str)

    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    try:
        config.orchestrator_config.model_name = req.model
        for ec in config.expert_configs.values():
            ec.llm_config.model_name = req.model
        config.validate()

        builder = MoEGraphBuilder(config)
        graph = builder.build()
        initial_state = create_initial_state(req.query)

        timeout = max(int(config.request_timeout), 1)
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state), timeout=timeout
        )

        final_answer = result.get("final_answer", "")
        error = result.get("code_execution_error", "")
        iters = result.get("code_execution_iterations", 0)

        if not final_answer and error:
            final_answer = f"**Orchestration Failed after {iters} attempts**\n\n```text\n{error}\n```"

        return QueryResponse(
            final_answer=final_answer,
            generated_code=result.get("generated_code", ""),
            code_execution_error=error,
            code_execution_iterations=iters,
            selected_experts=result.get("selected_experts", []),
            expert_responses=result.get("expert_responses", {}),
            execution_plan=result.get("execution_plan", {}),
            token_usage=result.get("token_usage", {}),
            trace_dna=result.get("trace_dna", []),
            sandbox_output=result.get("sandbox_output", ""),
            sandbox_security=(result.get("metadata", {}) or {}).get("sandbox_security", {}),
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Try a shorter query or increase REQUEST_TIMEOUT.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
