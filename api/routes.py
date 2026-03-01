"""API routes for the MoE orchestration system."""

import os
import asyncio

from fastapi import APIRouter, HTTPException

from api.schemas import (
    QueryRequest,
    QueryResponse,
    ModelsResponse,
    ConfigResponse,
    HealthResponse,
)
from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

router = APIRouter()

AVAILABLE_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
]


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.5.0")


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    has_env_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    return ConfigResponse(
        has_env_api_key=has_env_key,
        version="0.5.0",
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    return ModelsResponse(models=AVAILABLE_MODELS)


@router.get("/init")
async def get_init():
    """Combined config+models endpoint — one request instead of two."""
    has_env_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    return {
        "has_env_api_key": has_env_key,
        "version": "0.5.0",
        "models": AVAILABLE_MODELS,
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    api_key = (req.api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key provided. Set GROQ_API_KEY environment variable or pass api_key in the request.",
        )

    try:
        config = MoEConfig(groq_api_key=SecretStr(api_key))
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
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Try a shorter query or increase REQUEST_TIMEOUT.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
