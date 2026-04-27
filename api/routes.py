"""API routes for the MoE orchestration system."""

import os
import asyncio
import logging
import traceback

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("moe.routes")

from api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
)
from src.core.config import (
    ANTHROPIC_CHAT_MODELS,
    DEFAULT_LLM_MODEL,
    DEPRECATED_MODEL_REPLACEMENTS,
    GROQ_CHAT_MODELS,
    MoEConfig,
    OPENAI_CHAT_MODELS,
    SecretStr,
)
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

router = APIRouter()

AVAILABLE_MODELS = [
    *GROQ_CHAT_MODELS,
    *OPENAI_CHAT_MODELS,
    *ANTHROPIC_CHAT_MODELS,
]


def _validate_requested_model(model_name: str) -> None:
    return None


def _resolve_requested_model(model_name: str) -> str:
    replacement = DEPRECATED_MODEL_REPLACEMENTS.get(model_name)
    if replacement:
        logger.warning(
            "Requested model `%s` is deprecated; remapping to `%s`.",
            model_name,
            replacement,
        )
        return replacement
    return model_name


def _map_query_failure(error: Exception, model_name: str) -> HTTPException:
    text = str(error).strip()
    lowered = text.lower()

    if "request too large" in lowered or "error code: 413" in lowered:
        return HTTPException(
            status_code=413,
            detail=(
                f"Request too large for `{model_name}`. "
                "The generated prompt exceeded the provider request budget for a single call. "
                "Retry with a shorter query or reduced retrieved context."
            ),
        )

    if "rate_limit_exceeded" in lowered or "provider rate limit exceeded" in lowered or "rate limit" in lowered:
        return HTTPException(
            status_code=429,
            detail=(
                f"Provider rate limit exceeded for `{model_name}`. "
                f"Retry after the provider cooldown or switch to a smaller supported model like `{DEFAULT_LLM_MODEL}`."
            ),
        )

    if "model_decommissioned" in lowered or "decommissioned" in lowered:
        replacement = DEPRECATED_MODEL_REPLACEMENTS.get(model_name, DEFAULT_LLM_MODEL)
        return HTTPException(
            status_code=400,
            detail=(
                f"The model `{model_name}` is no longer supported by the provider. "
                f"Try `{replacement}` instead."
            ),
        )

    if "invalid_request_error" in lowered:
        return HTTPException(status_code=400, detail=text)

    return HTTPException(status_code=500, detail=text or f"{type(error).__name__} (no message)")


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.5.0")


@router.get("/init")
async def get_init():
    """Combined config+models endpoint — one request instead of two."""
    config = MoEConfig()
    has_env_key = bool(
        os.getenv("GROQ_API_KEY", "").strip() or 
        os.getenv("OPENAI_API_KEY", "").strip() or 
        os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    return {
        "has_env_api_key": has_env_key,
        "version": "0.5.0",
        "default_model": config.orchestrator_config.model_name,
        "models": AVAILABLE_MODELS,
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    api_key_str = (req.api_key or "").strip()
    requested_model = _resolve_requested_model((req.model or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL)

    _validate_requested_model(requested_model)

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
        config.orchestrator_config.model_name = requested_model
        for ec in config.expert_configs.values():
            ec.llm_config.model_name = requested_model
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
    except HTTPException:
        raise
    except Exception as e:
        http_error = _map_query_failure(e, requested_model)
        if http_error.status_code >= 500:
            logger.error("Query failed: %s\n%s", e, traceback.format_exc())
        else:
            logger.warning("Query failed: %s", http_error.detail)
        raise http_error
