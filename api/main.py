"""FastAPI backend for the Programmatic Multi-Agent Orchestration system."""

from contextlib import asynccontextmanager
from pathlib import Path

import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from src.utils.embeddings import get_embedding_model

logger = logging.getLogger("moe.api")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# override=True ensures .env values always win (even if the var already
# exists as an empty string in the process environment).
DOTENV_LOADED = load_dotenv(dotenv_path=_ENV_FILE, override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    key_present = bool(
        os.getenv("GROQ_API_KEY", "").strip() or 
        os.getenv("OPENAI_API_KEY", "").strip() or 
        os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    logger.info("dotenv parse result: %s", DOTENV_LOADED)
    logger.info("dotenv loaded from %s  (file exists: %s)", _ENV_FILE, _ENV_FILE.exists())
    logger.info("API_KEY detected: %s", key_present)
    if not key_present:
        logger.warning(
            "No API_KEY found. Create a .env file at %s with GROQ_API_KEY=<key>",
            _ENV_FILE,
        )
    
    # Pre-load embedding model to prevent 504 timeouts on first request
    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model loaded.")
    yield


app = FastAPI(
    title="Programmatic MoE Orchestration",
    description="Code-Driven Multi-Agent Orchestration API",
    version="0.5.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Add Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

app.add_middleware(

    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import router  # noqa: E402

app.include_router(router, prefix="/api")


_FAVICON = Path(__file__).resolve().parent.parent / "frontend" / "public" / "favicon.svg"


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    if _FAVICON.exists():
        return FileResponse(_FAVICON, media_type="image/svg+xml")
    return {"detail": "not found"}


@app.get("/")
async def root():
    return {
        "message": "Programmatic MoE Orchestration API",
        "docs": "/docs",
        "frontend": "http://localhost:5173",
        "health": "/api/health",
    }
