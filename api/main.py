"""FastAPI backend for the Programmatic Multi-Agent Orchestration system."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Programmatic MoE Orchestration",
    description="Code-Driven Multi-Agent Orchestration API",
    version="0.5.0",
    lifespan=lifespan,
)

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
