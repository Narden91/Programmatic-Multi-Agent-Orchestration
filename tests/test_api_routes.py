from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes as routes


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_init_exposes_supported_models_and_default_model():
    client = _build_client()

    response = client.get("/api/init")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == "llama-3.1-8b-instant"
    assert "llama-3.1-8b-instant" in payload["models"]
    assert "llama-3.1-70b-versatile" not in payload["models"]


def test_query_rejects_decommissioned_model_before_provider_call(monkeypatch):
    client = _build_client()

    response = client.post(
        "/api/query",
        json={"query": "hello", "model": "llama-3.1-70b-versatile"},
    )

    assert response.status_code == 400
    assert "decommissioned" in response.json()["detail"].lower()
    assert "llama-3.1-8b-instant" in response.json()["detail"]


def test_query_maps_provider_rate_limit_to_429(monkeypatch):
    class _Graph:
        async def ainvoke(self, state):
            raise Exception("Orchestrator: provider rate limit exceeded")

    class _Builder:
        def __init__(self, config):
            self.config = config

        def build(self):
            return _Graph()

    monkeypatch.setattr(routes, "MoEGraphBuilder", _Builder)
    monkeypatch.setattr(routes.MoEConfig, "validate", lambda self: True)

    client = _build_client()
    response = client.post(
        "/api/query",
        json={"query": "hello", "model": "llama-3.1-8b-instant"},
    )

    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()
    assert "llama-3.1-8b-instant" in response.json()["detail"]