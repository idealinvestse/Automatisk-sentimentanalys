"""OpenAPI contract snapshot — core paths and required request fields."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings

CORE_PATHS = (
    "/health",
    "/ready",
    "/analyze",
    "/upload",
    "/transcribe",
    "/calls",
    "/calls/{call_id}",
    "/llm/analysis-profiles",
    "/llm/analysis-profiles/{perspective_id}",
    "/llm/providers",
    "/analyze_pipeline",
    "/analyze_pipeline/partial",
    "/ws/transcription/ticket",
)

REQUIRED_BODY_FIELDS = {
    ("/analyze", "post"): {"texts"},
    ("/analyze_pipeline", "post"): {"segments"},
    ("/analyze_pipeline/partial", "post"): {"segments"},
    ("/transcribe", "post"): {"audio_path"},
    ("/calls/{call_id}", "put"): {"id"},
}


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


@pytest.fixture
def spec() -> dict:
    client = TestClient(create_app())
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_core_paths_present(spec: dict) -> None:
    paths = spec.get("paths") or {}
    missing = [path for path in CORE_PATHS if path not in paths]
    assert missing == [], f"OpenAPI missing paths: {missing}"


def test_openapi_required_request_fields(spec: dict) -> None:
    paths = spec["paths"]
    components = spec.get("components", {}).get("schemas", {})
    for (path, method), required in REQUIRED_BODY_FIELDS.items():
        op = paths[path][method]
        schema = op["requestBody"]["content"]["application/json"]["schema"]
        names = _schema_required(schema, components)
        assert required <= names, f"{method.upper()} {path} missing required {required - names}"


def _schema_required(schema: dict, components: dict) -> set[str]:
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        schema = components[name]
    return set(schema.get("required") or [])
