"""Readiness probe and degradation helpers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.degradation import collect_degraded_reasons
from src.api.settings import get_api_settings


def test_ready_endpoint_ok_in_dev(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    monkeypatch.delenv("API_USE_REDIS_CACHE", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_ready_503_production_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    monkeypatch.setattr(
        "src.api.routers.health.get_api_settings",
        lambda: replace(get_api_settings(), production=True, api_key=None),
    )
    res = client.get("/ready")
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["auth_configured"] is False


def test_ready_503_production_missing_media_root(monkeypatch) -> None:
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    monkeypatch.setattr(
        "src.api.routers.health.get_api_settings",
        lambda: replace(
            get_api_settings(),
            production=True,
            api_key="secret",
            media_root=None,
        ),
    )
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.json()["checks"]["media_root"] is False


def test_ready_503_when_redis_required_but_down(monkeypatch) -> None:
    monkeypatch.delenv("API_USE_REDIS_CACHE", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())

    class _DeadRedis:
        def ping(self) -> bool:
            raise ConnectionError("redis down")

    client.app.state.cache.redis_client = _DeadRedis()
    monkeypatch.setattr(
        "src.api.routers.health.get_api_settings",
        lambda: replace(get_api_settings(), use_redis_cache=True),
    )
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.json()["checks"]["redis"] is False


def test_collect_degraded_reasons_llm_skipped() -> None:
    report = SimpleNamespace(
        results={"degradation": {"reasons": ["diarization_unavailable"]}},
        llm={"llm_used": False, "skip_reason": "no_api_key"},
    )
    reasons = collect_degraded_reasons(report)
    assert "diarization_unavailable" in reasons
    assert any(r.startswith("llm:") for r in reasons)


def test_collect_degraded_reasons_extra_shapes() -> None:
    report = SimpleNamespace(
        results={
            "degradation": {"mode": "lite", "skipped": "asr"},
            "analyzer_routing": {"skipped": ["pii"]},
            "partial": {"incomplete": True},
        },
        llm={"used": False, "reason": "offline", "fallback": True},
    )
    reasons = collect_degraded_reasons(report)
    assert "lite" in reasons or "asr" in reasons
    assert "llm:offline" in reasons
    assert "llm:fallback" in reasons
    assert "analyzer_skipped:pii" in reasons
    assert "partial:incomplete" in reasons


def test_lifespan_starts_and_shuts_down(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    monkeypatch.setenv("API_MEDIA_ROOT", str(tmp_path))
    get_api_settings.cache_clear()
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200


def test_lifespan_production_without_redis(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.setenv("API_MEDIA_ROOT", str(tmp_path))
    monkeypatch.delenv("API_USE_REDIS_CACHE", raising=False)
    get_api_settings.cache_clear()
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_health_detail_includes_hub_backend(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    res = client.get("/status/health/detail")
    assert res.status_code == 200
    body = res.json()
    assert "transcription_events" in body
    assert "backend" in body["transcription_events"]
    assert "checks" in body
    assert body["status"] in ("ok", "degraded")
