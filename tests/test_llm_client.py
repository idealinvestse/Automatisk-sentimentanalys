"""Tests for the OpenRouter Mistral client (Task 3.1.1).

Covers:
- Lazy import guard and clear error when openai missing
- Structured chat with json_schema (mocked)
- Caching (in-memory + disk) + cost meta
- Retry behavior on rate limits / transient errors
- Privacy log emission (external data sent)
- Fallback to LLMError after retries
- Cache clear utility

These tests must never hit the real network. All OpenAI calls are patched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.errors import LLMError
from src.llm.openrouter_client import OpenRouterClient, _HAS_OPENAI


@pytest.fixture
def fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-1234567890abcdef")
    return "sk-test-1234567890abcdef"


def test_client_import_guard(monkeypatch):
    """Module imports cleanly; using it without openai raises helpful ImportError."""
    # Force the guard off (simulates `pip install` not done)
    import src.llm.openrouter_client as mod

    orig = mod._HAS_OPENAI
    try:
        mod._HAS_OPENAI = False
        mod.OpenAI = None  # type: ignore
        client = OpenRouterClient(api_key="dummy")
        with pytest.raises(ImportError, match="openai package is required"):
            client.structured_chat(messages=[{"role": "user", "content": "hi"}], task_name="test")
    finally:
        mod._HAS_OPENAI = orig


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed in this env; guard tested above")
def test_structured_chat_success_and_meta(fake_api_key, tmp_path, monkeypatch):
    """Happy path: strict schema -> parsed dict + rich meta. No real net call."""
    client = OpenRouterClient(
        api_key=fake_api_key,
        cache_dir=tmp_path / "llmcache",
        enable_cache=True,
    )

    # Build a fake completion object that the SDK would return
    fake_completion = MagicMock()
    fake_completion.model = "mistralai/mistral-medium-3.5"
    fake_completion.id = "gen-xyz123"
    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 1240
    fake_usage.completion_tokens = 380
    fake_usage.model_dump.return_value = {"prompt_tokens": 1240, "completion_tokens": 380, "total_tokens": 1620}
    fake_completion.usage = fake_usage
    fake_choice = MagicMock()
    fake_choice.message.content = json.dumps(
        {
            "trajectory": {"summary": "Kundens frustration ökade efter fakturafrågan."},
            "actionable_summary": {"problem": "Faktureringsfel", "recommendations_for_qa": ["Coacha agent på empathy"]},
        }
    )
    fake_completion.choices = [fake_choice]

    with patch.object(client, "_ensure_openai") as mock_ensure:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_completion
        mock_ensure.return_value = mock_client

        schema = {
            "type": "object",
            "properties": {
                "trajectory": {"type": "object"},
                "actionable_summary": {"type": "object"},
            },
            "required": ["trajectory", "actionable_summary"],
            "additionalProperties": False,
        }

        result, meta = client.structured_chat(
            messages=[
                {"role": "system", "content": "Du är expert på svensk callcenter-analys."},
                {"role": "user", "content": "SPEAKER_0 (agent): ... \nSPEAKER_1 (kund): ..."},
            ],
            json_schema=schema,
            task_name="actionable_summary",
            transcript_hash="abc123def456",
            temperature=0.1,
        )

    assert result["actionable_summary"]["problem"] == "Faktureringsfel"
    assert meta["model"] == "mistralai/mistral-medium-3.5"
    assert meta["cached"] is False
    assert meta["cost_usd"] is not None and meta["cost_usd"] > 0
    assert meta["task"] == "actionable_summary"
    assert mock_client.chat.completions.create.call_count == 1

    # Verify response_format was built correctly (strict json_schema)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    rf = call_kwargs.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "actionable_summary"


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_caching_works_and_is_free(fake_api_key, tmp_path):
    """Second identical call must hit cache and report cost=0 + cached=True."""
    client = OpenRouterClient(api_key=fake_api_key, cache_dir=tmp_path / "c2", enable_cache=True)

    fake_completion = MagicMock()
    fake_completion.model = client.DEFAULT_MODEL
    fake_completion.id = "gen-1"
    u = MagicMock(prompt_tokens=800, completion_tokens=200)
    u.model_dump.return_value = {"prompt_tokens": 800, "completion_tokens": 200}
    fake_completion.usage = u
    fake_choice = MagicMock()
    fake_choice.message.content = '{"ok": true}'
    fake_completion.choices = [fake_choice]

    with patch.object(client, "_ensure_openai") as m_ensure:
        m_client = MagicMock()
        m_client.chat.completions.create.return_value = fake_completion
        m_ensure.return_value = m_client

        msgs = [{"role": "user", "content": "Test transcript for cache key"}]
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        r1, m1 = client.structured_chat(msgs, json_schema=schema, task_name="test_cache", transcript_hash="h1")
        r2, m2 = client.structured_chat(msgs, json_schema=schema, task_name="test_cache", transcript_hash="h1")

    assert r1 == r2
    assert m1["cached"] is False
    assert m2["cached"] is True
    assert m2["cost_usd"] == 0.0 or m2.get("cost_usd", 0) == 0
    assert m_client.chat.completions.create.call_count == 1  # only first hit real


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_retries_then_llmerror_on_rate_limit(fake_api_key, tmp_path):
    """RateLimitError should be retried; after max_retries we get LLMError."""
    client = OpenRouterClient(
        api_key=fake_api_key, cache_dir=tmp_path / "c3", max_retries=2, enable_cache=False
    )

    from openai import RateLimitError as RealRateLimit  # type: ignore

    with patch.object(client, "_ensure_openai") as m_ensure:
        m_client = MagicMock()
        # Always raise rate limit
        m_client.chat.completions.create.side_effect = RealRateLimit(
            "rate limited", response=MagicMock(), body={}
        )
        m_ensure.return_value = m_client

        with pytest.raises(LLMError, match="after 2 attempts"):
            client.structured_chat(
                messages=[{"role": "user", "content": "x"}],
                task_name="retry_test",
            )

    assert m_client.chat.completions.create.call_count == 2


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_privacy_log_emitted(fake_api_key, tmp_path, caplog):
    """Every real (non-cached) call must emit the GDPR/egress warning log."""
    client = OpenRouterClient(api_key=fake_api_key, cache_dir=tmp_path / "c4", enable_cache=False)

    fake_c = MagicMock()
    fake_c.model = "mistralai/mistral-medium-3.5"
    fake_c.id = "g1"
    u = MagicMock()
    u.prompt_tokens = 10
    u.completion_tokens = 5
    u.model_dump.return_value = {}
    fake_c.usage = u
    ch = MagicMock()
    ch.message.content = '{"a":1}'
    fake_c.choices = [ch]

    caplog.set_level("INFO")

    with patch.object(client, "_ensure_openai") as me:
        mc = MagicMock()
        mc.chat.completions.create.return_value = fake_c
        me.return_value = mc

        client.structured_chat(
            messages=[{"role": "user", "content": "Känslig kundinfo här"}],
            task_name="privacy_check",
            transcript_hash="priv1",
        )

    logs = [r.message for r in caplog.records]
    assert any("EXTERNAL LLM CALL (OpenRouter/Mistral)" in m for m in logs)
    assert any("third-party service" in m for m in logs)


def test_clear_cache(fake_api_key, tmp_path):
    client = OpenRouterClient(api_key=fake_api_key, cache_dir=tmp_path / "clr", enable_cache=True)
    # Pre-populate fake cache entries
    (client.cache_dir).mkdir(parents=True, exist_ok=True)
    (client.cache_dir / "a.json").write_text('{"result": {}, "meta": {}}', encoding="utf-8")
    (client.cache_dir / "b.json").write_text('{"result": {}, "meta": {}}', encoding="utf-8")

    removed = client.clear_cache()
    assert removed == 2
    assert not any(client.cache_dir.glob("*.json"))
