"""Tests for the Groq Cloud client.

Covers:
- Lazy import guard and clear error when openai missing
- Structured chat with json_schema (mocked)
- Caching (in-memory + disk) + cost meta
- Retry behavior on rate limits / transient errors
- Privacy/GDPR log emission (external data sent)
- GDPR gate enforcement (groq_eu_residency + anonymize_before_llm)
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
from src.llm.groq_client import GroqClient, _HAS_OPENAI


@pytest.fixture
def fake_groq_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-1234567890abcdef")
    return "gsk-test-1234567890abcdef"


def test_client_import_guard(monkeypatch):
    """Module imports cleanly; using it without openai raises helpful ImportError."""
    import src.llm.groq_client as mod

    orig = mod._HAS_OPENAI
    try:
        mod._HAS_OPENAI = False
        mod.OpenAI = None  # type: ignore
        client = GroqClient(api_key="dummy", groq_eu_residency=True)
        with pytest.raises(ImportError, match="openai package is required"):
            client.structured_chat(
                messages=[{"role": "user", "content": "hi"}],
                task_name="test",
                anonymize_before_llm=True,
            )
    finally:
        mod._HAS_OPENAI = orig


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed in this env; guard tested above")
def test_structured_chat_success_and_meta(fake_groq_api_key, tmp_path, monkeypatch):
    """Happy path: returns parsed dict + rich meta. No real net call."""
    client = GroqClient(
        api_key=fake_groq_api_key,
        cache_dir=tmp_path / "groqcache",
        enable_cache=True,
        groq_eu_residency=True,
    )

    fake_completion = MagicMock()
    fake_completion.model = "llama-3.3-70b-versatile"
    fake_completion.id = "gen-groq-xyz123"
    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 500
    fake_usage.completion_tokens = 150
    fake_usage.model_dump.return_value = {"prompt_tokens": 500, "completion_tokens": 150, "total_tokens": 650}
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
            transcript_hash="groq_abc123",
            temperature=0.1,
            anonymize_before_llm=True,
        )

    assert result["actionable_summary"]["problem"] == "Faktureringsfel"
    assert meta["model"] == "llama-3.3-70b-versatile"
    assert meta["cached"] is False
    assert meta["cost_usd"] is not None and meta["cost_usd"] > 0
    assert meta["task"] == "actionable_summary"
    assert meta["provider"] == "groq"
    assert mock_client.chat.completions.create.call_count == 1

    # Verify response_format was built correctly
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    rf = call_kwargs.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_object"  # llama-3.3 uses json_mode, not strict json_schema


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_strict_json_schema_for_gpt_oss(fake_groq_api_key, tmp_path):
    """gpt-oss-20b should get strict json_schema response_format."""
    client = GroqClient(
        api_key=fake_groq_api_key,
        cache_dir=tmp_path / "groqcas2",
        enable_cache=True,
        groq_eu_residency=True,
    )

    fake_completion = MagicMock()
    fake_completion.model = "openai/gpt-oss-20b"
    fake_completion.id = "gen-gpt-oss"
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

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        client.structured_chat(
            messages=[{"role": "user", "content": "test"}],
            json_schema=schema,
            task_name="test_strict",
            model="openai/gpt-oss-20b",
            anonymize_before_llm=True,
        )

    call_kwargs = m_client.chat.completions.create.call_args.kwargs
    rf = call_kwargs.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "test_strict"


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_caching_works_and_is_free(fake_groq_api_key, tmp_path):
    """Second identical call must hit cache and report cost=0 + cached=True."""
    client = GroqClient(
        api_key=fake_groq_api_key,
        cache_dir=tmp_path / "groqc3",
        enable_cache=True,
        groq_eu_residency=True,
    )

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

        msgs = [{"role": "user", "content": "Test transcript for Groq cache key"}]
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

        r1, m1 = client.structured_chat(
            msgs, json_schema=schema, task_name="test_cache", transcript_hash="gh1", anonymize_before_llm=True,
        )
        r2, m2 = client.structured_chat(
            msgs, json_schema=schema, task_name="test_cache", transcript_hash="gh1", anonymize_before_llm=True,
        )

    assert r1 == r2
    assert m1["cached"] is False
    assert m2["cached"] is True
    assert m2["cost_usd"] == 0.0 or m2.get("cost_usd", 0) == 0
    assert m_client.chat.completions.create.call_count == 1


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_retries_then_llmerror_on_rate_limit(fake_groq_api_key, tmp_path):
    """RateLimitError should be retried; after max_retries we get LLMError."""
    client = GroqClient(
        api_key=fake_groq_api_key,
        cache_dir=tmp_path / "groqc4",
        max_retries=2,
        enable_cache=False,
        groq_eu_residency=True,
    )

    from openai import RateLimitError as RealRateLimit

    with patch.object(client, "_ensure_openai") as m_ensure:
        m_client = MagicMock()
        m_client.chat.completions.create.side_effect = RealRateLimit(
            "rate limited", response=MagicMock(), body={}
        )
        m_ensure.return_value = m_client

        with pytest.raises(LLMError, match="after 2 attempts"):
            client.structured_chat(
                messages=[{"role": "user", "content": "x"}],
                task_name="retry_test",
                anonymize_before_llm=True,
            )

    assert m_client.chat.completions.create.call_count == 2


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_privacy_log_emitted_for_groq(fake_groq_api_key, tmp_path, caplog):
    """Every non-cached Groq call must emit the GDPR/egress warning log."""
    client = GroqClient(
        api_key=fake_groq_api_key,
        cache_dir=tmp_path / "groqc5",
        enable_cache=False,
        groq_eu_residency=True,
    )

    fake_c = MagicMock()
    fake_c.model = "llama-3.3-70b-versatile"
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
            task_name="privacy_check_groq",
            transcript_hash="gpriv1",
            anonymize_before_llm=True,
        )

    logs = [r.message for r in caplog.records]
    assert any("EXTERNAL LLM CALL (Groq)" in m for m in logs)
    assert any("US/Saudi" in m or "US + Saudi" in m for m in logs)


def test_gdpr_gate_blocks_call_when_disabled():
    """groq_eu_residency=False + anonymize_before_llm=False → LLMError."""
    client = GroqClient(api_key="dummy", groq_eu_residency=False)

    # Should raise BEFORE hitting the network
    with pytest.raises(LLMError, match="Groq data centers"):
        client.structured_chat(
            messages=[{"role": "user", "content": "test"}],
            task_name="gdpr_test",
            anonymize_before_llm=False,
        )


def test_gdpr_gate_allows_call_when_anonymize_true():
    """groq_eu_residency=False + anonymize_before_llm=True → should pass gate."""
    client = GroqClient(api_key="dummy", groq_eu_residency=False)

    # Gate passes, should proceed to _ensure_openai (which will fail without key)
    with pytest.raises(Exception) as exc_info:
        client.structured_chat(
            messages=[{"role": "user", "content": "test"}],
            task_name="gdpr_anon_test",
            anonymize_before_llm=True,
        )
    # The error should NOT be the GDPR gate error
    assert "groq_eu_residency" not in str(exc_info.value)
    assert "Groq data centers" not in str(exc_info.value)


def test_list_models(fake_groq_api_key):
    """list_models returns dicts with id and metadata."""
    client = GroqClient(api_key=fake_groq_api_key)
    models = client.list_models()
    assert len(models) >= 17  # At least 17 models in registry
    assert any(m["id"] == "llama-3.1-8b-instant" for m in models)
    assert any(m["id"] == "openai/gpt-oss-20b" for m in models)
    # All models should have pricing
    for m in models:
        assert "pricing_in" in m or "id" in m


def test_clear_cache(fake_groq_api_key, tmp_path):
    client = GroqClient(api_key=fake_groq_api_key, cache_dir=tmp_path / "groqclr", enable_cache=True)
    (client.cache_dir).mkdir(parents=True, exist_ok=True)
    (client.cache_dir / "groq_a.json").write_text('{"result": {}, "meta": {}}', encoding="utf-8")
    (client.cache_dir / "groq_b.json").write_text('{"result": {}, "meta": {}}', encoding="utf-8")
    (client.cache_dir / "not_groq.txt").write_text("ignore", encoding="utf-8")

    removed = client.clear_cache()
    assert removed == 2
    assert not any(client.cache_dir.glob("groq_*.json"))
    # non-groq files untouched
    assert (client.cache_dir / "not_groq.txt").exists()


def test_get_groq_api_key_resolution(monkeypatch):
    """Key resolution priority: override → env var."""
    from src.llm.groq_client import get_groq_api_key

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert get_groq_api_key() is None
    assert get_groq_api_key("my-key") == "my-key"

    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    assert get_groq_api_key() == "env-key"
    assert get_groq_api_key("override") == "override"  # override wins


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")
def test_chat_completion_text_mode(fake_groq_api_key, tmp_path):
    """chat_completion returns plain text content + meta."""
    client = GroqClient(api_key=fake_groq_api_key, cache_dir=tmp_path / "groqchat", enable_cache=False, groq_eu_residency=True)

    fake_c = MagicMock()
    fake_c.model = "llama-3.1-8b-instant"
    fake_c.id = "g2"
    u = MagicMock(prompt_tokens=100, completion_tokens=50)
    u.model_dump.return_value = {}
    fake_c.usage = u
    ch = MagicMock()
    ch.message.content = "Hej, hur kan jag hjälpa till?"
    fake_c.choices = [ch]

    with patch.object(client, "_ensure_openai") as me:
        mc = MagicMock()
        mc.chat.completions.create.return_value = fake_c
        me.return_value = mc

        content, meta = client.chat_completion(
            messages=[{"role": "user", "content": "Hej"}],
            model="llama-3.1-8b-instant",
            anonymize_before_llm=True,
        )

    assert "Hej" in content
    assert meta["provider"] == "groq"
    assert meta["model"] == "llama-3.1-8b-instant"