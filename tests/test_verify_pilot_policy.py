"""Tests for scripts/verify_pilot_policy.py."""

from __future__ import annotations

import pytest

from scripts.verify_pilot_policy import (
    check_anonymize_default,
    check_asr_schema_default,
    check_cloud_keys,
    check_llm_enabled_profiles_anonymize,
    check_production_guards,
    check_runtime_pilot_locks,
)


def test_anonymize_default_true() -> None:
    ok, msg = check_anonymize_default()
    assert ok is True
    assert "True" in msg


def test_llm_enabled_profiles_require_anonymize() -> None:
    ok, msg = check_llm_enabled_profiles_anonymize()
    assert ok is True
    assert "anonymize_before_llm" in msg


def test_runtime_locks_fail_lmstudio_in_strict_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.delenv("SENTIMENT_PILOT_ALLOW_LMSTUDIO", raising=False)
    monkeypatch.delenv("API_ALLOW_CLIENT_LLM_KEY", raising=False)

    class _Cfg:
        class llm:
            enabled = True
            provider = "lmstudio"

        class asr:
            provider = "local"

    monkeypatch.setattr(
        "src.install.user_config.load_user_config",
        lambda: _Cfg(),
    )
    ok, messages = check_runtime_pilot_locks(strict=True)
    assert ok is False
    assert any("lmstudio" in m.lower() for m in messages)


def test_asr_schema_default_local() -> None:
    ok, msg = check_asr_schema_default()
    assert ok is True
    assert "local" in msg


def test_production_guards_skipped_when_not_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    ok, msg = check_production_guards()
    assert ok is True
    assert "skipped" in msg.lower() or "not set" in msg.lower()


def test_production_guards_fail_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    ok, msg = check_production_guards()
    assert ok is False
    assert "SENTIMENT_API_KEY" in msg


def test_cloud_keys_strict_prod_fails_on_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_STT_API_KEY", raising=False)
    ok, messages = check_cloud_keys(strict=True)
    assert ok is False
    assert any("FAIL" in m for m in messages)


def test_cloud_keys_warn_only_when_not_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    ok, messages = check_cloud_keys(strict=False)
    assert ok is True
    assert any("WARN" in m for m in messages)
