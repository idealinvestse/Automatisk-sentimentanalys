"""Holistic fallback meta must block a second LLM QA call."""

from __future__ import annotations

import pytest

from src.llm.router_client import RouterBackedClient
from src.pipeline_steps import (
    PipelineLLMContext,
    _holistic_fallback_payload,
    _profile_anonymize_before_llm,
    _resolve_mistral_compat_client,
    should_run_llm_qa,
)


def _ctx(*, use_mistral_llm: bool = True) -> PipelineLLMContext:
    return PipelineLLMContext(
        profile="callcenter",
        provider="openrouter",
        use_mistral_llm=use_mistral_llm,
        deep_analysis=True,
        llm_model=None,
        llm_api_key="sk-test",
        groq_eu_residency=False,
    )


def test_compat_client_keeps_router_profiles_off_openrouter() -> None:
    expected = {
        "sv_optimal": "sv_optimal",
        "free_sequential": "free_sequential",
        "auto": "free_sequential",
        "router": "free_sequential",
    }
    for provider, profile in expected.items():
        ctx = PipelineLLMContext(
            profile="callcenter",
            provider=provider,
            use_mistral_llm=True,
            deep_analysis=True,
            llm_model=None,
            llm_api_key=None,
            groq_eu_residency=False,
        )
        client, _model = _resolve_mistral_compat_client(ctx, segment_count=4)
        assert isinstance(client, RouterBackedClient)
        assert profile in client.provider


def test_compat_client_openrouter_uses_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    class _Resolved:
        client = sentinel
        model = "mistralai/mistral-medium-3-5"

    monkeypatch.setattr(
        "src.llm.client_factory.resolve_llm_client",
        lambda *args, **kwargs: _Resolved(),
    )
    ctx = _ctx()
    client, model = _resolve_mistral_compat_client(ctx, segment_count=2)
    assert client is sentinel
    assert model == "mistralai/mistral-medium-3-5"


def test_complaint_profile_anonymizes_for_groq_gate() -> None:
    assert _profile_anonymize_before_llm("complaint") is True
    assert _profile_anonymize_before_llm("support") is True
    assert _profile_anonymize_before_llm("sales") is False


def test_holistic_exception_payload_has_meta_llm_error() -> None:
    payload = _holistic_fallback_payload(
        reason="llm_error",
        provider="openrouter",
        error="boom",
    )
    assert payload["meta"]["llm_used"] is False
    assert payload["meta"]["llm_fallback_reason"] == "llm_error"
    assert should_run_llm_qa(_ctx(), payload, credentials_available=True) is False


def test_qa_does_not_retry_when_legacy_top_level_reason_lacks_meta() -> None:
    legacy = {"llm_used": False, "llm_fallback_reason": "timeout", "error": "timeout"}
    assert should_run_llm_qa(_ctx(), legacy, credentials_available=True) is False


def test_qa_runs_only_after_successful_holistic() -> None:
    ok = {"meta": {"llm_used": True, "provider": "openrouter"}}
    assert should_run_llm_qa(_ctx(), ok, credentials_available=True) is True
    assert should_run_llm_qa(_ctx(), ok, credentials_available=False) is False


def test_qa_skips_lmstudio_unreachable() -> None:
    down = {"meta": {"llm_used": False, "llm_fallback_reason": "lmstudio_unreachable"}}
    assert should_run_llm_qa(_ctx(), down, credentials_available=True) is False


def test_qa_skips_ccp_and_missing_key() -> None:
    ccp = {"meta": {"llm_used": False, "llm_fallback_reason": "ccp_failed"}}
    missing = {"meta": {"llm_used": False, "llm_fallback_reason": "missing_api_key"}}
    assert should_run_llm_qa(_ctx(), ccp, credentials_available=True) is False
    assert should_run_llm_qa(_ctx(), missing, credentials_available=True) is False
