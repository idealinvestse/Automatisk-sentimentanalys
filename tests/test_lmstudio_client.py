from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.errors import LLMError
from src.llm.client_factory import provider_requires_api_key, resolve_llm_client
from src.llm.context_budget import resolve_context_budget
from src.llm.lmstudio_client import LMStudioClient, validate_loopback_base_url
from src.pipeline_steps import PipelineLLMContext, _llm_credentials_available


class _Tokenizer:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        return self.tokens


def test_context_budget_reserves_output_and_margin() -> None:
    budget = resolve_context_budget(
        _Tokenizer(59760),
        [{"role": "user", "content": "hej"}],
        requested_context=70000,
        loaded_context=70000,
        output_tokens=8192,
        safety_margin_tokens=2048,
    )
    assert budget.fits
    assert budget.remaining_tokens == 0


def test_context_budget_rejects_underloaded_model() -> None:
    with pytest.raises(LLMError) as exc_info:
        resolve_context_budget(
            _Tokenizer(100),
            [{"role": "user", "content": "hej"}],
            requested_context=70000,
            loaded_context=35072,
            output_tokens=8192,
        )
    assert exc_info.value.error_code == "llm_context_not_ready"


def test_context_budget_never_truncates_overflow() -> None:
    with pytest.raises(LLMError) as exc_info:
        resolve_context_budget(
            _Tokenizer(59761),
            [{"role": "user", "content": "hej"}],
            requested_context=70000,
            loaded_context=70000,
            output_tokens=8192,
            safety_margin_tokens=2048,
        )
    assert exc_info.value.error_code == "llm_context_budget_exceeded"


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:1234",
        "http://localhost:1234",
        "http://192.168.1.5:1234",
        "http://127.0.0.1:1234/v1/other",
        "http://user:secret@127.0.0.1:1234",
    ],
)
def test_lmstudio_endpoint_rejects_non_loopback_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(LLMError):
        validate_loopback_base_url(url)


def test_lmstudio_endpoint_accepts_numeric_loopback() -> None:
    assert validate_loopback_base_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


def test_lmstudio_model_status_reports_loaded_context() -> None:
    client = LMStudioClient(
        base_url="http://127.0.0.1:1234",
        default_model="the-model",
        enable_cache=False,
    )
    with patch.object(
        client,
        "_native_get",
        return_value={
            "models": [
                {
                    "key": "the-model",
                    "max_context_length": 262144,
                    "quantization": {"name": "Q4_K_S"},
                    "capabilities": {"reasoning": {"default": "on"}},
                    "loaded_instances": [{"id": "the-model", "config": {"context_length": 35072}}],
                }
            ]
        },
    ):
        status = client.model_status()
    assert status.loaded_context == 35072
    assert status.max_context == 262144
    assert status.quantization == "Q4_K_S"


def test_lmstudio_preflight_warns_but_does_not_fail_on_reasoning_on() -> None:
    """Reasoning-on must not hard-fail preflight; structured output still works."""
    client = LMStudioClient(
        base_url="http://127.0.0.1:1234",
        default_model="the-model",
        enable_cache=False,
    )
    with (
        patch.object(
            client,
            "model_status",
            return_value=type(
                "Status",
                (),
                {
                    "loaded": True,
                    "loaded_context": 70000,
                    "max_context": 262144,
                    "model": "the-model",
                    "instance_id": "the-model",
                    "quantization": "Q4_K_S",
                    "reasoning_default": "on",
                    "to_dict": lambda self: {"reasoning_default": "on"},
                },
            )(),
        ),
        patch.object(client, "count_chat_tokens", return_value=100),
    ):
        budget = client._preflight(
            [{"role": "user", "content": "hej"}],
            model="the-model",
            output_tokens=8192,
        )
    assert budget.fits
    assert budget.loaded_context == 70000


def test_pipeline_accepts_keyless_local_provider() -> None:
    ctx = PipelineLLMContext(
        profile="callcenter",
        provider="lmstudio",
        use_mistral_llm=True,
        deep_analysis=True,
        llm_model=None,
        llm_api_key=None,
        groq_eu_residency=False,
    )
    assert _llm_credentials_available(ctx) is True


def test_factory_resolves_keyless_local_provider() -> None:
    cfg = {
        "providers": {
            "lmstudio": {
                "base_url": "http://127.0.0.1:1234",
                "default_model": "the-model",
                "requested_context": 70000,
            }
        }
    }
    resolved = resolve_llm_client("lmstudio", config=cfg)
    assert resolved.local is True
    assert resolved.model == "the-model"
    assert provider_requires_api_key("lmstudio") is False


def test_local_provider_client_is_not_cloud_compatible() -> None:
    """LM Studio client must not be an OpenRouter or Groq client instance."""
    cfg = {
        "providers": {
            "lmstudio": {
                "base_url": "http://127.0.0.1:1234",
                "default_model": "the-model",
                "requested_context": 70000,
            }
        }
    }
    resolved = resolve_llm_client("lmstudio", config=cfg)
    assert resolved.client is not None
    client_class = type(resolved.client).__name__
    assert client_class == "LMStudioClient"
    assert resolved.client.provider == "lmstudio"


def test_unknown_provider_rejected_by_factory() -> None:
    from src.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        resolve_llm_client("unknown_provider_xyz")
