"""Single provider/model resolution point for all LLM-backed analysis steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.errors import ConfigurationError
from .provider_secrets import get_provider_api_key, load_provider_config

LOCAL_LLM_PROVIDERS = frozenset({"lmstudio"})
ROUTER_PROVIDERS = frozenset({"auto", "free_sequential", "sv_optimal", "router"})
DIRECT_COMPAT_PROVIDERS = frozenset({"mistral", "nvidia", "cerebras"})
KNOWN_PROVIDERS = LOCAL_LLM_PROVIDERS | ROUTER_PROVIDERS | DIRECT_COMPAT_PROVIDERS | {
    "openrouter",
    "groq",
}


@dataclass(frozen=True)
class ResolvedLLMClient:
    """Resolved transport and immutable routing identity."""

    provider: str
    model: str
    client: Any | None
    local: bool


def provider_requires_api_key(provider: str) -> bool:
    """Return whether provider configuration requires a cloud credential."""
    return provider.lower() not in LOCAL_LLM_PROVIDERS


def resolve_llm_client(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> ResolvedLLMClient:
    """Resolve a provider once, without implicit cross-provider model fallback."""
    provider = (provider or "openrouter").strip().lower()
    if provider not in KNOWN_PROVIDERS:
        raise ConfigurationError(
            f"Unknown LLM provider: {provider}",
            error_code="unknown_llm_provider",
            details={"provider": provider},
        )
    cfg = config or load_provider_config()
    providers = cfg.get("providers") or {}

    if provider == "lmstudio":
        from .lmstudio_client import LMStudioClient

        spec = providers.get(provider) or {}
        resolved_model = model or str(spec.get("default_model") or "")
        if not resolved_model:
            raise ConfigurationError("LM Studio default_model is not configured")
        local_client = LMStudioClient(
            base_url=str(spec.get("base_url") or "http://127.0.0.1:1234"),
            default_model=resolved_model,
            api_key=api_key or get_provider_api_key(provider, config=cfg),
            requested_context=int(spec.get("requested_context") or 70000),
            safety_margin_tokens=int(spec.get("safety_margin_tokens") or 2048),
            timeout=float(spec.get("timeout_seconds") or 900),
            enable_cache=bool(spec.get("enable_cache", False)),
        )
        return ResolvedLLMClient(provider, resolved_model, local_client, True)

    if provider == "groq":
        from .groq_client import GroqClient

        groq_client = GroqClient(api_key=api_key) if api_key else GroqClient()
        groq_model = model or getattr(groq_client, "default_model", None)
        if not groq_model:
            raise ConfigurationError("Groq model is not configured")
        return ResolvedLLMClient(provider, str(groq_model), groq_client, False)

    if provider in ROUTER_PROVIDERS:
        from .router_client import RouterBackedClient

        profile = "sv_optimal" if provider == "sv_optimal" else "free_sequential"
        router_client = RouterBackedClient(profile=profile, tier="balanced", default_model=model)
        return ResolvedLLMClient(
            provider,
            str(model or router_client.default_model),
            router_client,
            False,
        )

    if provider in {"openrouter", "mistral", "nvidia", "cerebras"}:
        from .openai_compat_client import OpenAICompatClient

        spec = providers.get(provider) or {}
        default_base = "https://openrouter.ai/api/v1" if provider == "openrouter" else ""
        base_url = str(spec.get("base_url") or default_base)
        curated = spec.get("curated_sv") or {}
        compat_model = model or (
            curated.get("balanced") if isinstance(curated, dict) else None
        )
        if not compat_model:
            raise ConfigurationError(f"No model configured for provider={provider}")
        compat_client = OpenAICompatClient(
            provider=provider,
            api_key=api_key or get_provider_api_key(provider, config=cfg),
            base_url=base_url,
            default_model=str(compat_model),
            extra_headers=dict(spec.get("headers_extra") or {}),
        )
        return ResolvedLLMClient(provider, str(compat_model), compat_client, False)

    return ResolvedLLMClient(provider, model or "", None, False)
