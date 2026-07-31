"""Adapter: MultiProviderRouter as an OpenRouterClient-compatible structured_chat client."""

from __future__ import annotations

from typing import Any

from .multi_provider_router import MultiProviderRouter, RouterProfile, RoutingTier


class RouterBackedClient:
    """Exposes structured_chat/chat_completion via MultiProviderRouter failover."""

    def __init__(
        self,
        profile: str | RouterProfile = RouterProfile.FREE_SEQUENTIAL,
        tier: str | RoutingTier = RoutingTier.BALANCED,
        default_model: str | None = None,
    ) -> None:
        self.router = MultiProviderRouter(profile=profile, tier=tier)
        self.default_model = default_model or "auto"
        self.provider = f"router:{self.router.profile.value}"

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        model: str | None = None,  # noqa: ARG002 — router selects
        task_name: str = "structured",
        temperature: float = 0.15,
        max_tokens: int = 4096,
        transcript_hash: str | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.router.structured_chat_with_failover(
            messages,
            json_schema,
            task_name=task_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,  # noqa: ARG002
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        return self.router.chat_completion_with_failover(
            messages, max_tokens=max_tokens, temperature=temperature
        )
