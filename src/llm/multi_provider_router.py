"""Multi-provider LLM router with free-sequential and Swedish-optimal profiles.

Design:
- free_sequential: only free/free-tier models; walk providers one-by-one; honor cooldowns
  after 429 so we never hammer multiple free tiers in parallel.
- sv_optimal: pick best Swedish-capable model per provider; sequential by default, with
  optional parallel fan-out for independent tasks.

Rate-limit state is persisted under data/model_catalogs/rate_limit_state.json.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..core.errors import LLMError
from .model_catalog import list_free_models, load_provider_catalog
from .openai_compat_client import OpenAICompatClient
from .provider_secrets import get_provider_api_key, list_configured_providers, load_provider_config

logger = logging.getLogger(__name__)


class RouterProfile(StrEnum):
    FREE_SEQUENTIAL = "free_sequential"
    SV_OPTIMAL = "sv_optimal"


class RoutingTier(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


@dataclass(frozen=True)
class RouteChoice:
    provider: str
    model: str
    profile: str
    tier: str
    reason: str


class RateLimitTracker:
    """Simple persistent per-provider RPM / cooldown tracker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.debug("rate state save failed: %s", exc)

    def is_available(self, provider: str, rpm: int = 30) -> bool:
        with self._lock:
            now = time.time()
            entry = self._state.get(provider) or {}
            cooldown_until = float(entry.get("cooldown_until") or 0)
            if now < cooldown_until:
                return False
            window_start = float(entry.get("window_start") or 0)
            count = int(entry.get("window_count") or 0)
            if now - window_start >= 60:
                return True
            return count < max(1, rpm)

    def record_success(self, provider: str) -> None:
        with self._lock:
            now = time.time()
            entry = self._state.setdefault(provider, {})
            window_start = float(entry.get("window_start") or 0)
            if now - window_start >= 60:
                entry["window_start"] = now
                entry["window_count"] = 1
            else:
                entry["window_count"] = int(entry.get("window_count") or 0) + 1
            entry["last_success"] = now
            self._save()

    def record_rate_limit(self, provider: str, cooldown_seconds: float = 60.0) -> None:
        with self._lock:
            now = time.time()
            entry = self._state.setdefault(provider, {})
            entry["cooldown_until"] = now + cooldown_seconds
            entry["last_429"] = now
            self._save()
            logger.warning(
                "Provider %s entered cooldown for %.0fs after rate limit",
                provider,
                cooldown_seconds,
            )


class MultiProviderRouter:
    """Select provider+model and execute chat with failover."""

    def __init__(
        self,
        profile: RouterProfile | str = RouterProfile.FREE_SEQUENTIAL,
        *,
        config: dict[str, Any] | None = None,
        tier: RoutingTier | str = RoutingTier.BALANCED,
    ) -> None:
        self.cfg = config or load_provider_config()
        self.profile = RouterProfile(str(profile))
        self.tier = RoutingTier(str(tier).lower()) if not isinstance(tier, RoutingTier) else tier
        cat_cfg = self.cfg.get("catalog") or {}
        state_path = Path(cat_cfg.get("rate_state_file") or "data/model_catalogs/rate_limit_state.json")
        self.rates = RateLimitTracker(state_path)

    # ------------------------------------------------------------------ select
    def _provider_order(self) -> list[str]:
        profiles = self.cfg.get("profiles") or {}
        prof = profiles.get(self.profile.value) or {}
        order = list(prof.get("provider_order") or [])
        configured = list_configured_providers(self.cfg)
        # Prefer providers that actually have keys
        ordered = [p for p in order if configured.get(p)]
        # Append any other configured providers not listed
        for p, ok in configured.items():
            if ok and p not in ordered and p != "groq":
                ordered.append(p)
        return ordered

    def _curated_sv_model(self, provider: str) -> str | None:
        providers = self.cfg.get("providers") or {}
        spec = providers.get(provider) or {}
        curated = spec.get("curated_sv") or {}
        if isinstance(curated, dict):
            return curated.get(self.tier.value) or curated.get("balanced")
        # openrouter special
        profiles = self.cfg.get("profiles") or {}
        sv = (profiles.get("sv_optimal") or {}).get("openrouter_sv") or {}
        if provider == "openrouter":
            return sv.get(self.tier.value) or sv.get("balanced")
        return None

    def _pick_free_model(self, provider: str) -> str | None:
        free = list_free_models(provider, self.cfg)
        if free:
            return free[0]
        # curated only
        spec = (self.cfg.get("providers") or {}).get(provider) or {}
        curated = list(spec.get("curated_free") or [])
        return curated[0] if curated else None

    def _model_exists_in_catalog(self, provider: str, model: str) -> bool:
        cat = load_provider_catalog(provider, self.cfg)
        if not cat:
            return True  # optimistic if no catalog yet
        ids = {m.get("id") for m in (cat.get("models") or []) if isinstance(m, dict)}
        return not ids or model in ids

    def select_route(self, *, prefer_provider: str | None = None) -> RouteChoice:
        """Pick next available provider+model for the active profile."""
        order = self._provider_order()
        if prefer_provider and prefer_provider in order:
            order = [prefer_provider] + [p for p in order if p != prefer_provider]

        free_only = bool(
            ((self.cfg.get("profiles") or {}).get(self.profile.value) or {}).get("free_only")
        )
        profiles = self.cfg.get("profiles") or {}
        prof = profiles.get(self.profile.value) or {}
        cooldown = float(prof.get("cooldown_seconds_on_429") or 60)

        for provider in order:
            spec = (self.cfg.get("providers") or {}).get(provider) or {}
            rpm = int(spec.get("default_rpm") or 30)
            if not self.rates.is_available(provider, rpm=rpm):
                logger.debug("skip %s — rate cooldown/window full", provider)
                continue
            if not get_provider_api_key(provider, config=self.cfg):
                continue

            if free_only or self.profile == RouterProfile.FREE_SEQUENTIAL:
                model = self._pick_free_model(provider)
                reason = "free_tier"
            else:
                model = self._curated_sv_model(provider)
                reason = f"sv_optimal:{self.tier.value}"
                if model and not self._model_exists_in_catalog(provider, model):
                    # fall back to first catalog model
                    cat = load_provider_catalog(provider, self.cfg)
                    models = (cat or {}).get("models") or []
                    if models:
                        model = models[0].get("id")
                        reason = "sv_optimal:catalog_fallback"

            if not model:
                continue
            return RouteChoice(
                provider=provider,
                model=str(model),
                profile=self.profile.value,
                tier=self.tier.value,
                reason=reason,
            )

        raise LLMError(
            f"No available provider/model for profile={self.profile.value}",
            details={"profile": self.profile.value, "order": order, "cooldown_hint": cooldown},
        )

    def build_client(self, choice: RouteChoice) -> OpenAICompatClient:
        spec = (self.cfg.get("providers") or {}).get(choice.provider) or {}
        base_url = str(spec.get("base_url") or "")
        if not base_url:
            raise LLMError(f"No base_url for provider={choice.provider}")
        return OpenAICompatClient(
            provider=choice.provider,
            api_key=get_provider_api_key(choice.provider, config=self.cfg),
            base_url=base_url,
            default_model=choice.model,
            extra_headers=dict(spec.get("headers_extra") or {}),
        )

    # ----------------------------------------------------------------- execute
    def structured_chat_with_failover(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        task_name: str = "structured",
        temperature: float = 0.15,
        max_tokens: int = 4096,
        max_attempts: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Try providers sequentially until one succeeds (free profile stays single-provider-at-a-time)."""
        profiles = self.cfg.get("profiles") or {}
        prof = profiles.get(self.profile.value) or {}
        attempts_left = int(max_attempts or prof.get("max_provider_attempts") or 4)
        cooldown = float(prof.get("cooldown_seconds_on_429") or 60)
        tried: list[str] = []
        last_err: Exception | None = None

        while attempts_left > 0:
            attempts_left -= 1
            try:
                choice = self.select_route(
                    prefer_provider=None if not tried else next(
                        (p for p in self._provider_order() if p not in tried), None
                    )
                )
            except LLMError as exc:
                last_err = exc
                break
            if choice.provider in tried:
                # force next
                tried.append(choice.provider)
                continue
            tried.append(choice.provider)
            client = self.build_client(choice)
            try:
                result, meta = client.structured_chat(
                    messages,
                    json_schema,
                    model=choice.model,
                    task_name=task_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.rates.record_success(choice.provider)
                meta = {
                    **meta,
                    "route": {
                        "provider": choice.provider,
                        "model": choice.model,
                        "profile": choice.profile,
                        "tier": choice.tier,
                        "reason": choice.reason,
                        "tried": tried,
                    },
                }
                return result, meta
            except LLMError as exc:
                last_err = exc
                err_s = str(exc).lower()
                if "rate" in err_s or "429" in err_s:
                    self.rates.record_rate_limit(choice.provider, cooldown)
                logger.warning(
                    "Router failover: %s/%s failed: %s",
                    choice.provider,
                    choice.model,
                    exc,
                )
                continue
            except Exception as exc:
                last_err = exc
                logger.warning("Router failover unexpected: %s", exc)
                continue

        raise LLMError(
            f"All providers failed for profile={self.profile.value} task={task_name}. "
            f"Tried={tried}. Last error: {last_err}",
            details={"tried": tried, "profile": self.profile.value, "task": task_name},
        ) from last_err

    def chat_completion_with_failover(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, Any]]:
        choice = self.select_route()
        client = self.build_client(choice)
        try:
            text, meta = client.chat_completion(
                messages, model=choice.model, max_tokens=max_tokens, temperature=temperature
            )
            self.rates.record_success(choice.provider)
            meta["route"] = {
                "provider": choice.provider,
                "model": choice.model,
                "profile": choice.profile,
                "tier": choice.tier,
                "reason": choice.reason,
            }
            return text, meta
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                profiles = self.cfg.get("profiles") or {}
                prof = profiles.get(self.profile.value) or {}
                self.rates.record_rate_limit(
                    choice.provider, float(prof.get("cooldown_seconds_on_429") or 60)
                )
            # one failover hop
            text, meta = self._chat_failover_once(messages, exclude=choice.provider, max_tokens=max_tokens, temperature=temperature)
            return text, meta

    def _chat_failover_once(
        self,
        messages: list[dict[str, str]],
        *,
        exclude: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any]]:
        for provider in self._provider_order():
            if provider == exclude:
                continue
            try:
                choice = self.select_route(prefer_provider=provider)
            except LLMError:
                continue
            if choice.provider == exclude:
                continue
            client = self.build_client(choice)
            text, meta = client.chat_completion(
                messages, model=choice.model, max_tokens=max_tokens, temperature=temperature
            )
            self.rates.record_success(choice.provider)
            meta["route"] = {
                "provider": choice.provider,
                "model": choice.model,
                "profile": choice.profile,
                "tier": choice.tier,
                "reason": choice.reason + "+failover",
            }
            return text, meta
        raise LLMError("chat_completion failover exhausted")

    def map_parallel(
        self,
        tasks: dict[str, dict[str, Any]],
        *,
        json_schema: dict[str, Any],
        temperature: float = 0.15,
        max_tokens: int = 2048,
        max_workers: int = 3,
    ) -> dict[str, Any]:
        """Run independent structured tasks in parallel (sv_optimal only).

        tasks: {task_name: {"messages": [...]}}
        Free profile always forces sequential (rate-limit safety).
        """
        if self.profile == RouterProfile.FREE_SEQUENTIAL:
            out = {}
            for name, payload in tasks.items():
                result, meta = self.structured_chat_with_failover(
                    payload["messages"],
                    json_schema,
                    task_name=name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                out[name] = {"result": result, "meta": meta}
            return out

        results: dict[str, Any] = {}
        # Parallel uses different providers when possible
        providers = self._provider_order()
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(tasks)))) as pool:
            futures = {}
            for idx, (name, payload) in enumerate(tasks.items()):
                prefer = providers[idx % len(providers)] if providers else None

                def _run(task_name=name, msgs=payload["messages"], pref=prefer):
                    # temporarily select preferred provider
                    choice = self.select_route(prefer_provider=pref)
                    client = self.build_client(choice)
                    result, meta = client.structured_chat(
                        msgs,
                        json_schema,
                        model=choice.model,
                        task_name=task_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self.rates.record_success(choice.provider)
                    meta["route"] = {
                        "provider": choice.provider,
                        "model": choice.model,
                        "profile": choice.profile,
                        "tier": choice.tier,
                        "reason": choice.reason + "+parallel",
                    }
                    return task_name, result, meta

                futures[pool.submit(_run)] = name

            for fut in as_completed(futures):
                try:
                    task_name, result, meta = fut.result()
                    results[task_name] = {"result": result, "meta": meta}
                except Exception as exc:
                    name = futures[fut]
                    results[name] = {"error": str(exc)}
        return results


def make_router(
    profile: str | None = None,
    tier: str = "balanced",
) -> MultiProviderRouter:
    """Factory: profile from env LLM_ROUTER_PROFILE or default free_sequential."""
    import os

    prof = profile or os.getenv("LLM_ROUTER_PROFILE") or "free_sequential"
    return MultiProviderRouter(profile=prof, tier=tier)
