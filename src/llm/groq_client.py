"""Groq Cloud client with structured outputs, caching and cost tracking.

Mirrors the OpenRouter client pattern but targets Groq's OpenAI-compatible endpoint:
    https://api.groq.com/openai/v1

Design rationale:
- Groq provides ultra-fast inference (~840 tps for Llama 8B) at competitive pricing.
- OpenAI-compatible API means we can reuse the existing `openai` SDK pattern.
- Strict json_schema mode is supported on `gpt-oss-20b` and `gpt-oss-120b` only.
- Broader json_mode (best-effort) is available on llama, qwen, compound, allam.
- ⚠️ Streaming + structured outputs are MUTUALLY EXCLUSIVE on Groq.
- Caching mirrors the OpenRouter content-addressable pattern under `.cache/llm/groq/`.
- Cost tracking uses verified pricing from groq_models in schemas.py.
- GDPR/privacy: Every egress is logged with "EXTERNAL LLM CALL (Groq)".
  Groq data centers are US + Saudi Arabia — NO confirmed EU hosting.
  Early PII redaction is MANDATORY before any Groq call (enforced in pipeline).

Usage:
    from src.llm.groq_client import GroqClient
    client = GroqClient()
    data, meta = client.structured_chat(
        messages=[{"role": "system", ...}, {"role": "user", ...}],
        json_schema=the_pydantic_model_json_schema,
        task_name="actionable_summary",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..core.errors import LLMError

logger = logging.getLogger(__name__)


# Lazy / optional dependency handling (same pattern as openrouter_client.py)
try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError

    _HAS_OPENAI = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    _HAS_OPENAI = False


def get_groq_api_key(override: str | None = None) -> str | None:
    """Resolve the effective Groq API key.

    Priority:
    1. `override` argument
    2. `GROQ_API_KEY` environment variable
    """
    if override and str(override).strip():
        return str(override).strip()
    return os.getenv("GROQ_API_KEY")


class GroqClient:
    """Reliable Groq Cloud client for fast LLM inference with optional structured output.

    See module docstring for full rationale, privacy notes, and limitations.
    """

    # Pricing from verified June 2026 data. Falls back to DEFAULT.
    PRICING: dict[str, dict[str, float]] = {
        "llama-3.1-8b-instant": {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
        "llama-3.3-70b-versatile": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
        "openai/gpt-oss-20b": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
        "openai/gpt-oss-120b": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11 / 1_000_000, "output": 0.34 / 1_000_000},
        "qwen/qwen3-32b": {"input": 0.29 / 1_000_000, "output": 0.59 / 1_000_000},
        "qwen/qwen3.6-27b": {"input": 0.60 / 1_000_000, "output": 3.00 / 1_000_000},
        "default": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    }

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_CACHE_DIR = Path(".cache") / "llm" / "groq"

    # Models that support strict json_schema (constrained decoding)
    STRICT_JSON_SCHEMA_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    # Models that support json_mode (best-effort)
    JSON_MODE_MODELS = {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b",
        "groq/compound",
        "groq/compound-mini",
        "allam-2-7b",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        enable_cache: bool = True,
        # GDPR/cost guard flags (enforced by pipeline, checked here for awareness)
        groq_eu_residency: bool = False,
    ) -> None:
        """Initialize Groq client.

        Args:
            api_key: Groq API key (prefer GROQ_API_KEY env var).
            base_url: Groq OpenAI-compatible endpoint.
            default_model: Default model for chat calls.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            cache_dir: Directory for response cache.
            enable_cache: Enable/disable response caching.
            groq_eu_residency: GDPR gate flag (must be True for sensitive EU data).
        """
        self.api_key = get_groq_api_key(api_key)
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.enable_cache = enable_cache
        self.cost_budget: float | None = None
        self.groq_eu_residency = groq_eu_residency

        if cache_dir is None:
            self.cache_dir = self.DEFAULT_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client: OpenAI | None = None

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not set. Groq calls will raise LLMError "
                "unless api_key is passed at call time. Local fallback paths remain available."
            )

    def _ensure_openai(self) -> OpenAI:
        """Lazy load the OpenAI client targeting Groq's endpoint."""
        if not _HAS_OPENAI or OpenAI is None:
            raise ImportError(
                "openai package is required for Groq LLM features. "
                "Install with: pip install 'openai>=1.30' "
                "The rest of the pipeline (local models, ASR, heuristics) works without it."
            )
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _compute_approx_cost(self, usage: Any, model: str) -> float | None:
        """Approximate USD cost from usage tokens + our pricing table."""
        if usage is None:
            return None
        try:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            prices = self.PRICING.get(model, self.PRICING["default"])
            cost = (pt * prices["input"]) + (ct * prices["output"])
            return round(cost, 6)
        except Exception:
            return None

    def _make_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        task_name: str,
        transcript_hash: str | None,
    ) -> str:
        """Content-addressable cache key."""
        payload = {
            "model": model,
            "task": task_name,
            "transcript_hash": transcript_hash or "",
            "messages": messages,
            "schema_name": (json_schema or {}).get("name") if json_schema else None,
            "provider": "groq",
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"groq_{key}.json"

    def _load_from_cache(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not self.enable_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                entry = json.load(f)
            result = entry.get("result") or {}
            meta = entry.get("meta") or {}
            meta = {**meta, "cached": True, "cache_path": str(path)}
            logger.debug("Groq LLM cache hit for key=%s (task=%s)", key[:12], meta.get("task"))
            return result, meta
        except Exception as e:
            logger.warning("Failed to read Groq LLM cache %s: %s. Will recompute.", path, e)
            return None

    def _save_to_cache(
        self,
        key: str,
        result: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        if not self.enable_cache:
            return
        path = self._cache_path(key)
        entry = {
            "cached_at": time.time(),
            "result": result,
            "meta": {k: v for k, v in meta.items() if k != "cached"},
        }
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            logger.debug("Groq LLM cache written key=%s", key[:12])
        except Exception as e:
            logger.warning("Failed to write Groq LLM cache %s: %s", path, e)

    def _build_response_format(
        self, json_schema: dict[str, Any] | None, task_name: str, model: str
    ) -> dict[str, Any] | None:
        """Build response_format for Groq's API.

        Groq supports two structured output modes:
        - strict json_schema: only gpt-oss-20b and gpt-oss-120b
        - json_mode (best-effort): llama, qwen, compound, allam

        ⚠️ Streaming + structured outputs are mutually exclusive on Groq.
        """
        if not json_schema:
            return None

        if model in self.STRICT_JSON_SCHEMA_MODELS:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": task_name or "call_analysis",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Fallback to best-effort json_mode for other models
        logger.debug(
            "Model %s does not support strict json_schema. Using json_mode (best-effort)."
            "For guaranteed structured output, use openai/gpt-oss-20b or openai/gpt-oss-120b.",
            model,
        )
        return {"type": "json_object"}

    def _check_gdpr_gate(self, anonymize_before_llm: bool) -> None:
        """Enforce GDPR residency gate.

        Raises LLMError if groq_eu_residency is False and we would send non-anonymized data.
        Callers should catch this and fallback to local analysis.
        """
        if not self.groq_eu_residency and not anonymize_before_llm:
            raise LLMError(
                "Groq data centers are US/Saudi Arabia (no EU hosting). "
                "groq_eu_residency is OFF and anonymize_before_llm is False. "
                "To use Groq, either enable groq_eu_residency in config "
                "or enable anonymize_before_llm to redact PII before sending. "
                "Fallback to local analysis is recommended.",
                error_code="groq_gdpr_gate_blocked",
            )

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        task_name: str = "holistic_analysis",
        transcript_hash: str | None = None,
        extra_headers: dict[str, str] | None = None,
        anonymize_before_llm: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute a chat completion with optional JSON schema enforcement.

        Args:
            messages: Chat messages.
            json_schema: Optional Pydantic model JSON schema for structured output.
            model: Model name (default: self.default_model).
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            task_name: Task identifier for logging/caching.
            transcript_hash: Hash for caching.
            extra_headers: Additional HTTP headers.
            anonymize_before_llm: Whether PII redaction was performed before this call.
                Required gate for non-EU residency.

        Returns:
            (parsed_result_dict, meta_dict).
            meta contains: model, usage, cost_usd, cached, task, tokens_used, etc.

        Raises:
            LLMError after exhausting retries or GDPR gate failure.
            ImportError if openai not installed.
        """
        if not messages:
            raise ValueError("messages list must be non-empty")

        model = model or self.default_model

        # GDPR residency gate (US/Saudi data centers — no EU hosting)
        self._check_gdpr_gate(anonymize_before_llm)

        client = self._ensure_openai()

        # Privacy / GDPR audit log
        transcript_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "EXTERNAL LLM CALL (Groq) | model=%s | task=%s | chars≈%d | "
            "groq_eu_residency=%s | anonymize_before_llm=%s | "
            "Data is being sent to Groq Cloud (US/Saudi Arabia data centers — NO EU hosting). "
            "Ensure PII redaction or groq_eu_residency flag is enabled for sensitive data. "
            "See docs/LLM_PROVIDERS.md for GDPR guidance.",
            model,
            task_name,
            transcript_len,
            self.groq_eu_residency,
            anonymize_before_llm,
        )

        cache_key = self._make_cache_key(model, messages, json_schema, task_name, transcript_hash)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            result, meta = cached
            meta = {**meta, "cost_usd": 0.0}
            return result, meta

        response_format = self._build_response_format(json_schema, task_name, model)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format=response_format,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers or {},
                )
                latency = round(time.time() - t0, 3)

                # Extract content
                choice = completion.choices[0]
                content = choice.message.content or "{}"
                try:
                    result: dict[str, Any] = json.loads(content)
                except json.JSONDecodeError as je:
                    logger.error("Groq returned non-JSON content: %s", content[:200])
                    raise LLMError(f"JSON parse failed (Groq, model={model}): {je}") from je

                usage = getattr(completion, "usage", None)
                cost = self._compute_approx_cost(usage, model)
                meta: dict[str, Any] = {
                    "model": getattr(completion, "model", model),
                    "task": task_name,
                    "usage": usage.model_dump() if usage else None,
                    "cost_usd": cost,
                    "latency_s": latency,
                    "cached": False,
                    "attempts": attempt + 1,
                    "id": getattr(completion, "id", None),
                    "provider": "groq",
                }

                self._save_to_cache(cache_key, result, meta)

                if self.cost_budget and (cost or 0) > self.cost_budget:
                    logger.warning(
                        "Groq LLM cost budget exceeded for task=%s: $%.5f > budget $%.5f",
                        task_name, cost or 0, self.cost_budget,
                    )
                    meta["budget_exceeded"] = True
                    meta["budget"] = self.cost_budget

                logger.info(
                    "Groq call OK | model=%s | task=%s | cost≈$%.5f | latency=%.2fs | cached=False",
                    model, task_name, cost or 0.0, latency,
                )
                try:
                    from ..core.metrics import record_llm_request

                    record_llm_request("groq", model, "success", latency)
                except Exception:
                    logger.debug("Failed to record Groq LLM metrics", exc_info=True)
                return result, meta

            except RateLimitError as e:
                last_exc = e
                wait = (2 ** attempt) * 1.2 + 0.5
                logger.warning(
                    "Groq rate limit (attempt %d/%d). Sleeping %.1fs before retry. Error: %s",
                    attempt + 1, self.max_retries, wait, e,
                )
                time.sleep(wait)
            except (APITimeoutError, APIError) as e:
                last_exc = e
                wait = (2 ** attempt) * 0.8
                logger.warning(
                    "Groq transient error (attempt %d/%d): %s. Backing off %.1fs",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                logger.error("Groq non-retryable failure: %s", e, exc_info=True)
                break

        raise LLMError(
            f"Groq call failed after {self.max_retries} attempts for task={task_name} model={model}. "
            f"Last error: {last_exc}. Caller should fallback to local analysis."
        ) from last_exc

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        anonymize_before_llm: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Plain text chat completion (no schema enforcement). Returns (content, meta)."""
        model = model or self.default_model

        self._check_gdpr_gate(anonymize_before_llm)

        client = self._ensure_openai()

        logger.info(
            "EXTERNAL LLM (Groq) plain call | model=%s | groq_eu_residency=%s | anonymize_before_llm=%s",
            model, self.groq_eu_residency, anonymize_before_llm,
        )
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            meta = {
                "model": getattr(completion, "model", model),
                "usage": usage.model_dump() if usage else None,
                "cost_usd": self._compute_approx_cost(usage, model),
                "cached": False,
                "provider": "groq",
            }
            return content, meta
        except Exception as e:
            raise LLMError(f"Groq plain chat_completion failed: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """Return static curated model list with metadata.

        Uses GROQ_MODELS from schemas.py as the source of truth.
        Falls back to built-in PRICING keys if schemas not available.
        """
        try:
            from .schemas import GROQ_MODELS
            return [{"id": name, **info} for name, info in GROQ_MODELS.items()]
        except ImportError:
            # Minimal fallback from pricing table
            return [
                {"id": name, "pricing_in": pricing["input"] * 1_000_000,
                 "pricing_out": pricing["output"] * 1_000_000}
                for name, pricing in self.PRICING.items()
                if name != "default"
            ]

    def clear_cache(self) -> int:
        """Delete all cached Groq LLM responses. Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for p in self.cache_dir.glob("groq_*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        logger.info("Cleared %d Groq LLM cache files from %s", count, self.cache_dir)
        return count