"""Generic OpenAI-compatible chat client for Mistral / NVIDIA / Cerebras / OpenRouter / Groq.

Mirrors the OpenRouterClient surface used by ConversationMistralAnalyzer:
  - structured_chat(messages, json_schema, task_name, ...)
  - chat_completion(messages, ...)
so analyzers can swap providers without code changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..core.errors import LLMError
from .provider_secrets import get_provider_api_key

logger = logging.getLogger(__name__)

try:
    from openai import APITimeoutError, AuthenticationError, OpenAI, RateLimitError  # type: ignore

    _HAS_OPENAI = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    _HAS_OPENAI = False


class OpenAICompatClient:
    """Minimal multi-provider OpenAI-compatible client with cache + retries."""

    DEFAULT_CACHE_DIR = Path(".cache") / "llm"

    def __init__(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        base_url: str,
        default_model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        enable_cache: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key or get_provider_api_key(provider)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.enable_cache = enable_cache
        self.extra_headers = extra_headers or {}
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        if not self.api_key:
            logger.warning("No API key for provider=%s — calls will fail-fast", provider)

    def _ensure_client(self) -> Any:
        if not _HAS_OPENAI or OpenAI is None:
            raise ImportError(
                "openai package required for multi-provider LLM. pip install 'openai>=1.30'"
            )
        if self._client is None:
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "timeout": self.timeout,
            }
            if self.extra_headers:
                kwargs["default_headers"] = self.extra_headers
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{self.provider}_{key}.json"

    def _make_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        task_name: str,
    ) -> str:
        blob = json.dumps(
            {"p": self.provider, "m": model, "t": task_name, "msg": messages, "s": json_schema},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:32]

    def _load_cache(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not self.enable_cache:
            return None
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["result"], {**data.get("meta", {}), "cached": True}
        except Exception:
            return None

    def _save_cache(self, key: str, result: dict[str, Any], meta: dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        path = self._cache_path(key)
        try:
            path.write_text(
                json.dumps({"result": result, "meta": meta}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("cache write failed: %s", exc)

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        model: str | None = None,
        task_name: str = "structured",
        temperature: float = 0.15,
        max_tokens: int = 4096,
        transcript_hash: str | None = None,  # noqa: ARG002 — API parity with OpenRouterClient
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.api_key:
            raise LLMError(
                f"API key missing for provider={self.provider}",
                details={"provider": self.provider, "task": task_name, "reason": "missing_api_key"},
            )
        model = model or self.default_model
        cache_key = self._make_cache_key(model, messages, json_schema, task_name)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        client = self._ensure_client()
        logger.info(
            "EXTERNAL LLM CALL (%s) | model=%s | task=%s | chars≈%d",
            self.provider,
            model,
            task_name,
            sum(len(m.get("content", "")) for m in messages),
        )

        last_exc: Exception | None = None
        schema_name = (json_schema or {}).get("title") or task_name or "response"
        for attempt in range(1, self.max_retries + 1):
            try:
                # Prefer strict json_schema; fall back to json_object if unsupported
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": str(schema_name)[:64],
                                "schema": json_schema,
                                "strict": True,
                            },
                        },
                    )
                except Exception as schema_exc:
                    logger.debug(
                        "%s strict json_schema unsupported (%s); retrying json_object",
                        self.provider,
                        schema_exc,
                    )
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )

                content = completion.choices[0].message.content or "{}"
                parsed = json.loads(content)
                usage = getattr(completion, "usage", None)
                meta = {
                    "model": getattr(completion, "model", model),
                    "provider": self.provider,
                    "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
                    "cached": False,
                    "attempt": attempt,
                }
                self._save_cache(cache_key, parsed, meta)
                return parsed, meta
            except RateLimitError as exc:
                last_exc = exc
                logger.warning("%s rate limited (attempt %d): %s", self.provider, attempt, exc)
                time.sleep(min(2**attempt, 20))
            except (APITimeoutError, AuthenticationError) as exc:
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("%s call failed attempt %d: %s", self.provider, attempt, exc)
                time.sleep(min(1.5**attempt, 10))

        raise LLMError(
            f"{self.provider} call failed after {self.max_retries} attempts for task={task_name} "
            f"model={model}. Last error: {last_exc}. Caller should fallback.",
            details={
                "provider": self.provider,
                "task": task_name,
                "model": model,
                "error": str(last_exc),
            },
        ) from last_exc

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if not self.api_key:
            raise LLMError(
                f"API key missing for provider={self.provider}",
                details={"provider": self.provider, "reason": "missing_api_key"},
            )
        model = model or self.default_model
        client = self._ensure_client()
        logger.info("EXTERNAL LLM (plain) %s model=%s", self.provider, model)
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
            "provider": self.provider,
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
            "cached": False,
        }
        return content, meta
