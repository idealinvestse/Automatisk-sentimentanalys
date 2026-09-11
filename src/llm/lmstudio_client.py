"""Strict local LM Studio client for post-transcription analysis."""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

from ..core.errors import LLMError
from .context_budget import ContextBudget, resolve_context_budget
from .openai_compat_client import OpenAICompatClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMStudioModelStatus:
    """Verified state for one downloaded LM Studio model."""

    model: str
    loaded: bool
    loaded_context: int | None
    max_context: int | None
    instance_id: str | None
    quantization: str | None
    reasoning_default: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible status object."""
        return {
            "model": self.model,
            "loaded": self.loaded,
            "loaded_context": self.loaded_context,
            "max_context": self.max_context,
            "instance_id": self.instance_id,
            "quantization": self.quantization,
            "reasoning_default": self.reasoning_default,
        }


def validate_loopback_base_url(base_url: str) -> str:
    """Allow only explicit loopback HTTP endpoints for the local provider."""
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise LLMError(
            "LM Studio endpoint must be an unauthenticated HTTP loopback URL",
            error_code="lmstudio_endpoint_forbidden",
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise LLMError(
            "LM Studio endpoint must use a numeric loopback address",
            error_code="lmstudio_endpoint_forbidden",
        ) from exc
    if not address.is_loopback:
        raise LLMError(
            "LM Studio endpoint must remain on loopback",
            error_code="lmstudio_endpoint_forbidden",
        )
    if parsed.path.rstrip("/") not in {"", "/v1"} or parsed.query or parsed.fragment:
        raise LLMError(
            "LM Studio endpoint may only contain the optional /v1 path",
            error_code="lmstudio_endpoint_forbidden",
        )
    return base_url.rstrip("/")


class LMStudioClient(OpenAICompatClient):
    """OpenAI-compatible client with local readiness and exact token preflight."""

    _inference_lock = threading.Semaphore(1)

    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        api_key: str | None = None,
        requested_context: int = 70000,
        safety_margin_tokens: int = 2048,
        timeout: float = 900.0,
        enable_cache: bool = True,
    ) -> None:
        base_url = validate_loopback_base_url(base_url)
        openai_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        super().__init__(
            "lmstudio",
            api_key=api_key or "lm-studio-local",
            base_url=openai_base,
            default_model=default_model,
            timeout=timeout,
            max_retries=1,
            enable_cache=enable_cache,
        )
        self.native_base_url = openai_base.removesuffix("/v1")
        self.requested_context = requested_context
        self.safety_margin_tokens = safety_margin_tokens
        self.last_budget: ContextBudget | None = None

    def _native_get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.native_base_url}{path}",
            headers={"Accept": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=min(self.timeout, 2.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise json.JSONDecodeError("expected object", "", 0)
                return cast(dict[str, Any], payload)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LLMError(
                "LM Studio is not reachable on the configured loopback endpoint",
                error_code="lmstudio_unreachable",
            ) from exc

    def model_status(self, model: str | None = None) -> LMStudioModelStatus:
        """Read current loaded context from LM Studio's native model endpoint."""
        model = model or self.default_model
        payload = self._native_get("/api/v1/models")
        entries = payload.get("models") or []
        entry = next((item for item in entries if item.get("key") == model), None)
        if not isinstance(entry, dict):
            raise LLMError(
                f"Configured LM Studio model is not downloaded: {model}",
                error_code="lmstudio_model_missing",
                details={"model": model},
            )
        instances = entry.get("loaded_instances") or []
        instance = instances[0] if instances else None
        raw_config = instance.get("config") if isinstance(instance, dict) else {}
        config = raw_config if isinstance(raw_config, dict) else {}
        capabilities = entry.get("capabilities") or {}
        reasoning = capabilities.get("reasoning") if isinstance(capabilities, dict) else {}
        quantization = entry.get("quantization") or {}
        return LMStudioModelStatus(
            model=model,
            loaded=bool(instance),
            loaded_context=int(config["context_length"]) if config.get("context_length") else None,
            max_context=int(entry["max_context_length"]) if entry.get("max_context_length") else None,
            instance_id=str(instance.get("id")) if isinstance(instance, dict) else None,
            quantization=str(quantization.get("name")) if quantization.get("name") else None,
            reasoning_default=(
                str(reasoning.get("default")) if isinstance(reasoning, dict) and reasoning.get("default") else None
            ),
        )

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        """Use the exact loaded model tokenizer and prompt template through the LM Studio SDK."""
        try:
            import lmstudio as lms  # type: ignore
        except ImportError as exc:
            raise LLMError(
                "lmstudio SDK is required for exact 70k token budgeting",
                error_code="lmstudio_sdk_missing",
            ) from exc
        host = urlparse(self.native_base_url).netloc
        try:
            with lms.Client(host) as client:
                model = client.llm.model(self.default_model)
                formatted = model.apply_prompt_template({"messages": messages})
                return len(model.tokenize(formatted))
        except Exception as exc:
            raise LLMError(
                "LM Studio tokenizer preflight failed",
                error_code="lmstudio_tokenizer_failed",
                details={"model": self.default_model},
            ) from exc

    def _preflight(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        output_tokens: int,
    ) -> ContextBudget:
        status = self.model_status(model)
        if not status.loaded or status.loaded_context is None:
            raise LLMError(
                f"Configured LM Studio model is not loaded: {model}",
                error_code="lmstudio_model_not_loaded",
                details=status.to_dict(),
            )
        if status.reasoning_default not in {None, "off"}:
            logger.warning(
                "LM Studio reasoning is enabled (default=%s) for %s; "
                "structured output may include reasoning tokens that consume budget. "
                "Validation will reject empty content.",
                status.reasoning_default,
                model,
            )
        budget = resolve_context_budget(
            self,
            messages,
            requested_context=self.requested_context,
            loaded_context=status.loaded_context,
            output_tokens=output_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
        )
        self.last_budget = budget
        return budget

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        *,
        model: str | None = None,
        task_name: str = "structured",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        transcript_hash: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run schema-constrained local inference only after exact context preflight."""
        model = model or self.default_model
        with self._inference_lock:
            budget = self._preflight(messages, model=model, output_tokens=max_tokens)
            result, meta = super().structured_chat(
                messages,
                json_schema,
                model=model,
                task_name=task_name,
                temperature=temperature,
                max_tokens=max_tokens,
                transcript_hash=transcript_hash,
            )
            return result, {
                **meta,
                "local": True,
                "api_cost_usd": 0.0,
                "context_budget": budget.to_dict(),
            }

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Run a plain local completion with the same fail-closed context policy."""
        model = model or self.default_model
        with self._inference_lock:
            budget = self._preflight(messages, model=model, output_tokens=max_tokens)
            text, meta = super().chat_completion(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if not text.strip():
                raise LLMError(
                    "LM Studio returned no final content",
                    error_code="lmstudio_empty_content",
                )
            return text, {
                **meta,
                "local": True,
                "api_cost_usd": 0.0,
                "context_budget": budget.to_dict(),
            }
