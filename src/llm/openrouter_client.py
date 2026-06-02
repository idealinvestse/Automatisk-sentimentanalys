"""OpenRouter client for Mistral models with strict structured outputs, caching and cost tracking.

This module implements Task 3.1.1 of UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md.

Design rationale (why this solution):
- Uses the official `openai` SDK against OpenRouter's OpenAI-compatible endpoint. This gives
  first-class support for the `response_format={"type": "json_schema", "json_schema": {...}, "strict": true}`
  contract that Mistral models on OpenRouter honor reliably. Alternative (raw httpx + manual parsing)
  would be fragile for strict schema validation.
- **European-first + privacy**: Defaults to `mistralai/mistral-medium-3.5` (and Large 3 as stronger option).
  OpenRouter routes to Mistral's infra; we explicitly log every egress of conversation data
  (required for GDPR accountability when processing Swedish callcenter PII).
- Hybrid architecture support: The client itself never decides "when" to call – that is profile/pipeline
  decision. Client only provides reliable, cached, retrying, observable execution + graceful error surfacing
  so callers can always fallback to local hybrid (analysis/llm_judge.py + heuristics).
- Caching is content-addressable on (model + task + full messages hash) so re-analysis of same transcript
  (very common in dev, eval, re-runs) incurs near-zero marginal cost/latency. Cache lives under .cache/llm/
  (gitignored) and survives process restarts.
- Cost control hooks: every call returns (result, meta) where meta contains tokens + approx_cost_usd.
  Later (Fas 3.4) profile `llm.cost_budget_per_call` can be enforced here or in analyzer.
- Error model: raises LLMError (subclass of BaseAnalysisError) only after retries. Callers catch and
  fallback. This matches the error style in transcription/ and core/errors.py.
- Lazy OpenAI client construction + optional dependency handling (like whisperx) so the rest of the
  app runs without `pip install openai` until the Mistral path is actually activated.
- Full type hints, structured logging, no global state.

Callcenter goal connection:
Local per-segment models (XLM-R sentiment, lexicon, trajectory heuristics) are excellent for speed and
offline/privacy, but miss cross-turn causality, sarcasm, implicit root causes and high-quality
"what should QA coach the agent on?" recommendations. Mistral Medium 3.5 via this client supplies
exactly that holistisk layer when the call profile or --deep-analysis requests it, while preserving
the local results as the merge base (see pipeline integration in 3.2.2).

Never send data externally without the user's explicit profile/flag activation and the prominent
log line below.

Usage (after 3.1.2+):
    from src.llm.openrouter_client import OpenRouterClient
    client = OpenRouterClient()
    # Key resolution order:
    #   1. api_key= passed to constructor
    #   2. OPENROUTER_API_KEY environment variable (recommended for prod/CI)
    #   3. Dev convenience files (gitignored): configs/openrouter.key, OPENROUTER_API_KEY.txt, etc.
    #      (only for local development - a loud warning is logged)
    data, meta = client.structured_chat(
        messages=[{"role": "system", ...}, {"role": "user", ...}],
        json_schema=the_pydantic_model_json_schema,
        task_name="actionable_summary",
        transcript_hash=some_sha_for_keying,
    )
    # data is validated JSON from Mistral; meta has model, cost_usd, cached etc.
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


# =============================================================================
# Public helper functions for convenient key loading (dev + dashboard override)
# =============================================================================

def _read_key_file(path: Path) -> str | None:
    """Internal: read a single potential key file, strip BOM and whitespace."""
    try:
        if path.exists() and path.is_file():
            raw = path.read_text(encoding="utf-8")
            content = raw.strip().lstrip("\ufeff").strip()
            if content and content.startswith("sk-or-"):
                return content
    except Exception as e:
        logger.debug("Could not read key file %s: %s", path, e)
    return None


def load_openrouter_key_from_file(
    key_file: str | Path | None = None,
    *,
    set_as_env: bool = True,
    silent: bool = False,
) -> str | None:
    """
    Load OpenRouter API key from a dev key file (for local development convenience).

    Args:
        key_file: Explicit path. If None, tries standard locations.
        set_as_env: If True, also does os.environ["OPENROUTER_API_KEY"] = key
        silent: If True, do not log the security warning.

    Returns:
        The key string, or None if not found / invalid.
    """
    candidates: list[Path] = []

    if key_file:
        candidates.append(Path(key_file))
    else:
        # Standard dev locations (must be in .gitignore!)
        candidates.extend([
            Path("configs") / "openrouter.key",
            Path("openrouter.key"),
            Path("OPENROUTER_API_KEY.txt"),
            Path(".openrouter_key"),
            Path("configs") / "openrouter_api_key.txt",
        ])

    key: str | None = None
    used_path: Path | None = None

    for p in candidates:
        loaded = _read_key_file(p)
        if loaded:
            key = loaded
            used_path = p
            break

    if key:
        if set_as_env:
            os.environ["OPENROUTER_API_KEY"] = key
        if not silent:
            logger.warning(
                "SECURITY: Loaded OpenRouter API key from FILE: %s. "
                "DEV USE ONLY. In production/CI always use the OPENROUTER_API_KEY environment variable.",
                used_path,
            )
        return key

    return None


def get_openrouter_api_key(override: str | None = None) -> str | None:
    """
    Resolve the effective OpenRouter API key with full priority:

    1. `override` (e.g. from Streamlit dashboard UI) - highest priority
    2. Environment variable OPENROUTER_API_KEY
    3. Dev key file (configs/openrouter.key etc.) - only for local development

    This is the recommended function to call from application code / dashboard
    when you want to support runtime override of the key.
    """
    if override and str(override).strip():
        return str(override).strip()

    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key

    # Try file but do NOT auto-set env here (let caller decide)
    return load_openrouter_key_from_file(set_as_env=False, silent=True)


# Lazy / optional dependency handling (pattern matched from src/transcription/whisperx.py and factory.py)
try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError  # type: ignore

    _HAS_OPENAI = True
except Exception:  # pragma: no cover - import guard
    OpenAI = None  # type: ignore
    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    _HAS_OPENAI = False


class OpenRouterClient:
    """Reliable OpenRouter client tuned for Mistral strict-schema holistic call analysis.

    See module docstring for full rationale, privacy notes and callcenter context.
    """

    # Pricing as of plan date / OpenRouter listings (double-checked via search; update as models evolve).
    # Values are USD per token. Used for approx cost reporting only.
    PRICING: dict[str, dict[str, float]] = {
        "mistralai/mistral-medium-3.5": {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
        "mistralai/mistral-medium-3-5": {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},  # OpenRouter slug variant
        "mistralai/mistral-large-3": {"input": 2.00 / 1_000_000, "output": 6.00 / 1_000_000},
        "mistralai/mistral-large-2512": {"input": 2.00 / 1_000_000, "output": 6.00 / 1_000_000},
        "default": {"input": 1.50 / 1_000_000, "output": 7.50 / 1_000_000},
    }

    DEFAULT_MODEL = "mistralai/mistral-medium-3.5"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_CACHE_DIR = Path(".cache") / "llm"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        enable_cache: bool = True,
    ) -> None:
        """
        Initialize client (does not create OpenAI instance until first use = lazy).

        Key loading order (for convenience in dev):
            1. Explicit `api_key` argument
            2. OPENROUTER_API_KEY environment variable (strongly preferred)
            3. Local gitignored key file (configs/openrouter.key, OPENROUTER_API_KEY.txt, etc.)
               → Only for local development. A prominent security warning is logged.
        """
        # Use the central resolver that supports UI override + env + file
        self.api_key = get_openrouter_api_key(api_key)

        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.enable_cache = enable_cache
        self.cost_budget: float | None = None  # settable post-init or via profile

        if cache_dir is None:
            self.cache_dir = self.DEFAULT_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client: OpenAI | None = None  # lazy

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set. Calls to Mistral via OpenRouter will raise LLMError "
                "unless api_key is passed at call time. Local fallback paths in pipeline/analyzer remain available."
            )

    @staticmethod
    def _try_load_key_from_file() -> str | None:
        """
        Deprecated internal method.

        Use the public helpers instead:
            - load_openrouter_key_from_file()
            - get_openrouter_api_key(override=...)
        """
        # Delegate to the new public implementation for backward compatibility
        return load_openrouter_key_from_file(set_as_env=False, silent=True)

    def _ensure_openai(self) -> OpenAI:
        """Lazy load the OpenAI client (and enforce optional dep)."""
        if not _HAS_OPENAI or OpenAI is None:
            raise ImportError(
                "openai package is required for Mistral/OpenRouter LLM features. "
                "Install with: pip install 'openai>=1.30'  (or add to requirements and pip install -r requirements.txt). "
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
        except Exception:  # never let cost calc break the flow
            return None

    def _make_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None,
        task_name: str,
        transcript_hash: str | None,
    ) -> str:
        """Content-addressable key. Includes everything that affects the LLM output."""
        payload = {
            "model": model,
            "task": task_name,
            "transcript_hash": transcript_hash or "",
            "messages": messages,  # order matters for prompt; caller should be stable
            "schema_name": (json_schema or {}).get("name") if json_schema else None,
            # schema hash would be overkill; full schema rarely changes per task
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

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
            logger.debug("LLM cache hit for key=%s (task=%s)", key[:12], meta.get("task"))
            return result, meta
        except Exception as e:  # corrupted cache is non-fatal
            logger.warning("Failed to read LLM cache %s: %s. Will recompute.", path, e)
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
            "meta": {k: v for k, v in meta.items() if k != "cached"},  # don't persist the transient flag
        }
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            logger.debug("LLM cache written key=%s", key[:12])
        except Exception as e:
            logger.warning("Failed to write LLM cache %s: %s", path, e)

    def _build_response_format(
        self, json_schema: dict[str, Any] | None, task_name: str
    ) -> dict[str, Any] | None:
        if not json_schema:
            return None
        # OpenRouter / Mistral expects this exact shape for guaranteed structured outputs
        return {
            "type": "json_schema",
            "json_schema": {
                "name": task_name or "call_analysis",
                "strict": True,
                "schema": json_schema,
            },
        }

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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute a chat completion with optional strict JSON schema.

        Returns:
            (parsed_result_dict, meta_dict)
            meta contains: model, usage, cost_usd, cached, task, tokens_used etc.

        Raises:
            LLMError after exhausting retries (caller must fallback).
            ImportError if openai not installed and this path is hit.
        """
        if not messages:
            raise ValueError("messages list must be non-empty")

        model = model or self.default_model
        client = self._ensure_openai()

        # Privacy / GDPR audit log – this is non-negotiable per plan rule 5
        transcript_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "EXTERNAL LLM CALL (OpenRouter/Mistral) | model=%s | task=%s | chars≈%d | "
            "Data (full conversation transcript + roles) is being sent to a third-party service. "
            "This is only done when the callcenter profile / --use-mistral-llm / deep_analysis enables it. "
            "See docs for PII redaction options (Fas 3.4).",
            model,
            task_name,
            transcript_len,
        )

        cache_key = self._make_cache_key(model, messages, json_schema, task_name, transcript_hash)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            result, meta = cached
            meta = {**meta, "cost_usd": 0.0}  # cached hits are free; override any stored cost
            return result, meta

        response_format = self._build_response_format(json_schema, task_name)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]  # SDK accepts list of dicts
                    response_format=response_format,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/idealinvestse/Automatisk-sentimentanalys",
                        "X-Title": "Automatisk-sentimentanalys (Mistral holistisk)",
                        **(extra_headers or {}),
                    },
                )
                latency = round(time.time() - t0, 3)

                # Extract content (strict mode => should always be valid JSON string)
                choice = completion.choices[0]
                content = choice.message.content or "{}"
                try:
                    result: dict[str, Any] = json.loads(content)
                except json.JSONDecodeError as je:
                    # This should be extremely rare with strict:true; still protect
                    logger.error("Mistral returned non-JSON despite strict schema: %s", content[:200])
                    raise LLMError(f"Strict JSON parse failed: {je}") from je

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
                    "provider": "openrouter",
                }
                if "x-ratelimit" in str(completion):  # best effort
                    meta["rate_limit_headers"] = "present"

                self._save_to_cache(cache_key, result, meta)

                # Cost budget warning (Fas 3.4)
                if self.cost_budget and (cost or 0) > self.cost_budget:
                    logger.warning(
                        "LLM cost budget exceeded for task=%s: $%.5f > budget $%.5f. "
                        "Consider stronger caching, shorter context, or cheaper model.",
                        task_name, cost or 0, self.cost_budget
                    )
                    meta["budget_exceeded"] = True
                    meta["budget"] = self.cost_budget

                logger.info(
                    "Mistral call OK | model=%s | task=%s | cost≈$%.5f | latency=%.2fs | cached=False",
                    model,
                    task_name,
                    cost or 0.0,
                    latency,
                )
                return result, meta

            except RateLimitError as e:
                last_exc = e
                wait = (2 ** attempt) * 1.2 + 0.5
                logger.warning(
                    "OpenRouter rate limit (attempt %d/%d). Sleeping %.1fs before retry. Error: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            except (APITimeoutError, APIError) as e:
                last_exc = e
                wait = (2 ** attempt) * 0.8
                logger.warning(
                    "OpenRouter transient error (attempt %d/%d): %s. Backing off %.1fs",
                    attempt + 1,
                    self.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            except Exception as e:
                # Non-retryable (auth, bad schema, etc.)
                last_exc = e
                logger.error("OpenRouter non-retryable failure: %s", e, exc_info=True)
                break

        # All retries exhausted
        raise LLMError(
            f"OpenRouter/Mistral call failed after {self.max_retries} attempts for task={task_name} model={model}. "
            f"Last error: {last_exc}. Caller should fallback to local analysis."
        ) from last_exc

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Plain text chat completion (no schema enforcement). Returns (content, meta)."""
        model = model or self.default_model
        client = self._ensure_openai()

        logger.info("EXTERNAL LLM (plain) call to %s", model)
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
            }
            return content, meta
        except Exception as e:
            raise LLMError(f"Plain chat_completion failed: {e}") from e

    def clear_cache(self) -> int:
        """Delete all cached LLM responses. Returns number of files removed. Useful for testing/privacy."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        logger.info("Cleared %d LLM cache files from %s", count, self.cache_dir)
        return count
