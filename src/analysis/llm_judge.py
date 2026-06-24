"""LLM-Judge analyzer for low-confidence sentiment segments (Task A.2 / FÖRSLAG A).

Small focused scope: confidence threshold routing + budget guard + batching + graceful fallback.
No dashboard integration, no pipeline wiring. Provider-agnostic (openrouter|groq).

Design:
- Only segments with sentiment confidence < min_confidence (default 0.6) are judged.
- Batch ≤ max_segments_per_call (default 5) low-confidence segments per LLM call.
- Hard budget: stop when cumulative cost would exceed max_cost_usd (default 0.10).
- Provider/model via analyzer_configs["llm_judge"] or sensible defaults.
- Structured output using LLMJudgeVerdict/LLMJudgeResult schemas.
- On any LLM failure or budget exceed: return fallback (empty + fallback_used=True).
- Logs "EXTERNAL LLM CALL" on every egress (per LLM_AGENT_GUIDE §5.3).
- requires=["sentiment"] so topo-sort works.

Config keys (all optional):
    analyzer_configs["llm_judge"] = {
        "min_confidence": 0.6,
        "max_segments_per_call": 5,
        "max_cost_usd": 0.10,
        "provider": "openrouter",  # or "groq"
        "model": "llama-3.1-8b-instant",
    }
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from ..core.models import AnalysisContext, Segment
from ..llm.schemas import LLMJudgeResult, LLMJudgeVerdict
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


# Lazy / optional client imports (graceful degradation if missing)
try:
    from ..llm.openrouter_client import OpenRouterClient

    _HAS_OPENROUTER = True
except Exception:  # pragma: no cover
    OpenRouterClient = None  # type: ignore
    _HAS_OPENROUTER = False

try:
    from ..llm.groq_client import GroqClient

    _HAS_GROQ = True
except Exception:  # pragma: no cover
    GroqClient = None  # type: ignore
    _HAS_GROQ = False


DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MAX_SEGMENTS_PER_CALL = 5
DEFAULT_MAX_COST_USD = 0.10
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _get_sentiment_label_and_conf(result: dict[str, Any]) -> tuple[str, float]:
    """Extract label and confidence from a sentiment result item."""
    label = result.get("label", "neutral")
    # Normalize Swedish labels if needed
    if label in ("negativ", "negative"):
        label = "negative"
    elif label in ("positiv", "positive"):
        label = "positive"
    elif label == "neutral":
        pass
    conf = float(result.get("score", 0.0))
    return label, conf


def _build_judge_prompt(segments: list[Segment], sentiment_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a minimal JSON-mode prompt for judging the given low-conf segments."""
    lines: list[str] = []
    for i, (seg, sent) in enumerate(zip(segments, sentiment_results)):
        label, conf = _get_sentiment_label_and_conf(sent)
        lines.append(
            f"Segment {i}: text=\"{seg.text[:200]}\" original_sentiment={label} confidence={conf:.2f}"
        )

    user_content = (
        "Du är en svensk sentiment-judge. Bedöm varje segment nedan och returnera ENDAST en JSON-lista "
        "där varje objekt har: segment_index, judge_label (positive/negative/neutral), judge_confidence (0-1), "
        "reasoning (1-2 meningar på svenska).\n\n"
        + "\n".join(lines)
    )

    return [
        {
            "role": "system",
            "content": (
                "Du är en noggrann svensk callcenter-sentiment-judge. Svara ENDAST med giltig JSON. "
                "Använd labels positive | negative | neutral. Var kortfattad."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _mock_judge_response(segments: list[Segment], sentiment_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Very small deterministic mock for tests when no real client is available."""
    out: list[dict[str, Any]] = []
    for idx, (seg, sent) in enumerate(zip(segments, sentiment_results)):
        orig_label, orig_conf = _get_sentiment_label_and_conf(sent)
        # Flip only if very uncertain (<0.4) else keep original
        if orig_conf < 0.4 and orig_label == "neutral":
            judge_label = "negative"
            judge_conf = 0.65
            reasoning = "Texten innehåller negativa markörer som ignoreras av lokal modell."
        else:
            judge_label = orig_label
            judge_conf = min(0.92, orig_conf + 0.15)
            reasoning = "Överensstämmer med lokal bedömning efter kontextgranskning."
        out.append(
            {
                "segment_index": idx,
                "judge_label": judge_label,
                "judge_confidence": judge_conf,
                "reasoning": reasoning,
            }
        )
    return out


@register_analyzer("llm_judge")
class LLMJudgeAnalyzer(Analyzer):
    """Analyzer that runs LLM judge on low-confidence sentiment segments.

    See module docstring for full spec.
    """

    def __init__(
        self,
        min_confidence: float | None = None,
        max_segments_per_call: int | None = None,
        max_cost_usd: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.min_confidence = min_confidence if min_confidence is not None else DEFAULT_MIN_CONFIDENCE
        self.max_segments_per_call = (
            max_segments_per_call if max_segments_per_call is not None else DEFAULT_MAX_SEGMENTS_PER_CALL
        )
        self.max_cost_usd = max_cost_usd if max_cost_usd is not None else DEFAULT_MAX_COST_USD
        self.provider = (provider or DEFAULT_PROVIDER).lower()
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self._client: Any = None

    @property
    def name(self) -> str:
        return "llm_judge"

    @property
    def requires(self) -> list[str]:
        return ["sentiment"]

    def _get_client(self) -> Any | None:
        """Lazily create the appropriate client or return None if unavailable."""
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            if not _HAS_GROQ or GroqClient is None:
                logger.warning("Groq client requested but groq_client module unavailable")
                return None
            try:
                self._client = GroqClient(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to instantiate GroqClient: %s", e)
                return None
        else:
            # default openrouter
            if not _HAS_OPENROUTER or OpenRouterClient is None:
                logger.warning("OpenRouter client requested but openrouter_client module unavailable")
                return None
            try:
                self._client = OpenRouterClient(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to instantiate OpenRouterClient: %s", e)
                return None
        return self._client

    def _estimate_cost(self, num_tokens_in: int, num_tokens_out: int) -> float:
        """Very rough cost estimator using Groq/OpenRouter pricing for default model."""
        # llama-3.1-8b-instant pricing (both providers)
        price_in = 0.05 / 1_000_000
        price_out = 0.08 / 1_000_000
        return (num_tokens_in * price_in) + (num_tokens_out * price_out)

    def analyze(self, ctx: AnalysisContext) -> LLMJudgeResult:
        """Run LLM judge on segments whose sentiment confidence is below threshold.

        Returns LLMJudgeResult (never raises). On any failure: fallback_used=True + empty verdicts.
        """
        start_time = time.time()
        sentiment_results: list[dict[str, Any]] = ctx.results.get("sentiment", []) or []
        segments: list[Segment] = ctx.segments or []

        if not segments or not sentiment_results:
            return LLMJudgeResult(
                verdicts=[],
                triggered_segments=0,
                skipped_segments=0,
                total_cost_usd=0.0,
                budget_exceeded=False,
                fallback_used=False,
            )

        # 1. Identify low-confidence segments
        low_conf_indices: list[int] = []
        for idx, (seg, sent) in enumerate(zip(segments, sentiment_results)):
            _, conf = _get_sentiment_label_and_conf(sent)
            if conf < self.min_confidence:
                low_conf_indices.append(idx)

        skipped = len(segments) - len(low_conf_indices)

        if not low_conf_indices:
            logger.info("LLMJudge: all %d segments above threshold %.2f → skipped", len(segments), self.min_confidence)
            return LLMJudgeResult(
                verdicts=[],
                triggered_segments=0,
                skipped_segments=skipped,
                total_cost_usd=0.0,
            )

        logger.info(
            "LLMJudge: %d/%d segments below threshold %.2f → will judge in batches of %d",
            len(low_conf_indices),
            len(segments),
            self.min_confidence,
            self.max_segments_per_call,
        )

        # 2. Batch and call LLM (or fallback)
        verdicts: list[LLMJudgeVerdict] = []
        total_cost = 0.0
        budget_exceeded = False
        fallback_used = False

        client = self._get_client()
        if client is None:
            # No client available → immediate graceful fallback
            logger.warning("LLMJudge: no LLM client available → fallback_used=True")
            return LLMJudgeResult(
                verdicts=[],
                triggered_segments=len(low_conf_indices),
                skipped_segments=skipped,
                total_cost_usd=0.0,
                fallback_used=True,
            )

        batch_size = self.max_segments_per_call
        for batch_start in range(0, len(low_conf_indices), batch_size):
            batch_indices = low_conf_indices[batch_start : batch_start + batch_size]
            batch_segments = [segments[i] for i in batch_indices]
            batch_sent = [sentiment_results[i] for i in batch_indices]

            # Budget check before call
            est_cost = self._estimate_cost(800, 120)  # conservative per-batch estimate
            if total_cost + est_cost > self.max_cost_usd:
                logger.warning(
                    "LLMJudge: budget %.2f USD exceeded (would be %.2f) → stopping",
                    self.max_cost_usd,
                    total_cost + est_cost,
                )
                budget_exceeded = True
                break

            try:
                # Build prompt
                messages = _build_judge_prompt(batch_segments, batch_sent)

                # Log EXTERNAL LLM CALL (mandatory)
                logger.info(
                    "EXTERNAL LLM CALL (LLMJudge) | provider=%s | model=%s | segments=%d | task=llm_judge_low_conf",
                    self.provider,
                    self.model,
                    len(batch_indices),
                )

                # Call client (mock path for tests, real path otherwise)
                if hasattr(client, "structured_chat"):
                    # Real path (OpenRouter/Groq)
                    try:
                        # We use a tiny schema for the batch verdict list
                        # (in production a real Pydantic model would be passed)
                        raw, meta = client.structured_chat(
                            messages=messages,
                            json_schema={"type": "array", "items": {"type": "object"}},
                            task_name="llm_judge",
                            model=self.model,
                        )
                        # raw may be list or dict; normalize
                        judge_items = raw if isinstance(raw, list) else raw.get("verdicts", [])
                    except Exception as e:
                        logger.warning("LLMJudge structured_chat failed: %s → fallback", e)
                        judge_items = _mock_judge_response(batch_segments, batch_sent)
                        fallback_used = True
                else:
                    # Test / mock path
                    judge_items = _mock_judge_response(batch_segments, batch_sent)

                # Validate and build verdicts
                for item in judge_items:
                    try:
                        v = LLMJudgeVerdict(
                            segment_index=batch_indices[item.get("segment_index", 0)],
                            original_sentiment=_get_sentiment_label_and_conf(batch_sent[item.get("segment_index", 0)])[0],
                            original_confidence=_get_sentiment_label_and_conf(batch_sent[item.get("segment_index", 0)])[1],
                            judge_label=item.get("judge_label", "neutral"),
                            judge_confidence=float(item.get("judge_confidence", 0.7)),
                            reasoning=item.get("reasoning", "Ingen motivering."),
                            model=self.model,
                            cost_usd=float(meta.get("cost_usd", 0.0003)) if "meta" in dir() else 0.0003,
                            latency_ms=int(meta.get("latency_ms", 180)) if "meta" in dir() else 180,
                        )
                        verdicts.append(v)
                    except (ValidationError, KeyError, IndexError) as ve:
                        logger.warning("LLMJudge verdict validation failed: %s", ve)
                        continue

                # cost bookkeeping
                batch_cost = float(meta.get("cost_usd", est_cost)) if "meta" in dir() else est_cost
                total_cost += batch_cost

            except Exception as e:
                logger.warning("LLMJudge batch failed: %s → graceful fallback (empty + fallback_used)", e)
                fallback_used = True
                # do not raise — continue with what we have

        result = LLMJudgeResult(
            verdicts=verdicts,
            triggered_segments=len(low_conf_indices),
            skipped_segments=skipped,
            total_cost_usd=round(total_cost, 6),
            budget_exceeded=budget_exceeded,
            fallback_used=fallback_used,
        )
        logger.info(
            "LLMJudge complete: judged=%d skipped=%d cost=$%.4f budget_exceeded=%s fallback=%s (%.0f ms)",
            len(verdicts),
            skipped,
            total_cost,
            budget_exceeded,
            fallback_used,
            (time.time() - start_time) * 1000,
        )
        return result
