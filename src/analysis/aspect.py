"""Aspect-Based Sentiment Analysis (ABSA) analyzer for call center use.

Implements Task 1.5 using a lightweight hybrid approach:
- Keyword / phrase triggers for callcenter aspects (Swedish + English).
- Runs the existing sentiment pipeline on the evidence span (or full segment).
- Produces structured output with aspect, sentiment, score, evidence text, timestamps.

This matches the project principle of "hybrid först" – small efficient models + heuristics
before considering heavier zero-shot models.

Registered as "aspect" so it participates in the topological analyzer execution
and appears in CallAnalysisReport.results["aspect"].

Callcenter aspects (initial set):
  kundtjänst_kvalitet, teknisk_lösning, fakturering_pris, väntetid,
  agent_attityd, produkt_kvalitet, uppföljning, annat

The output is additive and does not break existing CallAnalysisReport structure.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..core.models import AnalysisContext, Segment
from .base import Analyzer
from .registry import register_analyzer
from .sentiment import SentimentAnalyzer  # reuse the existing sentiment logic

logger = logging.getLogger(__name__)

# Swedish-focused call center aspect triggers (can be extended via profile later)
ASPECT_TRIGGERS: dict[str, list[str]] = {
    "fakturering_pris": [
        "faktura", "fakturering", "betala", "betalning", "pris", "kostar", "kostnad",
        "pengar", "återbetalning", "kredit", "debiter", "moms", "avgift", "räkning",
        "fakturadatum", "betalningsvillkor"
    ],
    "kundtjänst_kvalitet": [
        "kundtjänst", "support", "service", "hjälp", "bemötande", "svarstid",
        "väntetid", "kö", "chatt", "telefonsupport", "ärendehantering"
    ],
    "teknisk_lösning": [
        "teknisk", "problem", "fel", "bugg", "fungerar inte", "laddar inte",
        "app", "webb", "login", "lösenord", "uppkoppling", "internet", "bredband"
    ],
    "väntetid": [
        "väntetid", "kö", "vänta", "lång tid", "svarstid", "dröja", "långsam"
    ],
    "agent_attityd": [
        "attityd", "bemötande", "snäll", "oförskämd", "hjälpsam", "oartigt",
        "förstår", "beklagar", "tack", "empati", "lyssnar"
    ],
    "produkt_kvalitet": [
        "produkt", "kvalitet", "bra", "dålig", "funkar", "inte funkar", "defekt",
        "retur", "reklamation"
    ],
    "uppföljning": [
        "uppföljning", "återkoppling", "återkomma", "höra av sig", "kontakta igen",
        "oppföljning", "follow up"
    ],
}

# Compile regex for faster matching (word boundaries, case insensitive)
_ASPECT_REGEX = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(t) for t in triggers) + r")\b", re.IGNORECASE)
    for aspect, triggers in ASPECT_TRIGGERS.items()
}


@register_analyzer("aspect")
class AspectAnalyzer(Analyzer):
    """Analyzer that extracts callcenter aspects with associated sentiment."""

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        # Reuse the sentiment analyzer (lazy inside)
        self._sentiment: SentimentAnalyzer | None = None

    @property
    def name(self) -> str:
        return "aspect"

    @property
    def requires(self) -> list[str]:
        return ["sentiment"]

    def _get_sentiment(self) -> SentimentAnalyzer:
        if self._sentiment is None:
            self._sentiment = SentimentAnalyzer(device=self.device)
        return self._sentiment

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        if not ctx.segments:
            return []

        results: list[dict[str, Any]] = []
        sentiment_results = ctx.results.get("sentiment") or []

        for seg_idx, seg in enumerate(ctx.segments):
            text = seg.text or ""
            if not text.strip():
                continue

            matched_aspects = self._extract_aspects(text)

            if not matched_aspects:
                continue

            sent: dict[str, Any] = {"label": "neutral", "score": 0.0}
            if seg_idx < len(sentiment_results) and isinstance(sentiment_results[seg_idx], dict):
                sent = sentiment_results[seg_idx]
            else:
                try:
                    mini_ctx = AnalysisContext(segments=[seg])
                    sent_list = self._get_sentiment().analyze(mini_ctx)
                    sent = sent_list[0] if sent_list else sent
                except Exception:
                    pass

            for aspect in matched_aspects:
                evidence = self._extract_evidence(text, aspect)
                results.append({
                    "aspect": aspect,
                    "sentiment": sent.get("label", "neutral"),
                    "score": float(sent.get("score", 0.0)),
                    "evidence": evidence,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": getattr(seg, "speaker", None),
                })

        logger.debug("ABSA found %d aspect mentions", len(results))
        return results

    def _extract_aspects(self, text: str) -> list[str]:
        found: list[str] = []
        for aspect, regex in _ASPECT_REGEX.items():
            if regex.search(text):
                found.append(aspect)
        # If nothing specific matched, optionally tag "annat" for long negative segments
        if not found and len(text.split()) > 8:
            # very naive fallback for "annat"
            found.append("annat")
        return found

    def _extract_evidence(self, text: str, aspect: str) -> str:
        """Return a short evidence span around the trigger (simple heuristic)."""
        regex = _ASPECT_REGEX.get(aspect)
        if not regex:
            return text[:120]

        match = regex.search(text)
        if not match:
            return text[:120]

        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 60)
        return text[start:end].strip()


# Also export the aspect list for profiles / docs
CALLCENTER_ASPECTS = list(ASPECT_TRIGGERS.keys()) + ["annat"]
