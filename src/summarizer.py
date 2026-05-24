"""Automatic call summarization and action item extraction for Swedish calls.

Generates:
    - Short summary (max 5 sentences)
    - Action items with responsible party (agent/kund)
    - Key topics mentioned

Uses extractive summarization by default (no model required).
Optionally supports model-based summarization via HuggingFace models.

Usage:
    from src.summarizer import CallSummarizer
    cs = CallSummarizer()
    result = cs.summarize(segments, diarization)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """An action item extracted from a call."""

    description: str
    responsible: str  # "agent" or "kund" or "both"
    priority: str = "medium"  # "low", "medium", "high", "critical"
    source_segment: str = ""


@dataclass
class CallSummary:
    """Complete call summary."""

    summary: str = ""
    summary_sentences: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    call_outcome: str = ""  # "resolved", "pending", "escalated", "unclear"
    overall_sentiment: str = ""  # "positiv", "neutral", "negativ"
    duration_summary: str = ""
    backend: str = "extractive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "summary_sentences": self.summary_sentences,
            "action_items": [
                {"description": a.description, "responsible": a.responsible, "priority": a.priority}
                for a in self.action_items
            ],
            "key_topics": self.key_topics,
            "call_outcome": self.call_outcome,
            "overall_sentiment": self.overall_sentiment,
            "duration_summary": self.duration_summary,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Action item patterns (Swedish)
# ---------------------------------------------------------------------------
ACTION_PATTERNS_AGENT: list[tuple[str, str]] = [
    ("ska återkomma", "agent"),
    ("skickar information", "agent"),
    ("lägger en beställning", "agent"),
    ("registrerar ärendet", "agent"),
    ("eskalerar till", "agent"),
    ("bokar in", "agent"),
    ("återkopplar", "agent"),
    ("kontrollerar", "agent"),
    ("undersöker", "agent"),
    ("krediterar", "agent"),
]

ACTION_PATTERNS_KUND: list[tuple[str, str]] = [
    ("ska skicka in", "kund"),
    ("återkommer med", "kund"),
    ("skickar dokument", "kund"),
    ("betalar fakturan", "kund"),
    ("testar och återkommer", "kund"),
    ("inväntar leverans", "kund"),
    ("loggar in och testar", "kund"),
    ("fyller i formuläret", "kund"),
]

OUTCOME_PATTERNS: list[tuple[str, str]] = [
    ("löste problemet", "resolved"),
    ("är löst", "resolved"),
    ("fungerar nu", "resolved"),
    ("klart", "resolved"),
    ("ordnade sig", "resolved"),
    ("återkommer imorgon", "pending"),
    ("inväntar svar", "pending"),
    ("väntar på", "pending"),
    ("eskalerat", "escalated"),
    ("vidarebefordrat", "escalated"),
    ("chef", "escalated"),
]


# ---------------------------------------------------------------------------
# Swedish stop words for keyword extraction
# ---------------------------------------------------------------------------
SWEDISH_STOP_WORDS: set[str] = {
    "och",
    "att",
    "det",
    "som",
    "en",
    "på",
    "är",
    "av",
    "för",
    "med",
    "till",
    "den",
    "har",
    "de",
    "inte",
    "om",
    "ett",
    "han",
    "men",
    "vi",
    "du",
    "hon",
    "jag",
    "sig",
    "från",
    "var",
    "så",
    "kan",
    "man",
    "när",
    "skulle",
    "eller",
    "då",
    "nu",
    "ska",
    "också",
    "får",
    "få",
    "efter",
    "upp",
    "ut",
    "in",
    "över",
    "under",
    "vid",
    "mot",
    "genom",
    "utan",
    "än",
    "ju",
    "väl",
    "nog",
    "bara",
    "här",
    "där",
    "hur",
    "vad",
    "vilken",
    "vilket",
    "denna",
    "detta",
    "dessa",
    "min",
    "din",
    "sin",
    "vår",
    "er",
    "deras",
    "mycket",
    "mer",
    "mest",
    "lite",
    "mindre",
    "minst",
    "allt",
    "alla",
    "någon",
    "något",
    "några",
    "ingen",
    "inget",
    "inga",
    "samma",
    "själv",
    "redan",
    "alltid",
    "aldrig",
    "ofta",
    "sällan",
    "ibland",
    "kanske",
}


class CallSummarizer:
    """Summarize Swedish call center conversations.

    Args:
        backend: 'extractive' (default, no model) or 'model' (requires HF model).
        model_name: HuggingFace model for summarization (for 'model' backend).
        max_summary_sentences: Maximum number of sentences in summary.
    """

    def __init__(
        self,
        backend: str = "extractive",
        model_name: str = "KBLab/bart-base-swedish-cased",
        max_summary_sentences: int = 5,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.max_summary_sentences = max_summary_sentences
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summarize(
        self,
        segments: list[dict[str, Any]],
        diarization: dict[str, Any] | None = None,
        sentiment_results: list[dict[str, Any]] | None = None,
        intent_results: list[tuple[str, float]] | None = None,
    ) -> CallSummary:
        """Generate a complete call summary.

        Args:
            segments: ASR transcript segments with optional 'speaker' key.
            diarization: Optional diarization result dict.
            sentiment_results: Optional per-segment sentiment results.
            intent_results: Optional per-segment intent classifications.

        Returns:
            CallSummary with summary, action items, key topics, and outcome.
        """
        # Build full transcript
        full_text = " ".join(s.get("text", "") for s in segments)

        # Extractive summary
        summary_sentences = self._extractive_summary(segments)

        # Action items
        action_items = self._extract_action_items(full_text, segments)

        # Key topics
        key_topics = self._extract_key_topics(full_text)

        # Call outcome
        call_outcome = self._determine_outcome(full_text, sentiment_results)

        # Overall sentiment
        overall_sentiment = self._overall_sentiment(sentiment_results)

        # Duration info
        duration_summary = self._duration_info(segments, diarization)

        return CallSummary(
            summary=" ".join(summary_sentences),
            summary_sentences=summary_sentences,
            action_items=action_items,
            key_topics=key_topics,
            call_outcome=call_outcome,
            overall_sentiment=overall_sentiment,
            duration_summary=duration_summary,
            backend=self.backend,
        )

    # ------------------------------------------------------------------
    # Extractive summarization
    # ------------------------------------------------------------------
    def _extractive_summary(self, segments: list[dict[str, Any]]) -> list[str]:
        """Extract the most important sentences from the transcript."""
        if not segments:
            return ["Ingen transkribering tillgänglig."]

        # Score each segment/sentence by length and position
        scored: list[tuple[str, float]] = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text or len(text) < 10:
                continue

            # Simple heuristic: longer sentences near start/end are more important
            score = min(1.0, len(text) / 200.0)  # length bonus
            score += 0.3 if i < 3 else 0  # opening statements
            score += 0.2 if i > len(segments) - 3 else 0  # closing statements
            score += 0.15 if seg.get("speaker", "") not in ("", "UNKNOWN") else 0

            scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.max_summary_sentences]
        # Sort back by original order using index stored during scoring
        indexed_top = [(i, t) for i, (t, s) in enumerate(scored) if t in {x[0] for x in top}]
        indexed_top.sort(key=lambda x: x[0])
        return [t for _, t in indexed_top[: self.max_summary_sentences]]

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------
    def _extract_action_items(
        self, full_text: str, segments: list[dict[str, Any]]
    ) -> list[ActionItem]:
        """Extract action items from the conversation."""
        items: list[ActionItem] = []
        lowered = full_text.lower()

        for pattern, responsible in ACTION_PATTERNS_AGENT:
            if pattern in lowered:
                # Find the segment containing this pattern
                source = ""
                for seg in segments:
                    if pattern in seg.get("text", "").lower():
                        source = seg.get("text", "")
                        break
                items.append(
                    ActionItem(
                        description=f"Agent {pattern}",
                        responsible=responsible,
                        priority="medium",
                        source_segment=source[:200],
                    )
                )

        for pattern, responsible in ACTION_PATTERNS_KUND:
            if pattern in lowered:
                items.append(
                    ActionItem(
                        description=f"Kund {pattern}",
                        responsible=responsible,
                        priority="low",
                    )
                )

        # Deduplicate
        seen = set()
        unique: list[ActionItem] = []
        for item in items:
            if item.description not in seen:
                seen.add(item.description)
                unique.append(item)

        return unique[:5]

    # ------------------------------------------------------------------
    # Key topics
    # ------------------------------------------------------------------
    def _extract_key_topics(self, full_text: str) -> list[str]:
        """Extract key topics using TF-like word frequency."""
        words = re.findall(r"[\wäöåÄÖÅ]+", full_text.lower())
        filtered = [w for w in words if w not in SWEDISH_STOP_WORDS and len(w) > 2]

        freq = Counter(filtered)
        # Return top words, excluding very common ones
        top = freq.most_common(20)
        return [w for w, c in top if c >= 2][:8]

    # ------------------------------------------------------------------
    # Call outcome
    # ------------------------------------------------------------------
    def _determine_outcome(
        self, full_text: str, sentiment_results: list[dict[str, Any]] | None
    ) -> str:
        """Determine the call outcome."""
        lowered = full_text.lower()

        for pattern, outcome in OUTCOME_PATTERNS:
            if pattern in lowered:
                return outcome

        # If sentiment improved during the call, likely resolved
        if sentiment_results and len(sentiment_results) >= 2:
            first_half = sentiment_results[: len(sentiment_results) // 2]
            second_half = sentiment_results[len(sentiment_results) // 2 :]
            first_pos = sum(1 for r in first_half if r.get("label") == "positiv")
            second_pos = sum(1 for r in second_half if r.get("label") == "positiv")
            if second_pos > first_pos:
                return "resolved"
            if second_pos < first_pos:
                return "pending"

        return "unclear"

    # ------------------------------------------------------------------
    # Overall sentiment
    # ------------------------------------------------------------------
    def _overall_sentiment(self, sentiment_results: list[dict[str, Any]] | None) -> str:
        """Compute overall sentiment from per-segment results."""
        if not sentiment_results:
            return "neutral"

        counts: Counter[str] = Counter()
        for r in sentiment_results:
            label = r.get("label", "neutral")
            counts[label] += 1

        return counts.most_common(1)[0][0] if counts else "neutral"

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------
    def _duration_info(
        self, segments: list[dict[str, Any]], diarization: dict[str, Any] | None
    ) -> str:
        """Build a human-readable duration summary."""
        if not segments:
            return ""

        start = segments[0].get("start", 0) or 0
        end = segments[-1].get("end", start + 1) or start + 1
        dur = end - start

        parts = [f"Samtalslängd: {dur:.0f} sekunder"]

        if diarization and diarization.get("segments"):
            speakers = set()
            for s in diarization["segments"]:
                speakers.add(s.get("speaker", "UNKNOWN"))
            parts.append(f"Antal talare: {len(speakers)}")

        if diarization and diarization.get("speakers"):
            for sp in diarization["speakers"]:
                timeline = [
                    (s["start"], s["end"])
                    for s in diarization["segments"]
                    if s.get("speaker") == sp
                ]
                if timeline:
                    sp_dur = sum(e - s for s, e in timeline)
                    pct = (sp_dur / dur * 100) if dur > 0 else 0
                    parts.append(f"{sp}: {pct:.0f}% av samtalet")

        return " | ".join(parts)


__all__ = ["CallSummarizer", "CallSummary", "ActionItem"]
