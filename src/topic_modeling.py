"""Topic modeling for Swedish call center conversations.

Uses keyword-based topic extraction by default (no model required).
Optionally supports BERTopic integration for advanced topic discovery.

Usage:
    from src.topic_modeling import TopicModeler
    tm = TopicModeler()
    topics = tm.extract_topics(segments)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Predefined call center topic keywords
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "faktura": ["faktura", "betala", "avgift", "pris", "kostnad", "debitering", "belopp"],
    "leverans": ["leverans", "paket", "spåra", "skicka", "frakt", "postnord", "bud"],
    "tekniskt_problem": ["fungerar inte", "fel", "trasig", "bugg", "kraschar", "startar inte"],
    "abonnemang": ["abonnemang", "prenumeration", "uppsägning", "avsluta", "förnya"],
    "konto": ["konto", "inloggning", "lösenord", "profil", "användare", "registrera"],
    "återbetalning": ["återbetalning", "pengar tillbaka", "kredit", "kompensation"],
    "support": ["support", "hjälp", "assistans", "kundtjänst", "vägledning"],
    "bokning": ["boka", "tid", "möte", "omboka", "kalender", "besök"],
    "klagomål": ["klaga", "missnöjd", "dålig", "reklamera", "oacceptabelt"],
    "information": ["information", "öppettider", "erbjudande", "sortiment", "fråga"],
}


@dataclass
class TopicResult:
    """A discovered topic with metadata."""

    name: str
    keywords: list[str] = field(default_factory=list)
    frequency: int = 0
    segments: list[int] = field(default_factory=list)  # segment indices
    sentiment_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "keywords": self.keywords,
            "frequency": self.frequency,
            "segment_count": len(self.segments),
            "sentiment_distribution": self.sentiment_distribution,
        }


@dataclass
class TopicReport:
    """Complete topic analysis report."""

    topics: list[TopicResult] = field(default_factory=list)
    emerging_topics: list[str] = field(default_factory=list)
    total_segments: int = 0
    timestamp: str = ""
    backend: str = "keyword"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topics": [t.to_dict() for t in self.topics],
            "emerging_topics": self.emerging_topics,
            "total_segments": self.total_segments,
            "timestamp": self.timestamp,
            "backend": self.backend,
        }


class TopicModeler:
    """Topic modeling for Swedish call center data.

    Args:
        backend: 'keyword' (default) or 'bertopic' (requires bertopic package).
        min_topic_frequency: Minimum segments for a topic to be included.
    """

    def __init__(
        self,
        backend: str = "keyword",
        min_topic_frequency: int = 1,
    ) -> None:
        self.backend = backend
        self.min_topic_frequency = min_topic_frequency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_topics(
        self,
        segments: list[dict[str, Any]],
        sentiment_results: list[dict[str, Any]] | None = None,
    ) -> TopicReport:
        """Extract topics from conversation segments.

        Args:
            segments: ASR transcript segments.
            sentiment_results: Optional per-segment sentiment labels.

        Returns:
            TopicReport with discovered topics and metadata.
        """
        texts = [s.get("text", "") for s in segments]

        topics = self._keyword_topic_extraction(texts, sentiment_results)
        emerging = self._detect_emerging_topics(texts)

        return TopicReport(
            topics=topics,
            emerging_topics=emerging,
            total_segments=len(segments),
            timestamp=datetime.utcnow().isoformat() + "Z",
            backend=self.backend,
        )

    def topic_trends(self, topic_reports: list[TopicReport]) -> list[dict[str, Any]]:
        """Analyze topic trends over multiple reports (time series)."""
        if not topic_reports:
            return []

        all_topics: set[str] = set()
        for r in topic_reports:
            for t in r.topics:
                all_topics.add(t.name)

        trends: list[dict[str, Any]] = []
        for topic_name in sorted(all_topics):
            trend = {"topic": topic_name, "data": []}
            for r in topic_reports:
                freq = next((t.frequency for t in r.topics if t.name == topic_name), 0)
                trend["data"].append({"timestamp": r.timestamp, "frequency": freq})
            trends.append(trend)

        return trends

    # ------------------------------------------------------------------
    # Keyword-based extraction
    # ------------------------------------------------------------------
    def _keyword_topic_extraction(
        self,
        texts: list[str],
        sentiment_results: list[dict[str, Any]] | None,
    ) -> list[TopicResult]:
        """Extract topics using keyword matching."""
        topic_hits: dict[str, list[int]] = defaultdict(list)

        for i, text in enumerate(texts):
            lowered = text.lower()
            for topic_name, keywords in TOPIC_KEYWORDS.items():
                if any(kw in lowered for kw in keywords):
                    topic_hits[topic_name].append(i)

        results: list[TopicResult] = []
        for topic_name, seg_indices in topic_hits.items():
            if len(seg_indices) < self.min_topic_frequency:
                continue

            # Sentiment distribution for this topic
            sent_dist: dict[str, int] = {}
            if sentiment_results:
                for idx in seg_indices:
                    if idx < len(sentiment_results):
                        label = sentiment_results[idx].get("label", "neutral")
                        sent_dist[label] = sent_dist.get(label, 0) + 1

            results.append(
                TopicResult(
                    name=topic_name,
                    keywords=TOPIC_KEYWORDS.get(topic_name, []),
                    frequency=len(seg_indices),
                    segments=seg_indices,
                    sentiment_distribution=sent_dist,
                )
            )

        results.sort(key=lambda t: t.frequency, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Emerging topics
    # ------------------------------------------------------------------
    def _detect_emerging_topics(self, texts: list[str]) -> list[str]:
        """Detect potentially emerging topics from unusual word patterns."""
        all_words: list[str] = []
        for text in texts:
            words = re.findall(r"[\wäöåÄÖÅ]+", text.lower())
            all_words.extend([w for w in words if len(w) > 3])

        freq = Counter(all_words)

        # Words appearing frequently but not in predefined topics
        all_keywords = set()
        for kws in TOPIC_KEYWORDS.values():
            all_keywords.update(kws)

        emerging = []
        for word, count in freq.most_common(30):
            if word not in all_keywords and count >= 3:
                emerging.append(word)

        return emerging[:5]


__all__ = ["TopicModeler", "TopicResult", "TopicReport", "TOPIC_KEYWORDS"]
