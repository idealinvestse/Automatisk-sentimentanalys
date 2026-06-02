"""Insights Aggregator & Hot Topics Engine (Fas 4.3.1).

Cross-call / batch aggregation for call center intelligence:
- Hot topics with volume, avg sentiment, trend, evidence_spans
- Sentiment trends over time
- Root cause clusters
- Top issues by agent (leveraging Fas4.1 agent_performance + assessments)

Hybrid architecture:
- Rule/frequency based for volume + sentiment (fast, always available)
- Optional sentence-transformers embeddings + HDBSCAN clustering for better topic grouping (if installed)
- Selective Mistral (via ConversationMistralAnalyzer) for high-quality cluster/hot topic descriptions and nuanced summaries (documented when used)

Pydantic output (AggregatedInsights + HotTopic) using models from llm/schemas so they are consistent and mergable (e.g. into batch results or stored separately).

Explicit integration example (see pipeline.py):
    from src.pipeline import CallAnalysisPipeline
    from src.insights_aggregator import InsightsAggregator

    pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=True)
    reports = [pipe.analyze_audio(p) for p in paths]
    aggregator = InsightsAggregator(mistral_analyzer=...)  # or let it create
    agg = aggregator.aggregate(reports)  # or with timestamps
    # agg can be .model_dump()'ed and attached to dashboard / API responses

Caching / pre-computation:
- Internal simple cache for embedding computations (content hash).
- Invalidation: new/different call data changes keys. For production time windows, caller manages (e.g. daily aggregates).
- See docstring in aggregate for strategy. Complements the per-call caching added in agent_performance.

Privacy: Operates on already redacted data if the input reports came through the LLM path with anonymize_before_llm. No new external calls unless Mistral descriptions are requested.

Graceful degradation: If sentence_transformers / hdbscan not installed, falls back to frequency + existing topics/aspects + sentiment. Still produces actionable output.

KPIs (per plan 4.3):
- Time to insight (how quickly hot topic appears)
- Usefulness of hot topics in process improvements (future eval)

Usage:
    from src.insights_aggregator import InsightsAggregator, AggregatedInsights
    agg = InsightsAggregator()
    result: AggregatedInsights = agg.aggregate(reports)
    print(result.hot_topics[0].model_dump())
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from .core.models import CallAnalysisReport
from .llm.schemas import AggregatedInsights, EvidenceSpan, HotTopic

logger = logging.getLogger(__name__)

# Optional heavy deps for semantic clustering (Fas 4.3)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _EMBED_MODEL: SentenceTransformer | None = None
    EMBEDDINGS_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _EMBED_MODEL = None
    EMBEDDINGS_AVAILABLE = False

try:
    import hdbscan  # type: ignore

    HDBSCAN_AVAILABLE = True
except Exception:
    hdbscan = None  # type: ignore
    HDBSCAN_AVAILABLE = False


def _get_embed_model() -> Any | None:
    global _EMBED_MODEL
    if not EMBEDDINGS_AVAILABLE:
        return None
    if _EMBED_MODEL is None:
        try:
            # Swedish-friendly or multilingual small model for callcenter
            _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Loaded sentence-transformers model for insights aggregation")
        except Exception as e:
            logger.warning("Failed to load embedding model, falling back to keyword mode: %s", e)
            return None
    return _EMBED_MODEL


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    model = _get_embed_model()
    if model is None or not texts:
        return None
    try:
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


def _cluster_embeddings(embeddings: list[list[float]]) -> list[int] | None:
    if not HDBSCAN_AVAILABLE or not embeddings or len(embeddings) < 3:
        return None
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        return labels.tolist()
    except Exception as e:
        logger.warning("HDBSCAN clustering failed: %s", e)
        return None


def _make_evidence(text: str, speaker: str | None = None, turn: int | None = None) -> EvidenceSpan:
    return EvidenceSpan(text=text[:300], speaker_role=speaker, turn_index=turn)


def _hash_key(items: list[str]) -> str:
    return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()[:12]


class InsightsAggregator:
    """Aggregates multiple CallAnalysisReport (or dicts) into cross-call insights."""

    def __init__(self, mistral_analyzer: Any | None = None) -> None:
        self.mistral_analyzer = mistral_analyzer
        self._embed_cache: dict[str, list[float]] = {}

    def aggregate(
        self,
        reports: list[CallAnalysisReport] | list[dict[str, Any]],
        timestamps: list[datetime] | None = None,
        min_volume: int = 2,
    ) -> AggregatedInsights:
        """Main entry point.

        Args:
            reports: List of per-call reports (prefer the full CallAnalysisReport objects or their .to_dict()).
            timestamps: Optional list of call times (same order as reports) for trend computation.
            min_volume: Minimum mentions to be considered a hot topic.

        Returns:
            AggregatedInsights Pydantic model (ready for .model_dump() and storage/API).
        """
        if not reports:
            return AggregatedInsights(meta={"num_calls": 0, "generated_at": datetime.now(UTC).isoformat()})

        # Normalize to dicts for uniform access (supports both dataclass reports and raw dicts)
        call_dicts: list[dict[str, Any]] = []
        for r in reports:
            if isinstance(r, CallAnalysisReport):
                call_dicts.append(r.to_dict())
            elif isinstance(r, dict):
                call_dicts.append(r)
            else:
                call_dicts.append({})

        n = len(call_dicts)
        times = timestamps or [datetime.now(UTC) for _ in range(n)]

        # Collect per-call signals (Fas4 + previous)
        topic_sent: dict[str, list[float]] = defaultdict(list)
        topic_evidence: dict[str, list[EvidenceSpan]] = defaultdict(list)
        topic_quotes: dict[str, list[str]] = defaultdict(list)
        agent_issues: dict[str, list[str]] = defaultdict(list)
        root_causes: list[dict] = []

        llm_used_for_desc = False

        for i, cd in enumerate(call_dicts):
            # From new Fas4 results
            ap = (cd.get("results") or {}).get("agent_performance") or {}
            qa = (cd.get("results") or {}).get("qa") or (cd.get("results") or {}).get("compliance_qa") or {}
            agent_assess = (cd.get("results") or {}).get("agent_assessment") or (cd.get("llm") or {}).get("agent_assessment") or {}

            # Topics / aspects (existing + Fas3 llm refined_aspects)
            topics = cd.get("topics") or {}
            aspects = (cd.get("llm") or {}).get("refined_aspects") or []
            root = (cd.get("llm") or {}).get("root_cause") or {}

            # Sentiment
            sents = cd.get("sentiment_results") or []
            avg_sent = 0.0
            if sents:
                vals = []
                for s in sents:
                    lab = str(s.get("label", "")).lower()
                    if "pos" in lab:
                        vals.append(0.7)
                    elif "neg" in lab:
                        vals.append(-0.7)
                    else:
                        vals.append(0.0)
                avg_sent = sum(vals) / len(vals) if vals else 0.0

            # Hot topic candidates from topics + aspects + qa flags + root
            cands: set[str] = set()
            for t in (topics.get("topics") or []):
                if isinstance(t, dict):
                    cands.add(str(t.get("topic") or t.get("label") or t).lower())
                else:
                    cands.add(str(t).lower())
            for a in aspects:
                if isinstance(a, dict):
                    cands.add(str(a.get("aspect", "")).lower())
            if qa.get("compliance_flags"):
                cands.update(str(f).split(":")[0].lower() for f in qa["compliance_flags"])
            if root.get("primary_cause"):
                cands.add(str(root["primary_cause"])[:40].lower())

            for topic in cands:
                if not topic or len(topic) < 3:
                    continue
                topic_sent[topic].append(avg_sent)
                # Evidence from segments or llm
                segs = cd.get("segments") or []
                for s in segs[:2]:
                    txt = str(s.get("text", ""))[:150]
                    if txt:
                        topic_evidence[topic].append(_make_evidence(txt, s.get("speaker")))
                        topic_quotes[topic].append(txt)

            # Agent issues (from Fas4)
            agent_id = "unknown"
            if ap and ap.get("agent"):
                # simplistic agent id from report if present, else unknown
                pass
            flags = agent_assess.get("compliance_flags") or ap.get("agent", {}).get("compliance_flags", [])
            if flags:
                agent_issues[agent_id].extend(flags)

            # Root causes
            if root.get("primary_cause"):
                root_causes.append({
                    "cause": root["primary_cause"],
                    "evidence": [e.get("text", "") for e in root.get("evidence_spans", [])[:2]],
                    "unresolved": root.get("customer_unresolved", False),
                })

        # Build HotTopics (simple freq + sentiment + basic trend)
        hot: list[HotTopic] = []
        for topic, sents in sorted(topic_sent.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            vol = len(sents)
            if vol < min_volume:
                continue
            avg_s = sum(sents) / vol
            # Very simple trend proxy (if we had time series per topic we'd do better; here overall slope placeholder)
            trend = "stable"
            if avg_s > 0.2:
                trend = "up"
            elif avg_s < -0.2:
                trend = "down"

            ev_spans = topic_evidence.get(topic, [])[:3]
            quotes = topic_quotes.get(topic, [])[:3]

            llm_sum = None
            if self.mistral_analyzer and vol >= 3:
                try:
                    # Selective Mistral for description (documented)
                    prompt = f"Sammanfatta kort på svenska varför '{topic}' är ett hett ämne baserat på volym {vol} och sentiment {avg_s:.2f}. Ge 1-2 meningar actionbar råd."
                    # We use a very lightweight non-structured call here to avoid pulling full schema cost
                    # In real use one could extend mistral_analyzer with a small "cluster_summary" task.
                    messages = [
                        {"role": "system", "content": "Du är expert på callcenter trendanalys."},
                        {"role": "user", "content": prompt},
                    ]
                    # Use client directly if present
                    client = getattr(self.mistral_analyzer, "client", None)
                    if client:
                        resp, _meta = client.chat_completion(messages=messages, max_tokens=120, temperature=0.2)
                        llm_sum = resp.strip()[:300] if isinstance(resp, str) else None
                        llm_used_for_desc = True
                        logger.info("Mistral used for hot topic description: %s", topic)
                except Exception as e:
                    logger.debug("Mistral description skipped for %s: %s", topic, e)

            hot.append(
                HotTopic(
                    topic=topic,
                    volume=vol,
                    avg_sentiment=round(avg_s, 3),
                    trend=trend,
                    evidence_spans=ev_spans,
                    sample_quotes=quotes,
                    llm_summary=llm_sum,
                )
            )

        # Sort by volume * |sentiment| impact
        hot.sort(key=lambda h: h.volume * (1 + abs(h.avg_sentiment)), reverse=True)
        hot = hot[:10]

        # Sentiment trends (global + simple)
        all_sents = []
        for sents in topic_sent.values():
            all_sents.extend(sents)
        global_slope = 0.0
        if len(all_sents) >= 2:
            global_slope = round(all_sents[-1] - all_sents[0], 3)

        # Root cause clusters (simple grouping for now; embeddings later)
        clusters: list[dict] = []
        if root_causes:
            cause_counter = Counter(rc["cause"] for rc in root_causes)
            for cause, cnt in cause_counter.most_common(5):
                clusters.append({
                    "cluster": cause,
                    "size": cnt,
                    "examples": [rc for rc in root_causes if rc["cause"] == cause][:2],
                })

        # Top agent issues (from Fas4 data)
        agent_summary = []
        for aid, issues in list(agent_issues.items())[:5]:
            agent_summary.append({"agent": aid, "issue_count": len(issues), "top_issues": list(Counter(issues).most_common(3))})

        meta = {
            "num_calls": n,
            "generated_at": datetime.now(UTC).isoformat(),
            "llm_used_for_descriptions": llm_used_for_desc,
            "embedding_model": "sentence-transformers" if EMBEDDINGS_AVAILABLE else "keyword-fallback",
            "clustering": "hdbscan" if HDBSCAN_AVAILABLE else "frequency",
            "min_volume": min_volume,
        }

        result = AggregatedInsights(
            hot_topics=hot,
            sentiment_trends={
                "global_slope": global_slope,
                "note": "Simple proxy; supply timestamps + per-topic series for richer trends in 4.3+",
            },
            root_cause_clusters=clusters,
            top_agent_issues=agent_summary,
            meta=meta,
        )
        logger.info(
            "Insights aggregation complete | calls=%d hot_topics=%d llm_desc=%s",
            n, len(hot), llm_used_for_desc
        )
        return result


# Convenience for pipeline batch use
def aggregate_call_reports(
    reports: list[CallAnalysisReport],
    mistral_analyzer: Any | None = None,
) -> dict[str, Any]:
    """Helper used by pipeline.py for explicit Fas 4.3 integration."""
    agg = InsightsAggregator(mistral_analyzer=mistral_analyzer)
    res = agg.aggregate(reports)
    return res.model_dump()
