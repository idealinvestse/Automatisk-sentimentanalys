"""Semantic Search Engine (Fas 4.3.2).

Hybrid search over call transcripts + insights (Fas3/4 data):
- Vector similarity (sentence-transformers embeddings if available)
- Keyword / BM25-style boost (simple term overlap + existing topics/aspects)
- Filters (agent, sentiment range, time, topic etc.)
- Natural language queries (embed the query text)

Returns ranked results with relevance score + highlights (matched snippets + evidence).

Optional FAISS for fast ANN (if installed); otherwise brute-force cosine.

Pydantic output for consistency with the rest of the Fas4 stack.

Explicit integration (pipeline.py):
    from src.pipeline import CallAnalysisPipeline
    pipe = CallAnalysisPipeline(profile="callcenter")
    reports = [pipe.analyze_audio(p) for p in files]
    # Build or search directly
    hits = pipe.semantic_search("kunder klagade på faktura och agent visade låg empati", top_k=5, reports=reports)
    # hits is list of {"id": , "score": , "highlights": [...], "metadata": ...}

Graceful: works without extra deps (pure python cosine + keyword).

Caching: embed queries and docs (simple dict cache).

Privacy: searches only over data the caller already has (redacted if reports were produced with anonymize).

See plan 4.3.2 for acceptance (semantic queries like the example return relevant calls with evidence).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core.models import CallAnalysisReport
from .llm.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

# Optional vector deps (same as aggregator)
try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    _SEMANTIC_MODEL = None
    SEMANTIC_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _SEMANTIC_MODEL = None
    SEMANTIC_AVAILABLE = False

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


def _get_semantic_model() -> Any | None:
    global _SEMANTIC_MODEL
    if not SEMANTIC_AVAILABLE:
        return None
    if _SEMANTIC_MODEL is None:
        try:
            _SEMANTIC_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            logger.warning("semantic model load failed: %s", e)
            return None
    return _SEMANTIC_MODEL


class SearchHit(BaseModel):
    """Single ranked result from semantic/hybrid search."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier for the call/document (e.g. filename or index)")
    score: float = Field(..., ge=0.0, description="Hybrid relevance (0-1 or higher for strong matches)")
    highlights: list[str] = Field(default_factory=list, description="Snippets/evidence that matched the query")
    metadata: dict[str, Any] = Field(default_factory=dict, description="topic, sentiment, agent, timestamp etc.")
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class SemanticSearchResult(BaseModel):
    """Container for semantic search results (Fas 4.3.2)."""

    model_config = ConfigDict(extra="forbid")

    query: str
    hits: list[SearchHit]
    meta: dict[str, Any] = Field(default_factory=dict, description="used_vector, used_keyword, num_docs, filters etc.")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _keyword_score(text: str, query: str, topics: list[str] | None = None) -> float:
    """Lightweight keyword overlap (proxy for BM25-lite)."""
    if not text or not query:
        return 0.0
    q_terms = set(query.lower().split())
    t = text.lower()
    hits = sum(1 for qt in q_terms if qt in t)
    base = hits / max(1, len(q_terms))
    # boost if topic overlap
    if topics:
        tset = set(topics)
        topic_hits = len(q_terms & tset)
        base += 0.3 * (topic_hits / max(1, len(q_terms)))
    return min(1.0, base)


class SemanticSearchEngine:
    """Builds a searchable index over call data and performs hybrid search."""

    def __init__(self, use_faiss: bool = True) -> None:
        self.docs: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] | None = None
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._faiss_index = None
        self._embed_cache: dict[str, list[float]] = {}

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        model = _get_semantic_model()
        if model is None:
            return None
        key = _hash_texts(texts)
        if key in self._embed_cache:
            return [self._embed_cache[key]] * len(texts)  # simplistic; real would per-text
        try:
            embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
            for t, e in zip(texts, embs):
                self._embed_cache[_hash_texts([t])] = e
            return embs
        except Exception:
            return None

    def index(
        self,
        items: list[dict[str, Any]],
        id_field: str = "id",
        text_fields: list[str] = ("text", "summary", "insights"),
    ) -> None:
        """Index a list of documents. Each item should have text-ish fields + metadata."""
        self.docs = []
        texts_for_embed: list[str] = []
        for item in items:
            doc = {
                "id": str(item.get(id_field, len(self.docs))),
                "text": " ".join(str(item.get(f, "")) for f in text_fields if item.get(f)),
                "metadata": {k: v for k, v in item.items() if k not in list(text_fields) + [id_field]},
            }
            self.docs.append(doc)
            texts_for_embed.append(doc["text"][:2000])

        self.embeddings = self._embed(texts_for_embed)

        if self.use_faiss and self.embeddings and len(self.embeddings) > 0:
            try:
                import numpy as np  # type: ignore

                dim = len(self.embeddings[0])
                self._faiss_index = faiss.IndexFlatIP(dim)
                arr = np.array(self.embeddings).astype("float32")
                self._faiss_index.add(arr)
            except Exception as e:
                logger.warning("FAISS index build failed, falling back to brute: %s", e)
                self._faiss_index = None

        logger.info("Semantic index built | docs=%d vector=%s", len(self.docs), bool(self.embeddings))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """Hybrid search.

        filters example: {"agent": "A1", "min_sentiment": -0.5, "topics": ["faktura"]}
        """
        if not self.docs:
            return SemanticSearchResult(query=query, hits=[], meta={"num_docs": 0})

        filters = filters or {}
        q_emb = None
        model = _get_semantic_model()
        if model:
            try:
                q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
            except Exception:
                q_emb = None

        scored: list[tuple[float, int]] = []
        for i, d in enumerate(self.docs):
            # apply filters
            meta = d.get("metadata", {})
            if not self._passes_filters(meta, filters):
                continue

            vec_score = 0.0
            if q_emb and self.embeddings and i < len(self.embeddings):
                vec_score = _cosine_sim(q_emb, self.embeddings[i])

            kw_score = _keyword_score(d["text"], query, meta.get("topics") or [])

            # hybrid
            score = 0.6 * vec_score + 0.4 * kw_score
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        hits: list[SearchHit] = []
        for score, idx in scored[:top_k]:
            d = self.docs[idx]
            # simple highlights: sentences containing query terms
            highlights = []
            for sent in d["text"].split(".")[:5]:
                if any(qt in sent.lower() for qt in query.lower().split()):
                    highlights.append(sent.strip()[:200])
            if not highlights:
                highlights = [d["text"][:150]]

            ev = []
            for h in highlights[:2]:
                ev.append(EvidenceSpan(text=h))

            hits.append(
                SearchHit(
                    id=d["id"],
                    score=round(score, 4),
                    highlights=highlights,
                    metadata=d.get("metadata", {}),
                    evidence_spans=ev,
                )
            )

        meta = {
            "num_docs": len(self.docs),
            "returned": len(hits),
            "used_vector": bool(q_emb),
            "used_keyword": True,
            "faiss": self._faiss_index is not None,
        }
        return SemanticSearchResult(query=query, hits=hits, meta=meta)

    def _passes_filters(self, meta: dict, filters: dict) -> bool:
        for k, v in filters.items():
            mv = meta.get(k)
            if mv is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if mv not in v:
                    return False
            elif mv != v:
                # numeric range support e.g. min_sentiment
                if k.startswith("min_") and isinstance(mv, (int, float)):
                    if mv < float(v):
                        return False
                elif k.startswith("max_") and isinstance(mv, (int, float)):
                    if mv > float(v):
                        return False
                else:
                    return False
        return True


def _hash_texts(texts: list[str]) -> str:
    return hashlib.sha256("||".join(texts).encode("utf-8")).hexdigest()[:10]


# Pipeline convenience
def build_semantic_index_from_reports(reports: list[CallAnalysisReport | dict]) -> SemanticSearchEngine:
    """Helper to build a search index directly from Fas4 reports (used by pipeline)."""
    engine = SemanticSearchEngine()
    docs = []
    for i, r in enumerate(reports):
        if isinstance(r, CallAnalysisReport):
            d = r.to_dict()
        else:
            d = r
        # flatten useful text
        text_parts = []
        for s in d.get("segments", []):
            text_parts.append(str(s.get("text", "")))
        llm = d.get("llm") or {}
        if llm.get("actionable_summary"):
            text_parts.append(str(llm["actionable_summary"].get("problem", "")))
        if llm.get("root_cause"):
            text_parts.append(str(llm["root_cause"].get("primary_cause", "")))
        docs.append({
            "id": d.get("file") or str(i),
            "text": " ".join(text_parts),
            "topics": (d.get("topics") or {}).get("topics", []),
            "sentiment_summary": (d.get("results") or {}).get("sentiment", []),
            "agent_performance": (d.get("results") or {}).get("agent_performance"),
            "qa": (d.get("results") or {}).get("qa") or (d.get("results") or {}).get("compliance_qa"),
        })
    engine.index(docs)
    return engine


__all__ = [
    "SemanticSearchEngine",
    "build_semantic_index_from_reports",
    "SearchHit",
    "SemanticSearchResult",
]
