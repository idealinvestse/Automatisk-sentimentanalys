"""Tests for Fas 4.3.2 semantic search engine."""

from __future__ import annotations

from src.semantic_search import (
    SemanticSearchEngine,
    _cosine_sim,
    _hash_texts,
    _keyword_score,
    build_semantic_index_from_reports,
)


class TestSemanticSearchHelpers:
    def test_cosine_sim_identical(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_cosine_sim_empty(self):
        assert _cosine_sim([], [1.0]) == 0.0

    def test_keyword_score_overlap(self):
        score = _keyword_score("fakturan är fel och arg", "faktura arg")
        assert score > 0.5

    def test_keyword_score_with_topics(self):
        score = _keyword_score("hej", "faktura", topics=["faktura", "support"])
        assert score > 0.0

    def test_hash_texts_stable(self):
        assert _hash_texts(["a", "b"]) == _hash_texts(["a", "b"])


class TestSemanticSearchEngine:
    def test_empty_index_returns_no_hits(self):
        engine = SemanticSearchEngine(use_faiss=False)
        result = engine.search("faktura")
        assert result.hits == []
        assert result.meta["num_docs"] == 0

    def test_keyword_search_ranks_relevant_doc(self):
        engine = SemanticSearchEngine(use_faiss=False)
        engine.index(
            [
                {"id": "1", "text": "Kunden klagade på fakturan", "topics": ["faktura"]},
                {"id": "2", "text": "Leverans var sen", "topics": ["leverans"]},
            ]
        )
        result = engine.search("faktura klagade", top_k=2)
        assert len(result.hits) >= 1
        assert result.hits[0].id == "1"
        assert result.hits[0].score > 0
        assert result.hits[0].highlights

    def test_filters_exclude_non_matching(self):
        engine = SemanticSearchEngine(use_faiss=False)
        engine.index(
            [
                {"id": "a", "text": "faktura problem", "agent": "Agent-1"},
                {"id": "b", "text": "faktura problem", "agent": "Agent-2"},
            ]
        )
        result = engine.search("faktura", filters={"agent": "Agent-1"})
        assert all(h.metadata.get("agent") == "Agent-1" for h in result.hits)

    def test_build_index_from_report_dicts(self):
        reports = [
            {
                "file": "call-1.wav",
                "segments": [{"text": "Fakturan stämmer inte"}],
                "topics": {"topics": ["faktura"]},
                "results": {"agent_performance": {"agent": {"empathy_score": 0.5}}},
            }
        ]
        engine = build_semantic_index_from_reports(reports)
        result = engine.search("faktura", top_k=1)
        assert len(result.hits) == 1
        assert result.hits[0].id == "call-1.wav"
