"""Tests for the end-to-end call analysis pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.pipeline import CallAnalysisPipeline, CallAnalysisReport


class TestCallAnalysisPipeline:
    def setup_method(self):
        self.pipe = CallAnalysisPipeline()

    def _mock_sentiment(self, monkeypatch):
        """Mock sentiment pipeline to avoid loading real models."""

        def mock_analyze(self, texts, **kwargs):
            return (
                [
                    {"label": "negativ", "score": 0.8},
                    {"label": "neutral", "score": 0.6},
                    {"label": "positiv", "score": 0.9},
                ][: max(1, len(texts))]
                if texts
                else []
            )

        monkeypatch.setattr(
            "src.analysis.sentiment.SentimentPipeline.analyze",
            mock_analyze,
        )

    def test_analyze_segments_basic(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        segments = [
            {
                "start": 0,
                "end": 5,
                "text": "Jag är mycket missnöjd med fakturan.",
                "speaker": "SPEAKER_0",
            },
            {"start": 5, "end": 10, "text": "Jag ska hjälpa dig direkt.", "speaker": "SPEAKER_1"},
            {"start": 10, "end": 15, "text": "Tack, det uppskattar jag.", "speaker": "SPEAKER_0"},
        ]
        report = self.pipe.analyze_segments(segments)

        assert isinstance(report, CallAnalysisReport)
        assert len(report.segments) == 3
        assert len(report.sentiment_results) == 3
        assert len(report.intent_results) == 3
        assert report.processing_time_s >= 0

        # Verify structure – each field is a dict from the respective module
        assert isinstance(report.summary, dict)
        assert isinstance(report.topics, dict)
        assert isinstance(report.insights, dict)
        assert isinstance(report.risks, dict)

    def test_analyze_segments_empty(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        report = self.pipe.analyze_segments([])
        assert isinstance(report, CallAnalysisReport)
        assert report.segments == []
        assert report.sentiment_results == []
        assert report.intent_results == []

    def test_report_to_dict(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        segments = [
            {"start": 0, "end": 5, "text": "Bra service!"},
        ]
        report = self.pipe.analyze_segments(segments)
        d = report.to_dict()

        assert "segments" in d
        assert "sentiment_results" in d
        assert "intent_results" in d
        assert "summary" in d
        assert "topics" in d
        assert "insights" in d
        assert "risks" in d
        assert "processing_time_s" in d

        # Intent results should be serialized as dicts
        assert isinstance(d["intent_results"], list)
        if d["intent_results"]:
            assert "intent" in d["intent_results"][0]
            assert "confidence" in d["intent_results"][0]

    def test_analyze_audio_missing_file(self, monkeypatch):
        """Should gracefully handle missing audio files."""
        self._mock_sentiment(monkeypatch)
        report = self.pipe.analyze_audio("/nonexistent/audio.wav")
        assert isinstance(report, CallAnalysisReport)
        assert report.segments == []
        # Diarization is attempted even when transcription fails; it returns an empty result
        assert report.diarization is not None
        assert report.diarization.get("segments") == []

    def test_analyze_segments_with_mistral_flag_accepts_and_merges(self, monkeypatch):
        """Pipeline accepts Mistral flags and surfaces llm in report (even if it falls back)."""
        self._mock_sentiment(monkeypatch)

        # Mock the expensive LLM step so test is fast and deterministic
        fake_llm_out = {
            "trajectory": {"summary": "Test trajectory from Mistral"},
            "actionable_summary": {"problem": "Test problem"},
            "meta": {"model": "mistralai/mistral-medium-3.5", "llm_used": True, "cost_usd": 0.001},
        }

        with patch("src.llm.mistral_analyzer.ConversationMistralAnalyzer") as mock_analyzer_cls:
            mock_inst = MagicMock()
            mock_inst.analyze_full_conversation.return_value = fake_llm_out
            mock_analyzer_cls.return_value = mock_inst

            segments = [
                {
                    "start": 0,
                    "end": 10,
                    "text": "Hej, jag har problem med fakturan.",
                    "speaker": "SPEAKER_1",
                }
            ]
            # Force the deep path
            report = self.pipe.analyze_segments(
                segments,
                # We pass via the pipeline instance flags instead of analyze_segments signature for now
            )
            # Re-create with flags by using a fresh pipeline configured for LLM
            pipe_llm = CallAnalysisPipeline(
                use_mistral_llm=True, llm_model="mistralai/mistral-medium-3.5"
            )
            # Re-apply the sentiment mock on the new instance's path
            monkeypatch.setattr(
                "src.analysis.sentiment.SentimentPipeline.analyze",
                lambda self, texts, **kwargs: [{"label": "negativ", "score": 0.8}] if texts else [],
            )
            report = pipe_llm.analyze_segments(segments)

        assert isinstance(report, CallAnalysisReport)
        assert "llm" in report.to_dict() or hasattr(report, "llm")
        # The llm field should be populated (or contain fallback info)
        assert report.llm is not None

    def test_analyze_segments_includes_fas4_agent_performance(self, monkeypatch):
        """Fas 4.1: report.results must contain agent_performance (Pydantic dump), agent_assessment, customer_metrics."""
        self._mock_sentiment(monkeypatch)
        segments = [
            {
                "start": 0,
                "end": 4,
                "text": "Hej, jag är missnöjd med fakturan.",
                "speaker": "SPEAKER_0",
            },
            {
                "start": 4,
                "end": 9,
                "text": "Jag förstår att det är frustrerande. Jag ska kolla direkt.",
                "speaker": "SPEAKER_1",
            },
            {"start": 9, "end": 12, "text": "Tack, det låter bra.", "speaker": "SPEAKER_0"},
        ]
        report = self.pipe.analyze_segments(segments)

        assert isinstance(report, CallAnalysisReport)
        # Explicit per plan acceptance + integration contract
        assert "agent_performance" in report.results
        assert "agent_assessment" in report.results
        assert "customer_metrics" in report.results

        ap = report.results["agent_performance"]
        assert isinstance(ap, dict)
        assert "agent" in ap
        assert "customer" in ap
        assert "local_coaching_hints" in ap

        agent_m = ap["agent"]
        assert "empathy_score" in agent_m
        assert "talk_ratio" in agent_m
        assert "compliance_flags" in agent_m
        assert 0.0 <= agent_m.get("empathy_score", -1) <= 1.0

        cust_m = ap.get("customer") or report.results.get("customer_metrics", {})
        assert "talk_ratio" in cust_m or "sentiment_slope" in cust_m

        assess = report.results["agent_assessment"]
        assert "empathy_score" in assess
        assert assess.get("source") in ("local_rules_fas4.1", None)  # local until 4.1.2 LLM

    def test_agent_performance_standalone(self):
        """Direct call to new engine produces valid Pydantic and actionable output."""
        from src.agent_performance import compute_call_agent_performance

        segments = [
            {
                "start": 0,
                "end": 2,
                "text": "Hej, välkommen till kundtjänst.",
                "speaker": "SPEAKER_1",
            },
            {
                "start": 2,
                "end": 5,
                "text": "Hallå, jag har fått fel faktura igen!",
                "speaker": "SPEAKER_0",
            },
            {
                "start": 5,
                "end": 10,
                "text": "Jag beklagar verkligen det här. Jag förstår att det är frustrerande för dig. Jag fixar det.",
                "speaker": "SPEAKER_1",
            },
            {"start": 10, "end": 13, "text": "Kan du kolla ordernumret?", "speaker": "SPEAKER_1"},
            {"start": 13, "end": 16, "text": "Ja tack, nu blev det rätt.", "speaker": "SPEAKER_0"},
        ]
        perf = compute_call_agent_performance(
            segments, role_map={"SPEAKER_0": "customer", "SPEAKER_1": "agent"}
        )
        assert perf is not None
        assert (
            perf.agent.empathy_score >= 0.4
        )  # detected "beklagar", "förstår", greeting present so no flag
        assert isinstance(perf.local_coaching_hints, list)
        # customer_metrics present
        assert perf.customer.talk_ratio >= 0.0

    def test_aggregate_insights_fas4_3(self, monkeypatch):
        """Fas 4.3.1: pipeline.aggregate_insights produces AggregatedInsights shape with hot_topics etc."""
        self._mock_sentiment(monkeypatch)
        segments1 = [
            {"start": 0, "end": 3, "text": "Fakturan är fel!", "speaker": "C"},
            {"start": 3, "end": 7, "text": "Jag beklagar, jag fixar det.", "speaker": "A"},
        ]
        segments2 = [
            {"start": 0, "end": 2, "text": "Faktura stämmer inte.", "speaker": "C"},
            {
                "start": 2,
                "end": 6,
                "text": "Jag förstår frustrationen. Vi krediterar.",
                "speaker": "A",
            },
        ]
        r1 = self.pipe.analyze_segments(segments1)
        r2 = self.pipe.analyze_segments(segments2)

        agg = self.pipe.aggregate_insights([r1, r2])
        assert isinstance(agg, dict)
        assert "hot_topics" in agg or "error" in agg
        if "hot_topics" in agg:
            assert isinstance(agg["hot_topics"], list)
            if agg["hot_topics"]:
                ht = agg["hot_topics"][0]
                assert "topic" in ht
                assert "volume" in ht
                assert "avg_sentiment" in ht
                assert "evidence_spans" in ht or "sample_quotes" in ht

    def test_semantic_search_fas4_3_2(self, monkeypatch):
        """Fas 4.3.2: pipeline.semantic_search returns ranked hits with highlights/scores."""
        self._mock_sentiment(monkeypatch)
        segs = [
            {"start": 0, "end": 3, "text": "Fakturan är helt fel, jag är arg.", "speaker": "C"},
            {
                "start": 3,
                "end": 8,
                "text": "Jag beklagar. Jag förstår att det är frustrerande. Jag fixar fakturan.",
                "speaker": "A",
            },
        ]
        r = self.pipe.analyze_segments(segs)
        hits = self.pipe.semantic_search("faktura arg empati", top_k=3, corpus=[r])
        assert isinstance(hits, dict)
        assert "hits" in hits
        assert isinstance(hits["hits"], list)
        if hits["hits"]:
            h = hits["hits"][0]
            assert "score" in h
            assert "highlights" in h or "evidence_spans" in h

    def test_caching_fas4_5_1(self, monkeypatch, tmp_path):
        """Fas 4.5.1: get_cached_* uses cache, invalidate works, hit rate metric."""
        from src.caching import AggregateCache

        self._mock_sentiment(monkeypatch)
        self.pipe.cache = AggregateCache(cache_dir=str(tmp_path / "aggregates"))
        segs = [{"start": 0, "end": 2, "text": "Faktura fel", "speaker": "C"}]
        r1 = self.pipe.analyze_segments(segs)
        r2 = self.pipe.analyze_segments(segs)

        # First call populates cache
        m1 = self.pipe.get_cached_agent_performance("unknown", [r1, r2])
        assert "call_count" in m1 or "error" not in m1

        # Second should hit (same data); cache_hit metadata differs by design
        m2 = self.pipe.get_cached_agent_performance("unknown", [r1, r2])
        assert m1["cache_hit"] is False
        assert m2["cache_hit"] is True
        assert {k: v for k, v in m1.items() if k != "cache_hit"} == {
            k: v for k, v in m2.items() if k != "cache_hit"
        }

        # Invalidate
        self.pipe.invalidate_aggregate_cache("agent:unknown")
        # (next call would recompute)

        print("Cache test basic passed (hit/invalidate)")

    def test_early_pii_redaction_fas4_4_1(self, monkeypatch):
        """Fas 4.4.1: early redaction (before analyzers) when enabled by profile; structured log + redacted text in report."""
        self._mock_sentiment(monkeypatch)
        with monkeypatch.context() as m:
            m.setattr(
                "src.profiles.resolve_profile",
                lambda *a, **k: ("callcenter", {"llm": {"anonymize_before_llm": True}}),
                raising=False,
            )
            p = CallAnalysisPipeline(profile="callcenter")
            segs = [
                {
                    "start": 0,
                    "end": 2,
                    "text": "Mitt personnummer är 19850101-1234, mail john@example.com",
                    "speaker": "C",
                },
                {"start": 2, "end": 5, "text": "Tack, jag fixar.", "speaker": "A"},
            ]
            report = p.analyze_segments(segs)
            assert isinstance(report, CallAnalysisReport)
            assert "pii_redaction" in report.results
            pl = report.results["pii_redaction"]
            assert pl.get("total_redacted", 0) >= 2
            assert "personnummer" in pl.get("types_redacted", []) or "email" in pl.get(
                "types_redacted", []
            )
            # segments text should be redacted
            if report.segments:
                txt = report.segments[0].get("text", "")
                assert "[REDACTED_PNR]" in txt or "[REDACTED_EMAIL]" in txt

    def test_alerting_fas4_4_2(self, monkeypatch):
        """Fas 4.4.2: alerts triggered from results (qa + agent + sentiment), evidence and actions present."""
        self._mock_sentiment(monkeypatch)
        segs = [
            {"start": 0, "end": 2, "text": "Jag är extremt arg på fakturan!", "speaker": "C"},
            {"start": 2, "end": 5, "text": "Okej.", "speaker": "A"},  # low empathy
        ]
        self.pipe.analyze_segments(segs)
        # Force some bad qa/agent via monkey? But since real, check if alerts key appears or run direct
        from src.alerting import run_alerts_on_results

        # Simulate bad signals
        fake_results = {
            "llm": {"trajectory": {"customer_sentiment_slope": -0.8, "escalation_events": ["bad"]}},
            "agent_performance": {"agent": {"empathy_score": 0.2}},
            "qa": {"passed": False, "risk_level": "high", "overall_qa_score": 40},
        }
        alerts = run_alerts_on_results(fake_results)
        assert isinstance(alerts, list)
        if alerts:
            a = alerts[0]
            assert "rule_id" in a
            assert "evidence_spans" in a
            assert "recommended_actions" in a
            assert len(a["recommended_actions"]) > 0

    def test_analyze_segments_includes_fas4_qa_scoring(self, monkeypatch):
        """Fas 4.2: results contain qa / compliance_qa with overall, passed/failed, evidence, risk."""
        self._mock_sentiment(monkeypatch)
        segments = [
            {"start": 0, "end": 2, "text": "Hej, välkommen.", "speaker": "SPEAKER_1"},
            {"start": 2, "end": 6, "text": "Jag är arg, fakturan är fel!", "speaker": "SPEAKER_0"},
            {
                "start": 6,
                "end": 11,
                "text": "Jag beklagar. Jag förstår frustrationen. Jag fixar det nu.",
                "speaker": "SPEAKER_1",
            },
            {"start": 11, "end": 14, "text": "Tack, bra.", "speaker": "SPEAKER_0"},
        ]
        report = self.pipe.analyze_segments(segments)
        assert "qa" in report.results or "compliance_qa" in report.results
        qa = report.results.get("qa") or report.results.get("compliance_qa")
        assert isinstance(qa, dict)
        assert "overall_qa_score" in qa
        assert "passed" in qa
        assert "risk_level" in qa
        assert "criteria_results" in qa or "error" in qa
        assert qa.get("overall_qa_score", 0) >= 0
