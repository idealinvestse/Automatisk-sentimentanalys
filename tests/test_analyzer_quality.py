"""Quality and regression tests for heuristic analyzers (ANALYSIS_QUALITY_REPORT)."""

from __future__ import annotations

import pytest

from src.analysis.active_listening import ActiveListeningBehaviorAnalyzer
from src.analysis.compliance_risk import ComplianceRiskAnalyzer
from src.analysis.customer_effort import CustomerEffortScoreAnalyzer
from src.analysis.dialect_sensitivity import DialectSensitivityAnalyzer
from src.analysis.emotion import EmotionAnalyzer
from src.analysis.intent import IntentAnalyzer
from src.analysis.intent_utils import intent_label, intents_as_tuples
from src.analysis.registry import ensure_analyzers_loaded
from src.analysis.root_cause import RootCauseInsightAnalyzer
from src.analysis.schemas import validate_analyzer_result
from src.analysis.trajectory import TrajectoryAnalyzer
from src.core.models import AnalysisContext, Segment


def _seg(text: str, speaker: str = "SPEAKER_0", start: float = 0.0) -> Segment:
    return Segment(start=start, end=start + 1.0, text=text, speaker=speaker)


@pytest.fixture(autouse=True)
def _load_analyzers() -> None:
    ensure_analyzers_loaded()


class TestComplianceRisk:
    def test_empty_path_schema_compatible(self) -> None:
        out = ComplianceRiskAnalyzer().analyze(AnalysisContext(segments=[]))
        validated = validate_analyzer_result("compliance_risk", out, mode="strict")
        assert validated["overall_risk_level"] == "low"
        assert validated["flagged_segments"] == []

    def test_flags_agent_over_promise(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Jag lovar att det fixas idag.", "SPEAKER_0")],
            results={"role": {"roles": {"SPEAKER_0": "agent", "SPEAKER_1": "customer"}}},
        )
        out = ComplianceRiskAnalyzer().analyze(ctx)
        assert out["overall_risk_level"] in ("medium", "high")
        assert out["flagged_segments"][0]["risks"] == ["over_promise"]

    def test_skips_customer_segments(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Jag lovar att stämma er.", "SPEAKER_1")],
            results={"role": {"roles": {"SPEAKER_0": "agent", "SPEAKER_1": "customer"}}},
        )
        out = ComplianceRiskAnalyzer().analyze(ctx)
        assert out["flagged_segments"] == []


class TestCustomerEffort:
    def test_empty_path_schema_compatible(self) -> None:
        out = CustomerEffortScoreAnalyzer().analyze(AnalysisContext(segments=[]))
        validated = validate_analyzer_result("customer_effort", out, mode="strict")
        assert validated["coaching_tips"] == []

    def test_detects_fillers(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Eh, alltså, typ så här liksom.")])
        out = CustomerEffortScoreAnalyzer().analyze(ctx)
        assert out["overall_ces"] > 20


class TestActiveListening:
    def test_returns_listening_score(self) -> None:
        segments = [
            _seg("Hej.", "SPEAKER_0", 0),
            _seg("Ja, precis.", "SPEAKER_1", 1.5),
        ]
        out = ActiveListeningBehaviorAnalyzer().analyze(AnalysisContext(segments=segments))
        assert 0 <= out["listening_score"] <= 100


class TestTrajectory:
    def test_customer_only_trend(self) -> None:
        segments = [
            _seg("Hej och välkommen.", "SPEAKER_0", 0),
            _seg("Jag är arg på er.", "SPEAKER_1", 1),
            _seg("Jag förstår.", "SPEAKER_0", 2),
            _seg("Fortfarande arg.", "SPEAKER_1", 3),
        ]
        ctx = AnalysisContext(
            segments=segments,
            results={
                "role": {"roles": {"SPEAKER_0": "agent", "SPEAKER_1": "customer"}},
                "sentiment": [
                    {"label": "neutral", "score": 0.0},
                    {"label": "negativ", "score": 0.8},
                    {"label": "neutral", "score": 0.0},
                    {"label": "negativ", "score": 0.9},
                ],
                "emotion": [
                    {"primary": "neutral", "scores": {"neutral": 0.9}},
                    {"primary": "frustration", "scores": {"frustration": 0.75}},
                    {"primary": "neutral", "scores": {"neutral": 0.9}},
                    {"primary": "ilska", "scores": {"ilska": 0.75}},
                ],
            },
        )
        out = TrajectoryAnalyzer().analyze(ctx)
        assert len(out["sentiment_trend"]) == 2
        assert out["escalation_events"] >= 1
        assert out["escalation_event_details"]


class TestRootCause:
    def test_detects_wait_time(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Jag har väntat i telefonkö i 40 minuter.")],
            results={"intent": [], "trajectory": {}, "sentiment": []},
        )
        out = RootCauseInsightAnalyzer().analyze(ctx)
        assert out["top_root_cause"] or out["root_causes"]


class TestDialectSensitivity:
    def test_common_words_not_flagged(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Här är jag och där är du, inte sant?")])
        out = DialectSensitivityAnalyzer().analyze(ctx)
        assert out["flagged_segments"] == []

    def test_dialect_marker_flagged(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Det var mycke bra, nä.")])
        out = DialectSensitivityAnalyzer().analyze(ctx)
        assert len(out["flagged_segments"]) >= 1


class TestIntentOutput:
    def test_returns_dict_format(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Jag vill avsluta mitt abonnemang.")])
        out = IntentAnalyzer().analyze(ctx)
        assert out[0]["intent"]
        assert "confidence" in out[0]

    def test_intent_utils_backward_compat(self) -> None:
        items = [{"intent": "complaint", "confidence": 0.9}]
        tuples = intents_as_tuples(items)
        assert tuples == [("complaint", 0.9)]
        assert intent_label(items[0]) == "complaint"


class TestEmotionHybrid:
    def test_sentiment_boosts_negative_emotion(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Det här fungerar verkligen dåligt.")],
            results={
                "sentiment": [{"label": "negativ", "score": 0.85}],
                "negation": [{"has_negation": False, "negation_count": 0}],
            },
        )
        out = EmotionAnalyzer().analyze(ctx)
        assert out[0]["primary"] in ("frustration", "besvikelse", "oro", "neutral")


class TestSpokenNormalizer:
    def test_segment_analysis_text_uses_normalized(self) -> None:
        from src.analysis.spoken_normalizer import SpokenNormalizerAnalyzer
        from src.analysis.text_utils import segment_analysis_text

        seg = _seg("Eh, alltså, typ bra service.")
        ctx = AnalysisContext(segments=[seg])
        normalized = SpokenNormalizerAnalyzer().analyze(ctx)[0]["normalized"]
        ctx.results["spoken_normalizer"] = [{"normalized": normalized, "original": seg.text}]
        assert segment_analysis_text(ctx, 0) == normalized
        assert "bra service" in normalized.lower()

    def test_normalizer_runs_before_sentiment_when_selected(self, monkeypatch) -> None:
        from src.analysis.registry import run_analyzers
        from src.analysis.sentiment import SentimentAnalyzer
        from src.analysis.text_utils import segment_analysis_text

        captured: list[list[str]] = []

        def mock_analyze(self, ctx):
            captured.append([segment_analysis_text(ctx, i) for i in range(len(ctx.segments or []))])
            return [{"label": "positiv", "score": 0.9}]

        monkeypatch.setattr(SentimentAnalyzer, "analyze", mock_analyze)
        ctx = AnalysisContext(segments=[_seg("Eh, alltså, typ bra service.")])
        run_analyzers(ctx, selected=["spoken_normalizer", "sentiment"])
        text = captured[0].lower()
        assert "bra service" in text
        assert "eh" not in text.split()
        assert "typ" not in text.split()
        assert "alltså" not in text


class TestCallDetailMarkdown:
    def test_build_insights_reads_llm_fields(self) -> None:
        from app.nicegui_dashboard.components.call_detail import _build_insights_markdown

        report = {
            "llm": {
                "root_cause": {"primary_cause": "Missad validering tidigt"},
                "actionable_summary": {
                    "problem": "Faktura",
                    "recommendations_for_qa": ["Coacha empati"],
                },
            },
            "results": {},
        }
        md = _build_insights_markdown(report)
        assert "Missad validering" in md
        assert "Coacha empati" in md


class TestIntentBenchmarkSmoke:
    def test_heuristic_holdout_subset(self) -> None:
        import json
        from pathlib import Path

        from sklearn.metrics import accuracy_score

        from src.intent import IntentClassifier

        path = Path("data/intent_train.jsonl")
        texts: list[str] = []
        labels: list[str] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["intent"])
                if len(texts) >= 20:
                    break

        clf = IntentClassifier(backend="heuristic")
        preds = [clf.classify(t)[0] for t in texts]
        assert accuracy_score(labels, preds) >= 0.5
