"""Golden-file regression tests for callcenter pipeline (synthetic transcripts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.pipeline import CallAnalysisPipeline

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "callcenter_golden"
GOLDEN_FIXTURES = sorted(FIXTURE_DIR.glob("cc_*.json"))


def _mock_sentiment(self, texts, **kwargs):
    results = []
    for text in texts:
        tl = text.lower()
        if any(
            w in tl for w in ("arg", "frustrerad", "oacceptabelt", "missnöjd", "dålig", "klagomål")
        ):
            results.append({"label": "negativ", "score": 0.85})
        elif any(w in tl for w in ("tack", "bra", "perfekt", "smidigt", "toppen")):
            results.append({"label": "positiv", "score": 0.85})
        else:
            results.append({"label": "neutral", "score": 0.5})
    return results


def _mock_intent_batch(self, texts):
    results = []
    for text in texts:
        tl = text.lower()
        if any(w in tl for w in ("faktura", "betala", "avgift")):
            results.append(("billing_inquiry", 0.9))
        elif any(w in tl for w in ("adress", "uppdatera", "kontaktuppgifter")):
            results.append(("account_update", 0.9))
        elif any(w in tl for w in ("wifi", "internet", "router", "teknik", "fungerar inte")):
            results.append(("technical_support", 0.9))
        elif any(w in tl for w in ("avsluta", "säga upp", "abonnemang")):
            results.append(("cancellation", 0.9))
        elif any(w in tl for w in ("klagomål", "missnöjd", "oacceptabelt")):
            results.append(("complaint", 0.9))
        elif any(w in tl for w in ("öppettider", "erbjudande", "information")):
            results.append(("information_request", 0.9))
        else:
            results.append(("other", 0.5))
    return results


@pytest.fixture
def mock_heavy_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.analysis.sentiment.SentimentPipeline.analyze",
        _mock_sentiment,
    )
    monkeypatch.setattr(
        "src.intent.IntentClassifier.classify_batch",
        _mock_intent_batch,
    )
    monkeypatch.setattr(
        "src.llm.mistral_analyzer.ConversationMistralAnalyzer.analyze_full_conversation",
        lambda self, **kwargs: {"fallback": True, "meta": {"llm_used": False}},
    )


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_golden_expectations(fixture_path: Path, report: Any) -> None:
    data = _load_fixture(fixture_path)
    expect = data["expect"]
    results = report.results or {}

    for key in expect["results_keys"]:
        assert key in results, f"Missing results key '{key}' in {fixture_path.name}"

    intents = [
        item.get("intent") if isinstance(item, dict) else item[0]
        for item in (report.intent_results or [])
    ]
    assert expect["intent_contains"] in intents

    if expect.get("has_negative_sentiment"):
        labels = [
            s.get("label", "") if isinstance(s, dict) else str(s)
            for s in (report.sentiment_results or [])
        ]
        assert any("neg" in str(lbl).lower() for lbl in labels)

    if expect.get("has_positive_sentiment"):
        labels = [
            s.get("label", "") if isinstance(s, dict) else str(s)
            for s in (report.sentiment_results or [])
        ]
        assert any("pos" in str(lbl).lower() for lbl in labels)


@pytest.mark.usefixtures("mock_heavy_backends")
@pytest.mark.parametrize("fixture_path", GOLDEN_FIXTURES, ids=[p.stem for p in GOLDEN_FIXTURES])
def test_callcenter_golden_pipeline_mocked(fixture_path: Path) -> None:
    data = _load_fixture(fixture_path)
    pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=False)
    report = pipe.analyze_segments(data["segments"])
    _assert_golden_expectations(fixture_path, report)


@pytest.mark.slow
def test_callcenter_golden_billing_real_classifiers() -> None:
    """Unmocked integration: real heuristic intent + sentiment on billing fixture."""
    fixture_path = FIXTURE_DIR / "cc_billing_escalation.json"
    if not fixture_path.is_file():
        pytest.skip("billing golden fixture missing")
    data = _load_fixture(fixture_path)
    pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=False)
    report = pipe.analyze_segments(data["segments"])
    intents = [
        item.get("intent") if isinstance(item, dict) else item[0]
        for item in (report.intent_results or [])
    ]
    assert "billing_inquiry" in intents
    labels = [
        s.get("label", "") if isinstance(s, dict) else str(s)
        for s in (report.sentiment_results or [])
    ]
    assert any("neg" in str(lbl).lower() for lbl in labels)
