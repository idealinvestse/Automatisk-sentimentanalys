"""Tests for Fas 4 KPI stubs in evaluate.py."""

from __future__ import annotations

from src.evaluate import (
    compute_alert_trigger_rate,
    compute_cache_hit_rate,
    compute_coaching_precision,
    compute_hot_topic_recall,
    compute_pii_redaction_coverage,
    compute_qa_score_consistency,
)


class TestFas4KPIs:
    def test_qa_score_consistency_empty(self):
        assert compute_qa_score_consistency([]) == {"agreement": 0.0, "n": 0}

    def test_qa_score_consistency_passed(self):
        qa = [
            {"passed": True, "risk_level": "low"},
            {"passed": True, "risk_level": "medium"},
            {"passed": False, "risk_level": "high"},
        ]
        result = compute_qa_score_consistency(qa)
        assert result["n"] == 3
        assert result["agreement"] == round(2 / 3, 3)

    def test_coaching_precision_empty(self):
        assert compute_coaching_precision([])["precision"] == 0.0

    def test_coaching_precision_with_human_labels(self):
        recs = [{"text": "a"}, {"text": "b"}]
        result = compute_coaching_precision(recs, human_judged_good=[True, False])
        assert result["precision"] == 0.5
        assert result["n"] == 2

    def test_coaching_precision_heuristic_evidence(self):
        recs = [{"evidence_spans": ["x"]}, {"text": "no evidence"}]
        result = compute_coaching_precision(recs)
        assert result["precision"] == 0.5
        assert result["note"] == "heuristic: has_evidence"

    def test_hot_topic_recall(self):
        agg = {"hot_topics": [{"topic": "Faktura"}, {"topic": "Support"}]}
        result = compute_hot_topic_recall(agg, ["faktura", "leverans"])
        assert result["recall"] == 0.5
        assert result["n_gold"] == 2

    def test_hot_topic_recall_no_gold(self):
        assert compute_hot_topic_recall({"hot_topics": []}, [])["recall"] == 0.0

    def test_pii_redaction_coverage_empty(self):
        assert compute_pii_redaction_coverage(None)["coverage"] == 0.0

    def test_pii_redaction_coverage_with_expected(self):
        log = {"events": [{"type": "email"}, {"type": "personnummer"}]}
        result = compute_pii_redaction_coverage(log, ["email", "phone"])
        assert result["coverage"] == 0.5
        assert result["n_events"] == 2

    def test_pii_redaction_coverage_no_expected(self):
        log = {"events": [{"type": "email"}]}
        result = compute_pii_redaction_coverage(log)
        assert result["coverage"] == 1.0

    def test_alert_trigger_rate(self):
        alerts = [{"severity": "high"}, {"severity": "medium"}, {"severity": "high"}]
        result = compute_alert_trigger_rate(alerts, total_calls=10)
        assert result["trigger_rate"] == 0.3
        assert result["by_severity"]["high"] == 2

    def test_alert_trigger_rate_zero_calls(self):
        assert compute_alert_trigger_rate([], 0)["trigger_rate"] == 0.0

    def test_cache_hit_rate(self):
        assert compute_cache_hit_rate(3, 10)["hit_rate"] == 0.3
        assert compute_cache_hit_rate(0, 0)["hit_rate"] == 0.0