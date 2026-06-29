"""Unit tests for llm_judge_panel component.

Focus on pure helper functions + basic render smoke tests.
Expanded coverage for TASK-05.
"""

from __future__ import annotations

import pytest

# We import the helpers directly (they are pure and easy to test)
from app.nicegui_dashboard.components.llm_judge_panel import (
    _get_verdicts,
    _is_changed,
    render_llm_judge_summary,
)


class TestGetVerdicts:
    """Tests for _get_verdicts helper."""

    def test_returns_empty_list_for_none(self):
        assert _get_verdicts(None) == []

    def test_returns_empty_list_for_empty_dict(self):
        assert _get_verdicts({}) == []

    def test_extracts_from_verdicts_key(self):
        data = {"verdicts": [{"judge_label": "negative"}]}
        assert _get_verdicts(data) == [{"judge_label": "negative"}]

    def test_extracts_from_results_key(self):
        data = {"results": [{"judge_label": "positive"}]}
        assert _get_verdicts(data) == [{"judge_label": "positive"}]

    def test_returns_list_directly(self):
        data = [{"judge_label": "neutral"}]
        assert _get_verdicts(data) == [{"judge_label": "neutral"}]

    def test_ignores_unknown_structure(self):
        assert _get_verdicts({"foo": "bar"}) == []
        assert _get_verdicts("not a dict") == []


class TestIsChanged:
    """Tests for _is_changed helper (core filter logic)."""

    def test_returns_false_when_same_label(self):
        v = {"original_sentiment": "negative", "judge_label": "negative"}
        assert _is_changed(v) is False

    def test_returns_true_when_label_changes(self):
        v = {"original_sentiment": "neutral", "judge_label": "negative"}
        assert _is_changed(v) is True

    def test_handles_different_key_names(self):
        v = {"original_label": "positive", "label": "negative"}
        assert _is_changed(v) is True

    def test_returns_false_for_empty_judge_label(self):
        v = {"original_sentiment": "neutral", "judge_label": ""}
        assert _is_changed(v) is False

    def test_is_case_insensitive(self):
        v = {"original_sentiment": "Negative", "judge_label": "negative"}
        assert _is_changed(v) is False

    def test_returns_false_when_judge_label_missing(self):
        v = {"original_sentiment": "neutral"}
        assert _is_changed(v) is False

    def test_returns_false_when_original_missing(self):
        v = {"judge_label": "positive"}
        assert _is_changed(v) is False


class TestRenderLLMJudgeSummary:
    """Tests for the compact summary badge helper (pure logic)."""

    def test_returns_none_for_empty_data(self):
        assert render_llm_judge_summary(None) is None
        assert render_llm_judge_summary({}) is None
        assert render_llm_judge_summary({"verdicts": []}) is None

    def test_filter_changed_only_logic(self):
        """Test filter counts for 'Endast ändrade' mode."""
        verdicts = [
            {"original_sentiment": "positive", "judge_label": "negative"},
            {"original_sentiment": "negative", "judge_label": "negative"},
            {"original_sentiment": "neutral", "judge_label": "positive"},
            {"original_sentiment": "positive", "judge_label": "positive"},
        ]

        changed_count = sum(1 for v in verdicts if _is_changed(v))
        assert changed_count == 2

        filtered = [v for v in verdicts if _is_changed(v)]
        assert len(filtered) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
