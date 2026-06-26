"""Unit tests for llm_judge_panel component.

Focus on pure helper functions + basic render smoke tests.
"""

from __future__ import annotations

import pytest

# We import the helpers directly (they are pure and easy to test)
from app.nicegui_dashboard.components.llm_judge_panel import (
    _get_verdicts,
    _is_changed,
    render_llm_judge_panel,
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


class TestRenderLLMJudgePanel:
    """Basic smoke tests for the render function."""

    def test_renders_without_error_with_empty_data(self):
        # Should not raise
        render_llm_judge_panel(None)
        render_llm_judge_panel({})
        render_llm_judge_panel({"verdicts": []})

    def test_renders_without_error_with_normal_data(self):
        data = {
            "verdicts": [
                {
                    "segment_index": 2,
                    "original_sentiment": "neutral",
                    "original_confidence": 0.45,
                    "judge_label": "negative",
                    "judge_confidence": 0.82,
                    "reasoning": "Kund nämner frustration.",
                }
            ]
        }
        # Should not raise
        render_llm_judge_panel(data)

    def test_renders_without_error_with_changed_only_filter(self):
        # This test mainly checks that the component doesn't crash
        # when the internal filter logic runs.
        data = {
            "verdicts": [
                {"original_sentiment": "positive", "judge_label": "negative"},
                {"original_sentiment": "negative", "judge_label": "negative"},
            ]
        }
        render_llm_judge_panel(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])