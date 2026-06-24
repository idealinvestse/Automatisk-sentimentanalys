"""Dashboard Fas 3 viz tests (B.5) – mocked data, no real API.

Covers emotion_timeline, hot_topic_wordcloud, llm_judge_breakdown helpers + render paths.
"""

from __future__ import annotations

from typing import Any

# --- emotion_timeline helpers ---
from app.nicegui_dashboard.components.emotion_timeline import (
    _extract_emotion_series,
    build_emotion_timeline_figure,
)

# --- hot_topic_wordcloud helpers ---
from app.nicegui_dashboard.components.hot_topic_wordcloud import (
    _extract_topics,
    build_hot_topics_treemap,
)

# --- llm_judge_breakdown helpers ---
from app.nicegui_dashboard.components.llm_judge_breakdown import (
    _extract_llm_judge_verdicts,
    _get_confidence_color,
)


def test_emotion_timeline_renders() -> None:
    """Component helper produces Plotly figure with demo data."""
    fig = build_emotion_timeline_figure(None)
    assert fig is not None
    assert len(fig.data) >= 1  # at least one trace from demo emotions


def test_emotion_timeline_empty_state() -> None:
    """Empty segments → figure still renders (demo fallback)."""
    report: dict[str, Any] = {"segments": []}
    indices, emo = _extract_emotion_series(report)
    assert indices == []
    assert emo == {}
    fig = build_emotion_timeline_figure(report)
    assert fig is not None


def test_wordcloud_renders() -> None:
    """Treemap renders from demo or report topics."""
    fig = build_hot_topics_treemap(None)
    assert fig is not None
    assert "treemap" in str(type(fig.data[0])).lower()


def test_wordcloud_empty_state() -> None:
    """Empty topics list → still produces demo treemap."""
    report: dict[str, Any] = {"results": {"topics": []}}
    topics = _extract_topics(report)
    assert isinstance(topics, list)
    fig = build_hot_topics_treemap(report)
    assert fig is not None


def test_llm_judge_breakdown_renders() -> None:
    """Helper functions exist and return sane values."""
    color = _get_confidence_color(0.85)
    assert color in ("green", "orange", "red", "grey")


def test_llm_judge_breakdown_empty_state() -> None:
    """triggered_segments=0 → empty verdict list."""
    report: dict[str, Any] = {"results": {"llm_judge_triggered_segments": 0}}
    verdicts = _extract_llm_judge_verdicts(report)
    assert verdicts == []


def test_color_confidence_coding() -> None:
    """Confidence → color mapping contract."""
    assert _get_confidence_color(0.9) == "green"
    assert _get_confidence_color(0.55) == "orange"
    assert _get_confidence_color(0.2) == "red"
    assert _get_confidence_color(None) == "grey"


def test_demo_data_fallback() -> None:
    """All helpers produce non-crashing output with None / empty input."""
    assert build_emotion_timeline_figure(None) is not None
    assert build_hot_topics_treemap(None) is not None
    assert _extract_llm_judge_verdicts(None) == []
