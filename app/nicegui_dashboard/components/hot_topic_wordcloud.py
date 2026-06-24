"""Hot Topic Wordcloud / Treemap component.

Fas 3 viz (Proposal B) – Uses Plotly treemap (no extra wordcloud dep).
"""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state


def _extract_topics(report_or_reports: Any) -> list[dict[str, Any]]:
    """Extract topic list from single report or list of reports.

    Supports:
    - report["results"]["topics"]
    - report["topics"]
    - list[report] → aggregated top-N topics
    """
    topics: list[dict[str, Any]] = []

    def _add_topic(t: dict[str, Any]) -> None:
        word = str(t.get("word") or t.get("topic") or t.get("term") or "").strip()
        if not word:
            return
        weight = t.get("weight") or t.get("frequency") or t.get("count") or 1.0
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            weight_f = 1.0
        topics.append({"word": word, "weight": max(0.1, weight_f)})

    if isinstance(report_or_reports, list):
        # Aggregate across reports
        from collections import Counter

        counter: Counter[str] = Counter()
        for r in report_or_reports:
            inner = (r or {}).get("results", {}) or {}
            raw_topics = inner.get("topics") or r.get("topics") or []
            for t in raw_topics if isinstance(raw_topics, list) else []:
                if isinstance(t, dict):
                    w = str(t.get("word") or t.get("topic") or "").strip().lower()
                    if w:
                        wt = t.get("weight") or t.get("frequency") or 1
                        try:
                            counter[w] += float(wt)
                        except (TypeError, ValueError):
                            counter[w] += 1
        for word, total in counter.most_common(25):
            topics.append({"word": word.title(), "weight": float(total)})
        return topics

    # Single report
    if isinstance(report_or_reports, dict):
        inner = report_or_reports.get("results", {}) or {}
        raw = inner.get("topics") or report_or_reports.get("topics") or []
        for t in raw if isinstance(raw, list) else []:
            if isinstance(t, dict):
                _add_topic(t)

    return topics


def _build_demo_topics() -> list[dict[str, Any]]:
    """Demo topics for headless mode."""
    return [
        {"word": "Pris", "weight": 18},
        {"word": "Leverans", "weight": 14},
        {"word": "Support", "weight": 12},
        {"word": "Kvalitet", "weight": 9},
        {"word": "Retur", "weight": 7},
        {"word": "Reklamation", "weight": 6},
        {"word": "Faktura", "weight": 5},
        {"word": "Teknik", "weight": 4},
    ]


def build_hot_topics_treemap(reports: list[dict[str, Any]] | dict[str, Any] | None) -> go.Figure:
    """Build Plotly treemap from topics data.

    Args:
        reports: Single report dict, list of reports, or None (uses demo data).

    Returns:
        Plotly treemap figure.
    """
    topics: list[dict[str, Any]] = []

    if reports is None:
        topics = _build_demo_topics()
    elif isinstance(reports, list) or isinstance(reports, dict):
        topics = _extract_topics(reports)

    if not topics:
        topics = _build_demo_topics()

    words = [t["word"] for t in topics]
    weights = [t["weight"] for t in topics]

    # Treemap: root → topics (size by weight, color by weight)
    fig = px.treemap(
        names=words,
        parents=[""] * len(words),
        values=weights,
        title="Heta ämnen (treemap)",
        color=weights,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_traces(
        hovertemplate="%{label}<br>Frekvens: %{value}<extra></extra>",
        textinfo="label+value",
    )
    return fig


def render_hot_topics_wordcloud(reports: list[dict[str, Any]] | dict[str, Any] | None) -> None:
    """Render treemap or empty state.

    Args:
        reports: Report(s) to extract topics from.
    """
    topics: list[dict[str, Any]] = []
    if reports is None:
        topics = _build_demo_topics()
    elif isinstance(reports, list) or isinstance(reports, dict):
        topics = _extract_topics(reports)

    if not topics:
        render_empty_state(
            icon="topic",
            title="Inga ämnen",
            hint="Inga topics extraherade från rapporter.",
        )
        return

    fig = build_hot_topics_treemap(reports)
    plot = ui.plotly(fig).classes("w-full")
    plot.props("style='height: 260px'")
