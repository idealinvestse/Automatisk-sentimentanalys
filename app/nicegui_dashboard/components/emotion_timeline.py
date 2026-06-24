"""Emotion Timeline component – Plotly line/area chart per segment.

Fas 3 viz (Proposal B) – Dashboard emotion scores over time.
"""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state


def _extract_emotion_series(report: dict[str, Any] | None) -> tuple[list[int], dict[str, list[float]]]:
    """Extract segment indices + emotion score dict from report segments.

    Returns:
        (segment_indices, emotion_dict) where emotion_dict keys are emotion names.
    """
    if not report:
        return [], {}

    segments: list[dict[str, Any]] = report.get("segments", []) or []
    if not segments:
        return [], {}

    segment_indices: list[int] = []
    emotion_data: dict[str, list[float]] = {}

    for idx, seg in enumerate(segments):
        emotions = seg.get("emotion") or {}
        if not isinstance(emotions, dict) or not emotions:
            continue

        segment_indices.append(idx)
        for emotion_name, score in emotions.items():
            if emotion_name not in emotion_data:
                emotion_data[emotion_name] = []
            # Ensure score is numeric (graceful fallback)
            try:
                emotion_data[emotion_name].append(float(score))
            except (TypeError, ValueError):
                emotion_data[emotion_name].append(0.0)

    return segment_indices, emotion_data


def _build_demo_emotion_data() -> tuple[list[int], dict[str, list[float]]]:
    """Fallback demo data for headless mode (5 segments, 4 emotions)."""
    indices = [0, 1, 2, 3, 4]
    return indices, {
        "joy": [0.65, 0.72, 0.58, 0.81, 0.69],
        "sadness": [0.12, 0.08, 0.21, 0.05, 0.15],
        "anger": [0.08, 0.15, 0.31, 0.09, 0.12],
        "fear": [0.15, 0.05, 0.10, 0.05, 0.04],
    }


def build_emotion_timeline_figure(report: dict[str, Any] | None) -> go.Figure:
    """Build Plotly line chart for emotion scores over segments.

    Args:
        report: Call report dict with 'segments' list (each may have 'emotion' dict).

    Returns:
        Plotly Figure (line chart with multiple emotion traces).
    """
    indices, emotion_dict = _extract_emotion_series(report)
    if not indices or not emotion_dict:
        indices, emotion_dict = _build_demo_emotion_data()

    if not indices:
        # Absolute fallback – should never reach here
        fig = go.Figure()
        fig.add_annotation(text="Ingen emotion-data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Build DataFrame for px.line (long form)
    records: list[dict[str, Any]] = []
    for emotion_name, scores in emotion_dict.items():
        for i, score in zip(indices, scores):
            records.append({"segment": i, "emotion": emotion_name, "score": score})

    if not records:
        fig = go.Figure()
        fig.add_annotation(text="Ingen emotion-data", x=0.5, y=0.5, showarrow=False)
        return fig

    import pandas as pd  # Local import – dashboard env has pandas via nicegui/plotly stack

    df = pd.DataFrame(records)
    fig = px.line(
        df,
        x="segment",
        y="score",
        color="emotion",
        markers=True,
        title="Emotion scores per segment",
        labels={"segment": "Segment index", "score": "Score (0-1)", "emotion": "Emotion"},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="Segment %{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>")
    return fig


def render_emotion_timeline(report: dict[str, Any] | None) -> None:
    """Render emotion timeline chart or empty state.

    Args:
        report: Selected call report (may be None or lack segments).
    """
    indices, emotion_dict = _extract_emotion_series(report)
    if not indices or not emotion_dict:
        render_empty_state(
            icon="timeline",
            title="Ingen emotion-data",
            hint="Välj ett samtal med emotion-analys eller använd demo-data.",
        )
        return

    fig = build_emotion_timeline_figure(report)
    plot = ui.plotly(fig).classes("w-full")
    plot.props("style='height: 280px'")
