"""Overview dashboard widgets (KPIs, trends, tables)."""

from __future__ import annotations

from typing import Any


def render_kpi_cards(reports: list[dict[str, Any]], filters: dict[str, Any]) -> dict[str, Any]:
    try:
        import streamlit as st

        cols = st.columns(4)
        total = len(reports)
        metrics = [
            ("Samtal", str(total)),
            ("Positiva", str(sum(1 for r in reports if r.get("overall_sentiment") == "positiv"))),
            ("Negativa", str(sum(1 for r in reports if r.get("overall_sentiment") == "negativ"))),
            ("Alerts", str(sum(len(r.get("results", {}).get("alerts", [])) for r in reports))),
        ]
        for col, (label, val) in zip(cols, metrics, strict=False):
            with col:
                st.metric(label, val)
    except Exception:
        pass
    return dict(filters)


def render_sentiment_trend(reports: list[dict[str, Any]], *, key: str = "trend", **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.markdown("#### Sentiment-trend")
        if not reports:
            st.caption("Ingen data.")
    except Exception:
        pass


def render_hot_topics(
    reports: list[dict[str, Any]],
    filters: dict[str, Any],
    *,
    key: str = "hot",
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        import streamlit as st

        st.markdown("#### Hot topics")
    except Exception:
        pass
    return dict(filters)


def render_agent_leaderboard(
    reports: list[dict[str, Any]],
    filters: dict[str, Any],
    *,
    key: str = "board",
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        import streamlit as st

        st.markdown("#### Agent leaderboard")
    except Exception:
        pass
    return dict(filters)


def render_alerts_panel(reports: list[dict[str, Any]], *, key: str = "alerts", **kwargs: Any) -> None:
    try:
        import streamlit as st

        st.markdown("#### Alerts")
    except Exception:
        pass


def render_filtered_calls_table(
    reports: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    *,
    key: str = "table",
    **kwargs: Any,
) -> str | None:
    try:
        import streamlit as st

        rows = filtered or reports
        if not rows:
            st.caption("Inga samtal.")
            return None
        options = [r.get("call_id") or r.get("id") or f"call_{i}" for i, r in enumerate(rows)]
        choice = st.selectbox("Öppna samtal", options, key=f"open_call_{key}")
        if st.button("Öppna detalj", key=f"open_btn_{key}"):
            return choice
    except Exception:
        pass
    return None
